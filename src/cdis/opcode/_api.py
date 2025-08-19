from dataclasses import dataclass, replace
from abc import ABC, abstractmethod
from functools import cached_property
from inspect import Signature, Parameter
import ast
from enum import Enum
import types
import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._compiler import Bytecode
    from .._vm import Frame


@dataclass
class Label:
    """Represents a label that can be jumped to in the bytecode.

    Used to implement if statements, loops and exception handlers.

    Attributes
    ----------
    index
    """

    _bytecode_index: int | None = None

    @property
    def index(self) -> int:
        """The position in the bytecode to jump to."""
        if self._bytecode_index is None:
            raise ValueError("Label was not initialized during compilation")
        return self._bytecode_index


@dataclass(frozen=True)
class ValueSource:
    """The source of a value in the bytecode."""

    sources: tuple["Instruction", ...]
    """The possible instructions that can produce this value."""

    value_type: type
    """The type of the value."""

    def unify_with(self, other: "ValueSource") -> "ValueSource":
        """Unify this value source with another value source."""
        return ValueSource(
            sources=tuple(set(self.sources) | set(other.sources)),
            value_type=_unify_types(self.value_type, other.value_type),
        )

    def __eq__(self, other):
        if isinstance(other, ValueSource):
            return self.value_type == other.value_type
        else:
            return False

    def __hash__(self):
        return hash(self.value_type)


def _find_closest_common_ancestor(*cls_list: type) -> type:
    from collections import defaultdict

    mros = [
        (list(cls.__mro__) if hasattr(cls, "__mro__") else [cls]) for cls in cls_list
    ]
    track = defaultdict(int)
    while mros:
        for mro in mros:
            cur = mro.pop(0)
            track[cur] += 1
            if track[cur] == len(cls_list):
                return cur
            if len(mro) == 0:
                mros.remove(mro)
    return object


def _unify_types(a: type, b: type) -> type:
    from typing import get_origin

    while get_origin(a) is not None:
        a = get_origin(a)
    while get_origin(b) is not None:
        b = get_origin(b)

    if issubclass(a, b):
        return b
    elif issubclass(b, a):
        return a
    return _find_closest_common_ancestor(a, b)


@dataclass(frozen=True)
class StackMetadata:
    """Represents the state of the stack for a given bytecode instruction."""

    stack: tuple[ValueSource, ...]
    """The values on the stack when the instruction is executed."""

    variables: dict[str, ValueSource]
    """The variables value sources when the instruction is executed."""

    synthetic_variables: tuple[ValueSource, ...]
    """The synthetic variables value sources when the instruction is executed."""

    dead: bool = False
    """True if the code is unreachable, False otherwise."""

    @classmethod
    def dead_code(cls) -> "StackMetadata":
        return cls(stack=(), variables={}, synthetic_variables=(), dead=True)

    def unify_with(self, other: "StackMetadata") -> "StackMetadata":
        if other.dead:
            return self
        if self.dead:
            return other

        if len(self.stack) != len(other.stack):
            raise ValueError("Stack size mismatch")

        new_stack = tuple(
            self.stack[index].unify_with(other.stack[index])
            for index in range(len(self.stack))
        )
        new_variables = {}
        own_keys = self.variables.keys() - other.variables.keys()
        their_keys = other.variables.keys() - self.variables.keys()
        shared_keys = self.variables.keys() & other.variables.keys()

        for key in own_keys:
            new_variables[key] = self.variables[key]
        for key in their_keys:
            new_variables[key] = other.variables[key]
        for key in shared_keys:
            new_variables[key] = self.variables[key].unify_with(other.variables[key])

        shared_synthetics = min(
            len(self.synthetic_variables), len(other.synthetic_variables)
        )
        unshared_synthetics = ()
        if len(self.synthetic_variables) > len(other.synthetic_variables):
            unshared_synthetics = self.synthetic_variables[shared_synthetics:]
        elif len(self.synthetic_variables) < len(other.synthetic_variables):
            unshared_synthetics = self.synthetic_variables[shared_synthetics:]

        new_synthetic_variables = (
            tuple(
                self.synthetic_variables[index].unify_with(
                    other.synthetic_variables[index]
                )
                for index in range(shared_synthetics)
            )
            + unshared_synthetics
        )

        return StackMetadata(
            stack=new_stack,
            variables=new_variables,
            synthetic_variables=new_synthetic_variables,
            dead=False,
        )

    def pop(self, count: int):
        """Pops the given number of values from the stack."""
        return replace(self, stack=self.stack[:-count])

    def push(self, *values: ValueSource) -> "StackMetadata":
        """Pushes the given values to the stack."""
        return replace(self, stack=self.stack + values)

    def set_variable(self, name: str, value: ValueSource) -> "StackMetadata":
        """Sets the value source for the given variable."""
        new_variables = dict(self.variables)
        value_sources = self.variables.get(name, None)
        if value_sources is None:
            new_variables[name] = value
        else:
            new_variables[name] = ValueSource(
                sources=(*value_sources.sources, *value.sources),
                value_type=_unify_types(value.value_type, value_sources.value_type),
            )
        return replace(self, variables=new_variables)

    def new_synthetic(self, value: ValueSource) -> "StackMetadata":
        """Creates a new synthetic variable with the given source."""
        return replace(self, synthetic_variables=self.synthetic_variables + (value,))

    def pop_synthetic(self) -> "StackMetadata":
        """Pops the last created synthetic variable."""
        return replace(self, synthetic_variables=self.synthetic_variables[:-1])

    def set_synthetic(self, index: int, value: ValueSource) -> "StackMetadata":
        """Sets the value source for the given synthetic variable."""
        if index >= len(self.synthetic_variables):
            return self.new_synthetic(value)
        else:
            return replace(
                self,
                synthetic_variables=self.synthetic_variables[:index]
                + (value,)
                + self.synthetic_variables[index + 1 :],
            )

    def __eq__(self, other):
        if self.dead != other.dead:
            return False
        if self.stack != other.stack:
            return False
        if self.variables != other.variables:
            return False
        if self.synthetic_variables != other.synthetic_variables:
            return False
        return True
        return (
            self.dead == other.dead
            and self.stack == other.stack
            and self.variables == other.variables
            and self.synthetic_variables == other.synthetic_variables
        )

    def __hash__(self):
        return hash((self.dead, self.stack, self.variables, self.synthetic_variables))


@dataclass(frozen=True)
class InnerFunction:
    """Represents a function defined inside another function."""

    bytecode: "Bytecode"
    """The bytecode of the body of the inner function."""
    annotate_function: "Bytecode"
    """A function to compute the annotation dict, as defined by https://peps.python.org/pep-0649/"""
    parameters_with_defaults: tuple[str, ...]
    """Parameters with default values."""

    @property
    def name(self) -> str:
        """The name of the function; possibly empty for lambdas."""
        return self.bytecode.function_name

    @property
    def signature(self) -> Signature:
        """The signature of the function."""
        return self.bytecode.signature

    @property
    def free_vars(self) -> frozenset[str]:
        """The free variables that must be bound before the function can be called."""
        return self.bytecode.free_names

    def bind(self, frame: "Frame", *default_values) -> "Bytecode":
        """Binds this inner function to the given frame."""
        if len(default_values) != len(self.parameters_with_defaults):
            raise ValueError(
                f"Expected {len(self.parameters_with_defaults)} default values, got {len(default_values)}."
            )
        new_closure = {}
        for free_var in self.free_vars:
            new_closure[free_var] = frame.closure[free_var]

        original_signature = self.signature
        new_parameters = []
        for parameter_name, parameter in original_signature.parameters.items():
            if parameter_name in self.parameters_with_defaults:
                index = self.parameters_with_defaults.index(parameter_name)
                new_parameters.append(
                    Parameter(
                        name=parameter_name,
                        kind=parameter.kind,
                        default=default_values[index],
                        annotation=parameter.annotation,
                    )
                )
            else:
                new_parameters.append(parameter)

        new_signature = original_signature.replace(parameters=new_parameters)
        new_annotate_function = replace(self.annotate_function, closure=new_closure)
        return replace(
            self.bytecode,
            annotate_function=new_annotate_function,
            signature=new_signature,
            closure=new_closure,
        )


class Opcode(ABC):
    """Represent a bytecode operation.

    Notes
    -----
    Each Opcode's Notes section will detail the state of the stack prior to
    and after the opcode like this:

        | Opcode
        | Stack Effect: +1
        | Prior: ..., a, b
        | After: ..., c, d, e

    When the same identifier is used in both prior and after, it represents
    the same, identical value. For instance, the `Dup` Opcode stack effect:

        | Dup
        | Stack Effect: +1
        | Prior: ..., value
        | After: ..., value, value

    `value` is repeated in after, meaning it a duplicate of the value from prior.
    """

    @abstractmethod
    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        """Computes the stack metadata for each index given by `next_bytecode_indices`."""
        ...

    @abstractmethod
    def execute(self, frame: "Frame") -> None:
        """Executes the bytecode operation on the given frame."""
        ...

    def next_bytecode_indices(self, instruction: "Instruction") -> tuple[int, ...]:
        """Returns the possible next bytecode indices for the given instruction."""
        return (instruction.bytecode_index + 1,)


@dataclass(frozen=True)
class Instruction:
    """An instruction in the bytecode."""

    opcode: Opcode
    """The operation this instruction performs.
    """

    bytecode_index: int
    """The bytecode index of this instruction.
    """

    lineno: int
    """The source line number of this instruction.
    """

    def __eq__(self, other: "Instruction") -> bool:
        return (
            isinstance(other, Instruction)
            and self.bytecode_index == other.bytecode_index
        )

    def __hash__(self) -> int:
        return hash(self.bytecode_index)


class FunctionType(Enum):
    FUNCTION = "function"
    GENERATOR = "generator"
    CLASS_BODY = "class_body"
    COROUTINE_GENERATOR = "coroutine_generator"
    ASYNC_FUNCTION = "async_function"
    ASYNC_GENERATOR = "async_generator"

    @staticmethod
    def for_function(function: types.FunctionType) -> "FunctionType":
        if inspect.isasyncgenfunction(function):
            return FunctionType.ASYNC_GENERATOR
        elif inspect.isgeneratorfunction(function):
            if function.__code__.co_flags & inspect.CO_ITERABLE_COROUTINE:
                return FunctionType.COROUTINE_GENERATOR
            else:
                return FunctionType.GENERATOR
        elif inspect.iscoroutinefunction(function):
            return FunctionType.ASYNC_FUNCTION
        else:
            return FunctionType.FUNCTION

    @staticmethod
    def for_function_ast(
        function_ast: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    ) -> "FunctionType":
        from .._compiler import is_generator

        match function_ast:
            case ast.ClassDef():
                return FunctionType.CLASS_BODY
            case ast.FunctionDef():
                if is_generator(function_ast):
                    # TODO: determine if it a coroutine generator from decorator_list
                    return FunctionType.GENERATOR
                return FunctionType.FUNCTION
            case ast.AsyncFunctionDef():
                return (
                    FunctionType.ASYNC_GENERATOR
                    if is_generator(function_ast)
                    else FunctionType.ASYNC_FUNCTION
                )
            case _:
                raise ValueError("Expected ast.FunctionDef or ast.AsyncFunctionDef")


class MethodType(Enum):
    VIRTUAL = "virtual"
    STATIC = "static"
    CLASS = "class"

    @staticmethod
    def for_function(function: types.FunctionType) -> "MethodType":
        if isinstance(function, classmethod):
            return MethodType.CLASS
        elif isinstance(function, staticmethod):
            return MethodType.STATIC
        else:
            return MethodType.VIRTUAL

    @staticmethod
    def for_function_ast(
        function_ast: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> "MethodType":
        # Note: this does assume people don't do something like
        # have another decorator call classmethod indirectly or
        # override classmethod/staticmethod
        for decorator in function_ast.decorator_list:
            match decorator:
                case ast.Name(id=name):
                    match name:
                        case "classmethod":
                            return MethodType.CLASS
                        case "staticmethod":
                            return MethodType.STATIC
                        case _:
                            pass
                case _:
                    pass
        else:
            return MethodType.VIRTUAL


@dataclass(frozen=True)
class ClassInfo:
    """Information about a class that is already constructed.

    This does not contain information about the bytecode that was run
    to generate the class.
    """

    name: str
    """The name of the class."""
    qualname: str
    """The qualified name of the class."""
    class_attributes: dict[str, type]
    """Attributes that are defined on the class."""
    instance_attributes: dict[str, type]
    """Attributes that are defined on the instance."""
    class_attribute_defaults: dict[str, object]
    """The values of the class attributes when this ClassInfo was created."""

    @cached_property
    def methods(self) -> dict[str, "Bytecode"]:
        from .._compiler import Bytecode

        out: dict[str, Bytecode] = {}

        for method_name, method in self.class_attribute_defaults.items():
            if isinstance(method, Bytecode):
                out[method_name] = method

        return out

    def as_class(self, *, vm=None, **vm_kwargs) -> type:
        from .._compiler import Bytecode
        from .._vm import CDisVM

        class Out:
            def __init__(self, *args, **kwargs):
                pass

        if vm is None:
            vm = CDisVM()

        Out.__name__ = self.name
        Out.__qualname__ = self.qualname

        for attribute_name, attribute_value in self.class_attribute_defaults.items():
            if isinstance(attribute_value, Bytecode):

                def _run(
                    owner, *args, _method_bytecode: Bytecode = attribute_value, **kwargs
                ):
                    return vm.run(_method_bytecode, owner, *args, **kwargs, **vm_kwargs)

                setattr(Out, attribute_name, _run)
            else:
                setattr(Out, attribute_name, attribute_value)

        def instance_getter(*vm_args, **vm_kwargs):
            def outer(*args, **kwargs):
                return Out(*args, **kwargs)

            return outer

        return Out


__all__ = (
    "Label",
    "ValueSource",
    "StackMetadata",
    "InnerFunction",
    "Opcode",
    "Instruction",
    "FunctionType",
    "MethodType",
    "ClassInfo",
)
