import enum
import types
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, replace, field
from abc import ABC, abstractmethod
from functools import cached_property
from inspect import Signature, Parameter
from typing import TYPE_CHECKING, Iterator, Any, Union, cast
from enum import Enum
import ast
import operator


if TYPE_CHECKING:
    from ._vm import Frame, CDisVM
    from ._compiler import Bytecode


class OperatorType(Enum):
    """Represent if an operator is a unary, binary or comparison operator."""

    UNARY_OPERATION = ast.unaryop
    """A unary operator, such as negation (-x) or invert (~x).
    """

    BINARY_OPERATION = ast.operator
    """A binary operator, such as (x + y) or (x - y)."""

    COMPARISON = ast.Eq | ast.NotEq | ast.Lt | ast.Gt | ast.LtE | ast.GtE
    """A comparison operator, such as (x < y), (x == y)."""


@dataclass(frozen=True)
class Operator:
    """A unary, binary or comparison operator."""

    ast: ast.operator | ast.cmpop | ast.unaryop
    """The AST node corresponding to the operator.
    """

    type: OperatorType
    """The type of the operator (unary, binary or comparison).
    """

    function: Callable
    """A function that can be called to perform the operation.
    """

    dunder_name: str
    """The dunder method corresponding to the operation from the left side.
    
    For example, for (x + y), this would be "__add__".
    """

    flipped_dunder_name: str | None = None
    """The dunder method corresponding to the operation from the right side.
    
    For example, for (x + y), this would be "__radd__".
    
    Optional.
    """


class UnaryOperator(Enum):
    """A unary operator, such as negation (-x) or invert (~x)."""

    Invert = Operator(
        ast.Invert(), OperatorType.UNARY_OPERATION, operator.invert, "__invert__"
    )
    """The inversion operator (~x).
    """

    UAdd = Operator(ast.UAdd(), OperatorType.UNARY_OPERATION, operator.pos, "__pos__")
    """The positive operator (+x).
    """

    USub = Operator(ast.USub(), OperatorType.UNARY_OPERATION, operator.neg, "__neg__")
    """The negation operator (-x).
    """


class BinaryOperator(Enum):
    """A binary or comparison operator, such as (x + y) or (x < y)."""

    Add = Operator(
        ast.Add(), OperatorType.BINARY_OPERATION, operator.add, "__add__", "__radd__"
    )
    """The addition operator (x + y)."""

    Sub = Operator(
        ast.Sub(), OperatorType.BINARY_OPERATION, operator.sub, "__sub__", "__rsub__"
    )
    """The subtraction operator (x - y)."""

    Mult = Operator(
        ast.Mult(), OperatorType.BINARY_OPERATION, operator.mul, "__mul__", "__rmul__"
    )
    """The multiplication operator (x * y)."""

    Div = Operator(
        ast.Div(),
        OperatorType.BINARY_OPERATION,
        operator.truediv,
        "__truediv__",
        "__rtruediv__",
    )
    """The true division operator (x / y)."""

    FloorDiv = Operator(
        ast.FloorDiv(),
        OperatorType.BINARY_OPERATION,
        operator.floordiv,
        "__floordiv__",
        "__rfloordiv__",
    )
    """The floor division operator (x // y)."""

    Mod = Operator(
        ast.Mod(), OperatorType.BINARY_OPERATION, operator.mod, "__mod__", "__rmod__"
    )
    """The moduli operator (x % y)."""

    Pow = Operator(
        ast.Pow(), OperatorType.BINARY_OPERATION, operator.pow, "__pow__", "__rpow__"
    )
    """The power operator (x ** y)."""

    LShift = Operator(
        ast.LShift(),
        OperatorType.BINARY_OPERATION,
        operator.lshift,
        "__lshift__",
        "__rlshift__",
    )
    """The left shift operator (x << y)."""

    RShift = Operator(
        ast.RShift(),
        OperatorType.BINARY_OPERATION,
        operator.rshift,
        "__rshift__",
        "__rrshift__",
    )
    """The right shift operator (x >> y)."""

    BitOr = Operator(
        ast.BitOr(), OperatorType.BINARY_OPERATION, operator.or_, "__or__", "__ror__"
    )
    """The bitwise or operator (x | y)."""

    BitXor = Operator(
        ast.BitXor(), OperatorType.BINARY_OPERATION, operator.xor, "__xor__", "__rxor__"
    )
    """The bitwise xor operator (x ^ y)."""

    BitAnd = Operator(
        ast.BitAnd(),
        OperatorType.BINARY_OPERATION,
        operator.and_,
        "__and__",
        "__rand__",
    )
    """The bitwise and operator (x & y)."""

    MatMult = Operator(
        ast.MatMult(),
        OperatorType.BINARY_OPERATION,
        operator.matmul,
        "__matmul__",
        "__rmatmul__",
    )
    """The matrix multiplication operator (x @ y)."""

    # Inplace
    IAdd = Operator(
        ast.Add(), OperatorType.BINARY_OPERATION, operator.add, "__iadd__", "__radd__"
    )
    """The inplace addition operator (x += y)."""

    ISub = Operator(
        ast.Sub(), OperatorType.BINARY_OPERATION, operator.sub, "__isub__", "__rsub__"
    )
    """The inplace subtraction operator (x -= y)."""

    IMult = Operator(
        ast.Mult(), OperatorType.BINARY_OPERATION, operator.mul, "__imul__", "__rmul__"
    )
    """The inplace multiplication operator (x *= y)."""

    IDiv = Operator(
        ast.Div(),
        OperatorType.BINARY_OPERATION,
        operator.truediv,
        "__itruediv__",
        "__rtruediv__",
    )
    """The inplace true division operator (x /= y)."""

    IFloorDiv = Operator(
        ast.FloorDiv(),
        OperatorType.BINARY_OPERATION,
        operator.floordiv,
        "__ifloordiv__",
        "__rfloordiv__",
    )
    """The inplace floor division operator (x //= y)."""

    IMod = Operator(
        ast.Mod(), OperatorType.BINARY_OPERATION, operator.mod, "__imod__", "__rmod__"
    )
    """The inplace moduli operator (x %= y)."""

    IPow = Operator(
        ast.Pow(), OperatorType.BINARY_OPERATION, operator.pow, "__ipow__", "__rpow__"
    )
    """The inplace power operator (x **= y)."""

    ILShift = Operator(
        ast.LShift(),
        OperatorType.BINARY_OPERATION,
        operator.lshift,
        "__ilshift__",
        "__rlshift__",
    )
    """The inplace left shift operator (x <<= y)."""

    IRShift = Operator(
        ast.RShift(),
        OperatorType.BINARY_OPERATION,
        operator.rshift,
        "__irshift__",
        "__rrshift__",
    )
    """The inplace right shift operator (x >>= y)."""

    IBitOr = Operator(
        ast.BitOr(), OperatorType.BINARY_OPERATION, operator.or_, "__ior__", "__ror__"
    )
    """The inplace bitwise or operator (x |= y)."""

    IBitXor = Operator(
        ast.BitXor(),
        OperatorType.BINARY_OPERATION,
        operator.xor,
        "__ixor__",
        "__rxor__",
    )
    """The inplace bitwise xor operator (x ^= y)."""

    IBitAnd = Operator(
        ast.BitAnd(),
        OperatorType.BINARY_OPERATION,
        operator.and_,
        "__iand__",
        "__rand__",
    )
    """The inplace bitwise and operator (x &= y)."""

    IMatMult = Operator(
        ast.MatMult(),
        OperatorType.BINARY_OPERATION,
        operator.matmul,
        "__imatmul__",
        "__rmatmul__",
    )
    """The inplace matrix multiplication operator (x @= y)."""

    # Comparison operators
    Eq = Operator(ast.Eq(), OperatorType.COMPARISON, operator.eq, "__eq__", "__eq__")
    """The equality operator (x == y)."""

    NotEq = Operator(
        ast.NotEq(), OperatorType.COMPARISON, operator.ne, "__ne__", "__ne__"
    )
    """The inequality operator (x != y)."""

    Lt = Operator(ast.Lt(), OperatorType.COMPARISON, operator.lt, "__lt__", "__gt__")
    """The less than operator (x < y)."""

    LtE = Operator(ast.LtE(), OperatorType.COMPARISON, operator.le, "__le__", "__ge__")
    """The less than or equal operator (x <= y)."""

    Gt = Operator(ast.Gt(), OperatorType.COMPARISON, operator.gt, "__gt__", "__lt__")
    """The greater than operator (x > y)."""

    GtE = Operator(ast.GtE(), OperatorType.COMPARISON, operator.ge, "__ge__", "__le__")
    """The greater than or equal operator (x >= y)."""


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

    def unify_with(self, other: 'ValueSource') -> 'ValueSource':
        """Unify this value source with another value source."""
        return ValueSource(
            sources=tuple(set(self.sources) | set(other.sources)),
            value_type=_unify_types(self.value_type, other.value_type)
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
    mros = [(list(cls.__mro__) if hasattr(cls, '__mro__') else [cls]) for cls in cls_list]
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
    def dead_code(cls) -> 'StackMetadata':
        return cls(stack=(), variables={}, synthetic_variables=(), dead=True)

    def unify_with(self, other: 'StackMetadata') -> 'StackMetadata':
        if other.dead:
            return self
        if self.dead:
            return other

        if len(self.stack) != len(other.stack):
            raise ValueError('Stack size mismatch')

        new_stack = tuple(self.stack[index].unify_with(other.stack[index]) for index in range(len(self.stack)))
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

        shared_synthetics = min(len(self.synthetic_variables), len(other.synthetic_variables))
        unshared_synthetics = ()
        if len(self.synthetic_variables) > len(other.synthetic_variables):
            unshared_synthetics = self.synthetic_variables[shared_synthetics:]
        elif len(self.synthetic_variables) < len(other.synthetic_variables):
            unshared_synthetics = self.synthetic_variables[shared_synthetics:]

        new_synthetic_variables = tuple(self.synthetic_variables[index].unify_with(other.synthetic_variables[index])
                                        for index in range(shared_synthetics)) + unshared_synthetics

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
        return (self.dead == other.dead and
                self.stack == other.stack and
                self.variables == other.variables and
                self.synthetic_variables == other.synthetic_variables
        )

    def __hash__(self):
        return hash((self.dead, self.stack, self.variables, self.synthetic_variables))


@dataclass(frozen=True)
class InnerFunction:
    """Represents a function defined inside another function.

    """
    bytecode: 'Bytecode'
    parameters_with_defaults: tuple[str, ...]

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

    def bind(self, frame: 'Frame', *default_values) -> 'Bytecode':
        """Binds this inner function to the given frame."""
        if len(default_values) != len(self.parameters_with_defaults):
            raise ValueError(f'Expected {len(self.parameters_with_defaults)} default values, got {len(default_values)}.')
        new_closure = {}
        for free_var in self.free_vars:
            new_closure[free_var] = frame.closure[free_var]

        original_signature = self.signature
        new_parameters = []
        for parameter_name, parameter in original_signature.parameters.items():
            if parameter_name in self.parameters_with_defaults:
                index = self.parameters_with_defaults.index(parameter_name)
                new_parameters.append(Parameter(
                    name=parameter_name,
                    kind=parameter.kind,
                    default=default_values[index],
                    annotation=parameter.annotation
                ))
            else:
                new_parameters.append(parameter)

        new_signature = original_signature.replace(parameters=new_parameters)
        return replace(self.bytecode, signature=new_signature, closure=new_closure)



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


########################################
# Constants
########################################
@dataclass(frozen=True)
class LoadConstant(Opcode):
    """Loads a constant onto the stack.

    Notes
    -----
        | LoadConstant
        | Stack Effect: +1
        | Prior: ...
        | After: ..., constant

    Attributes
    ----------
    constant: object
        The constant to be loaded. For instance, an int, float or str.

    Examples
    --------
    >>> 1
    LoadConstant(constant=1)

    >>> "hello"
    LoadConstant(constant="hello")
    """

    constant: object

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.push(
            ValueSource((instruction,), type(self.constant))
        ),

    def execute(self, frame: "Frame") -> None:
        frame.stack.append(self.constant)


########################################
# Variables
########################################
def _determine_type(types: list[type]) -> type:
    from functools import reduce
    from operator import and_
    from collections import Counter

    return next(iter(reduce(and_, (Counter(cls.mro()) for cls in types))))


@dataclass(frozen=True)
class LoadGlobal(Opcode):
    """Loads a global variable or builtin onto the stack.

    Notes
    -----
        | LoadGlobal
        | Stack Effect: +1
        | Prior: ...
        | After: ..., global

    Attributes
    ----------
    name: str
        The name of the global variable or builtin.

    Examples
    --------
    >>> global x
    ... x
    LoadGlobal(name="x")

    >>> int
    LoadGlobal(name="int")
    """

    name: str

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.push(
            ValueSource((instruction,), type(bytecode.globals.get(self.name, object())))
        ),

    def execute(self, frame: "Frame") -> None:
        try:
            frame.stack.append(frame.globals[self.name])
        except KeyError:
            frame.stack.append(frame.vm.builtins[self.name])


@dataclass(frozen=True)
class LoadLocal(Opcode):
    """Loads a local variable onto the stack.

    The local variable is not a cell variable (a variable
    shared with another function) or a synthethic variable
    (a variable introduced by the compiler).

    Raises `UnboundLocalError` if the local variable is not defined
    yet.

    Notes
    -----
        | LoadLocal
        | Stack Effect: +1
        | Prior: ...
        | After: ..., local

    Attributes
    ----------
    name: str
        The name of the local variable.

    Examples
    --------
    >>> x
    LoadLocal(name="x")
    """

    name: str

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        local_metadata = previous_stack_metadata.variables.get(self.name,
                                                               ValueSource((), object))
        return previous_stack_metadata.push(
            local_metadata
        ),

    def execute(self, frame: "Frame") -> None:
        try:
            frame.stack.append(frame.variables[self.name])
        except KeyError:
            raise UnboundLocalError(
                f"local variable '{self.name}' referenced before assignment"
            )


@dataclass(frozen=True)
class LoadCell(Opcode):
    """Loads a cell variable onto the stack.

    A cell variable is a variable shared with another function.
    They are typically implemented by creating a holder object called
    a cell, then reading/modifying an attribute of the cell to read/set
    the variable.

    Raises `NameError` if the cell variable is a free variable and undefined,
    and `UnboundLocalError` if the cell variable is not defined and not a free variable.

    Notes
    -----
        | LoadCell
        | Stack Effect: +1
        | Prior: ...
        | After: ..., cell_value

    Attributes
    ----------
    name: str
        The name of the cell variable.

    Examples
    --------
    >>> nonlocal x
    ... x
    LoadCell(name="x")
    """

    name: str
    is_free: bool

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.push(
            ValueSource(
                (instruction,),
                object,  # TODO
            )
        ),

    def execute(self, frame: "Frame") -> None:
        try:
            frame.stack.append(frame.closure[self.name].cell_contents)
        except (ValueError, KeyError):
            if self.is_free:
                raise NameError(
                    f"free variable '{self.name}' referenced before assignment in enclosing scope"
                )
            else:
                raise UnboundLocalError(
                    f"local variable '{self.name}' referenced before assignment"
                )


@dataclass(frozen=True)
class LoadSynthetic(Opcode):
    """Loads a synthetic variable onto the stack.

    A synthetic variable is a variable introduced by the compiler and is not included
    in `locals()`.

    A synthetic variable is always defined before being loaded.

    Notes
    -----
        | LoadSynthetic
        | Stack Effect: +1
        | Prior: ...
        | After: ..., synthetic

    Attributes
    ----------
    index: int
        The index of the synthetic variable.

    Examples
    --------
    >>> for item in collection:
    ...     pass
    LoadLocal(name="collection")
    GetIterator()
    StoreSynthetic(index=0)

    label loop_start

    LoadSynthetic(index=0)
    GetNextElseJumpTo(target=loop_end)
    StoreLocal(name="item")
    JumpTo(target=loop_start)

    label loop_end
    """

    index: int

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        synthetic_metadata = previous_stack_metadata.synthetic_variables[self.index]
        return previous_stack_metadata.push(synthetic_metadata),

    def execute(self, frame: "Frame") -> None:
        frame.stack.append(frame.synthetic_variables[self.index])


@dataclass(frozen=True)
class LoadAndBindInnerFunction(Opcode):
    """Loads and binds an inner function.
    The inner function's default values are expected to be on the stack
    in the order given by `inner_function.parameters_with_defaults`

    Notes
    -----
        | LoadAndBindInnerFunction
        | Stack Effect: 1 - len(inner_function.parameters_with_defaults)
        | Prior: ..., default1, default2, ..., defaultN
        | After: ..., bound_inner_function

    """
    inner_function: InnerFunction

    def next_stack_metadata(self, instruction: "Instruction", bytecode: "Bytecode",
                            previous_stack_metadata: StackMetadata) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata
                .pop(len(self.inner_function.parameters_with_defaults))
                .push(ValueSource((instruction,), object)  # TODO: Typing
        ),)

    def execute(self, frame: "Frame") -> None:
        default_parameters = self.inner_function.parameters_with_defaults
        default_parameters_values = frame.stack[len(frame.stack) - len(default_parameters):]
        frame.stack[len(frame.stack) - len(default_parameters):] = []
        frame.stack.append(self.inner_function.bind(frame, *default_parameters_values))


@dataclass(frozen=True)
class StoreGlobal(Opcode):
    """Stores the value at the top of the stack into a global variable.

    If a global variable has the same name as a builtin, it does not overwrite
    the builtin.

    Notes
    -----
        | StoreGlobal
        | Stack Effect: -1
        | Prior: ..., value
        | After: ...

    Attributes
    ----------
    name: str
        The name of the global variable.

    Examples
    --------
    >>> global x
    ... x = 10
    LoadConstant(constant=10)
    StoreGlobal(name="x")

    >>> global int
    >>> int = 5
    LoadConstant(constant=5)
    StoreGlobal(name="int")
    """

    name: str

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def execute(self, frame: "Frame") -> None:
        frame.globals[self.name] = frame.stack.pop()


@dataclass(frozen=True)
class StoreLocal(Opcode):
    """Stores the value at the top of stack into a local variable.

    The local variable is not a cell variable (a variable
    shared with another function) or a synthethic variable
    (a variable introduced by the compiler).

    Notes
    -----
        | LoadLocal
        | Stack Effect: -1
        | Prior: ..., value
        | After: ...

    Attributes
    ----------
    name: str
        The name of the local variable.

    Examples
    --------
    >>> x = 0
    LoadConstant(constant=0)
    StoreLocal(name="x")
    """

    name: str

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        top = previous_stack_metadata.stack[-1]
        return previous_stack_metadata.pop(1).set_variable(self.name, top),

    def execute(self, frame: "Frame") -> None:
        frame.variables[self.name] = frame.stack.pop()


@dataclass(frozen=True)
class StoreCell(Opcode):
    """Stores the value at the top of stack into a cell variable.

    A cell variable is a variable shared with another function.
    They are typically implemented by creating a holder object called
    a cell, then reading/modifying an attribute of the cell to read/set
    the variable.

    Notes
    -----
        | StoreCell
        | Stack Effect: -1
        | Prior: ..., value
        | After: ...

    Attributes
    ----------
    name: str
        The name of the cell variable.

    Examples
    --------
    >>> nonlocal x
    ... x = 0
    LoadConstant(constant=0)
    StoreCell(name="x")
    """

    name: str
    is_free: bool

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def execute(self, frame: "Frame") -> None:
        frame.closure[self.name].cell_contents = frame.stack.pop()


@dataclass(frozen=True)
class StoreSynthetic(Opcode):
    """Stores the value at the top of stack into a synthetic variable.

    A synthetic variable is a variable introduced by the compiler and is not included
    in `locals()`.

    Notes
    -----
        | StoreSynthetic
        | Stack Effect: -1
        | Prior: ..., value
        | After: ...

    Attributes
    ----------
    index: int
        The index of the synthetic variable.

    Examples
    --------
    >>> for item in collection:
    ...     pass
    LoadLocal(name="collection")
    GetIterator()
    StoreSynthetic(index=0)

    label loop_start

    LoadSynthetic(index=0)
    GetNextElseJumpTo(target=loop_end)
    StoreLocal(name="item")
    JumpTo(target=loop_start)

    label loop_end
    """

    index: int

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        top = previous_stack_metadata.stack[-1]
        return previous_stack_metadata.pop(1).set_synthetic(self.index, top),

    def execute(self, frame: "Frame") -> None:
        frame.synthetic_variables[self.index] = frame.stack.pop()


@dataclass(frozen=True)
class DeleteGlobal(Opcode):
    """Deletes a global variable.

    If a global variable has the same name as a builtin, it does not delete
    the builtin.

    Raises a NameError if the global variable is not defined.

    Notes
    -----
        | DeleteGlobal
        | Stack Effect: 0
        | Prior: ...
        | After: ...

    Attributes
    ----------
    name: str
        The name of the global variable.

    Examples
    --------
    >>> global x
    ... del x
    DeleteGlobal(name="x")

    >>> global int
    >>> del int
    DeleteGlobal(name="int")
    """

    name: str

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata,

    def execute(self, frame: "Frame") -> None:
        try:
            del frame.globals[self.name]
        except KeyError:
            raise NameError(f"name '{self.name}' is not defined")


@dataclass(frozen=True)
class DeleteLocal(Opcode):
    """Deletes a local variable.

    The local variable is not a cell variable (a variable
    shared with another function) or a synthethic variable
    (a variable introduced by the compiler).

    Raises `UnboundLocalError` if the local variable is not defined
    yet.

    Notes
    -----
        | DeleteLocal
        | Stack Effect: 0
        | Prior: ...
        | After: ...

    Attributes
    ----------
    name: str
        The name of the local variable.

    Examples
    --------
    >>> del x
    DeleteLocal(name="x")
    """

    name: str

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        new_metadata = previous_stack_metadata.pop(0)
        new_metadata.variables.pop(self.name)
        return new_metadata,

    def execute(self, frame: "Frame") -> None:
        try:
            del frame.variables[self.name]
        except KeyError:
            raise UnboundLocalError(
                f"local variable '{self.name}' referenced before assignment"
            )


@dataclass(frozen=True)
class DeleteCell(Opcode):
    """Deletes a cell variable.

    A cell variable is a variable shared with another function.
    They are typically implemented by creating a holder object called
    a cell, then reading/modifying an attribute of the cell to read/set
    the variable.

    Raises `NameError` if the cell variable is a free variable and undefined,
    and UnboundLocalError if the cell variable is not defined and not a free variable.

    Notes
    -----
        | DeleteCell
        | Stack Effect: 0
        | Prior: ...
        | After: ...

    Attributes
    ----------
    name: str
        The name of the cell variable.

    Examples
    --------
    >>> nonlocal x
    ... del x
    DeleteCell(name="x")
    """

    name: str
    is_free: bool

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.push(
            ValueSource(
                (instruction,),
                object,  # TODO
            )
        ),

    def execute(self, frame: "Frame") -> None:
        try:
            del frame.closure[self.name].cell_contents
        except (KeyError, ValueError):
            if self.is_free:
                raise NameError(
                    f"free variable  '{self.name}' referenced before assignment"
                )
            else:
                raise UnboundLocalError(
                    f"local variable '{self.name}' referenced before assignment"
                )


########################################
# Object
########################################
@dataclass(frozen=True)
class AsBool(Opcode):
    """Replaces top of stack with its truthfulness.

    Notes
    -----
        | AsBool
        | Stack Effect: 0
        | Prior: ..., object
        | After: ..., bool

    Examples
    --------
    >>> bool(obj)
    # This would normally use LoadGlobal(name='bool'), but
    # AsBool is used here to demostrate how it is used.
    LoadLocal(name="obj")
    AsBool()
    """
    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1).push(
            ValueSource(
                sources=(instruction,),
                value_type=bool,
            )
        ),

    def execute(self, frame: "Frame") -> None:
        obj = frame.stack.pop()
        frame.stack.append(bool(obj))


@dataclass(frozen=True)
class GetType(Opcode):
    """Replaces top of stack with its type

    Notes
    -----
        | GetType
        | Stack Effect: 0
        | Prior: ..., object
        | After: ..., type

    Examples
    --------
    >>> type(obj)
    # This would normally use LoadGlobal(name='type'), but
    # GetType is used here to demostrate how it is used.
    LoadLocal(name="obj")
    GetType()
    """
    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1).push(
            ValueSource(
                sources=(instruction,),
                value_type=type[previous_stack_metadata.stack[-1].value_type],
            )
        ),

    def execute(self, frame: "Frame") -> None:
        obj = frame.stack.pop()
        frame.stack.append(type(obj))


@dataclass(frozen=True)
class LoadAttr(Opcode):
    """Replaces top of stack with the result of an attribute lookup.

    Attribute lookup calls `__getattribute__` on the *type* of the object on top of stack.
    `__getattribute__` is relatively complex, handling descriptors,
    method resolution order and class variables.

    If `__getattribute__` raises AttributeError, it calls `__getattr__` if the type has it defined,
    otherwise it raises the `AttributeError`.

    For details on the Python implementation,
    see https://docs.python.org/3/howto/descriptor.html#invocation-from-an-instance.

    Notes
    -----
        | LoadAttr
        | Stack Effect: 0
        | Prior: ..., object
        | After: ..., attribute

    Attributes
    ----------
    name: str
        The name of the attribute.

    Examples
    --------
    >>> obj.attribute
    LoadLocal(name="obj")
    LoadAttr(name="attribute")
    """

    name: str

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1).push(
            ValueSource(
                sources=(instruction,),
                value_type=object,  # TODO
            )
        ),

    def execute(self, frame: "Frame") -> None:
        obj = frame.stack.pop()
        obj_type = type(obj)
        try:
            frame.stack.append(obj_type.__getattribute__(obj, self.name))
        except AttributeError:
            if not hasattr(obj_type, "__getattr__"):
                raise
            frame.stack.append(obj_type.__getattr__(obj, self.name))


@dataclass(frozen=True)
class StoreAttr(Opcode):
    """Sets an attribute of an object.

    Pop two items from the stack. The first (top of stack) is the object, and the
    second is the value.

    This calls `__setattr__` on the *type* of the object with value.

    Notes
    -----
        | StoreAttr
        | Stack Effect: -2
        | Prior: ..., value, object
        | After: ...

    Attributes
    ----------
    name: str
        The name of the attribute.


    Examples
    --------
    >>> obj.attribute = 10
    LoadConstant(constant=10)
    LoadLocal(name="obj")
    StoreAttr(name="attribute")
    """

    name: str

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(2),

    def execute(self, frame: "Frame") -> None:
        obj = frame.stack.pop()
        value = frame.stack.pop()
        type(obj).__setattr__(obj, self.name, value)


@dataclass(frozen=True)
class DeleteAttr(Opcode):
    """Deletes an attribute of the object on top of stack.

    This calls `__delattr__` on the *type* of the object.

    Notes
    -----
        | DeleteAttr
        | Stack Effect: -1
        | Prior: ..., object
        | After: ...

    Attributes
    ----------
    name: str
        The name of the attribute.

    Examples
    --------
    >>> del obj.attribute
    LoadLocal(name="obj")
    DeleteAttr(name="attribute")
    """

    name: str

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def execute(self, frame: "Frame") -> None:
        obj = frame.stack.pop()
        type(obj).__delattr__(obj, self.name)


########################################
# Stack
########################################
@dataclass(frozen=True)
class Nop(Opcode):
    """Does nothing.

    Used to implement pass statements.

    Notes
    -----
        | Nop
        | Stack Effect: 0
        | Prior: ...
        | After: ...

    Examples
    --------
    >>> pass
    Nop()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata,

    def execute(self, frame: "Frame") -> None:
        pass


@dataclass(frozen=True)
class Dup(Opcode):
    """Duplicates the value on top of stack.

    Notes
    -----
        | Dup
        | Stack Effect: +1
        | Prior: ..., value
        | After: ..., value, value

    Examples
    --------
    >>> x = y = 10
    LoadConstant(constant=10)
    Dup()
    StoreLocal(name="x")
    StoreLocal(name="y")
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        top = previous_stack_metadata.stack[-1]
        return replace(
            previous_stack_metadata, stack=previous_stack_metadata.stack + (top,)
        ),

    def execute(self, frame: "Frame") -> None:
        top = frame.stack[-1]
        frame.stack = frame.stack + [top]


@dataclass(frozen=True)
class DupX1(Opcode):
    """Duplicates the value on top of stack behind the value before it.
    Used for chained comparisons (i.e. x < y < z).

    Notes
    -----
        | DupX1
        | Stack Effect: +1
        | Prior: ..., second, first
        | After: ..., first, second, first

    Examples
    --------
    >>> x < y < z
    LoadLocal(name="x")
    LoadLocal(name="y")
    DupX1()
    BinaryOp(operator=BinaryOperator.Lt)
    Dup()
    IfFalse(target=exit_early)
    Pop()
    LoadLocal(name="z")
    BinaryOp(operator=BinaryOperator.Lt)
    JumpTo(target=done)
    label exit_early
    Swap()
    Pop()
    label done
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        top = previous_stack_metadata.stack[-1]
        return replace(
            previous_stack_metadata,
            stack=previous_stack_metadata.stack[:-2]
            + (top,)
            + previous_stack_metadata.stack[-2:],
        ),

    def execute(self, frame: "Frame") -> None:
        top = frame.stack[-1]
        frame.stack = frame.stack[:-2] + [top] + frame.stack[-2:]


@dataclass(frozen=True)
class Pop(Opcode):
    """Pops off the value on top of stack.
    Used to pop off unused values, such as in expression statements.

    Notes
    -----
        | Pop
        | Stack Effect: -1
        | Prior: ..., value
        | After: ...

    Examples
    --------
    >>> x
    LoadLocal(name="x")
    Pop()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def execute(self, frame: "Frame") -> None:
        frame.stack = frame.stack[:-1]


@dataclass(frozen=True)
class Swap(Opcode):
    """Swaps the two top items on the stack.

    Notes
    -----
        | Swap
        | Stack Effect: 0
        | Prior: ..., second, first
        | After: ..., first, second

    Examples
    --------
    >>> x < y < z
    LoadLocal(name="x")
    LoadLocal(name="y")
    DupX1()
    BinaryOp(operator=BinaryOperator.Lt)
    Dup()
    IfFalse(target=exit_early)
    Pop()
    LoadLocal(name="z")
    BinaryOp(operator=BinaryOperator.Lt)
    JumpTo(target=done)
    label exit_early
    Swap()
    Pop()
    label done
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return replace(
            previous_stack_metadata,
            stack=previous_stack_metadata.stack[:-2]
            + (previous_stack_metadata.stack[-1], previous_stack_metadata.stack[-2]),
        ),

    def execute(self, frame: "Frame") -> None:
        frame.stack = frame.stack[:-2] + [frame.stack[-1], frame.stack[-2]]


########################################
# Control Flow
########################################
@dataclass(frozen=True)
class ReturnValue(Opcode):
    """Returns the value on top of stack.

    Notes
    -----
        | ReturnValue
        | Stack Effect: N/A
        | Prior: ..., return_value
        | After: N/A

    Examples
    --------
    >>> return 10
    LoadConstant(constant=10)
    ReturnValue()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return StackMetadata(
            stack=(),
            variables={},
            synthetic_variables=(),
        ),

    def next_bytecode_indices(self, instruction: "Instruction") -> tuple[int, ...]:
        return ()

    def execute(self, frame: "Frame") -> None:
        out = frame.stack[-1]
        vm = frame.vm
        vm.frames.pop()
        vm.frames[-1].stack.append(out)


@dataclass(frozen=True)
class SaveGeneratorState(Opcode):
    """Saves the frame to the generator at TOS, then pops the generator.

    Notes
    -----
        | SaveGeneratorState
        | Stack Effect: -1
        | Prior: ..., generator
        | After: ...

    Attributes
    ----------
    stack_metadata: StackMetadata
        The state of the frame when this opcode is executed.

    Examples
    --------
    >>> yield 10
    LoadConstant(constant=10)
    LoadSynthetic(index=0)
    SaveGeneratorState(StackMetadata(stack=1, variables=(), closure=(), synthetic_variables=1))
    YieldValue()
    LoadSynthetic(index=0)
    DelegateOrRestoreGeneratorState(StackMetadata(stack=1, variables=(), closure=(), synthetic_variables=1))
    Pop()
    """
    state_id: int
    stack_metadata: StackMetadata

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        if self.stack_metadata is None:
            return previous_stack_metadata.pop(1),
        return self.stack_metadata,

    def execute(self, frame: "Frame") -> None:
        generator = frame.stack.pop()
        # This relies on YieldValue not modifying the frame
        # before popping it!
        generator._state_id = self.state_id
        generator._saved_state = copy(frame)
        generator._saved_state.stack = copy(frame.stack)
        generator._saved_state.synthetic_variables = copy(frame.synthetic_variables)
        generator._saved_state.variables = copy(frame.variables)
        generator._saved_state.closure = copy(frame.closure)


@dataclass(frozen=True)
class SetGeneratorDelegate(Opcode):
    """TOS is generator, and the item below it is the delegate.

    Notes
    -----
        | SetGeneratorDelegate
        | Stack Effect: -2
        | Prior: ..., iterable, generator
        | After: ...

    Examples
    --------
    >>> yield from [1, 2, 3]
    NewList()
    LoadConstant(constant=1)
    ListAppend()
    LoadConstant(constant=2)
    ListAppend()
    LoadConstant(constant=3)
    ListAppend()
    GetIter()
    LoadSynthetic(index=0)
    SetGeneratorDelegate()
    LoadSynthetic(index=0)
    SaveGeneratorState(StackMetadata(stack=1, variables=(), closure=(), synthetic_variables=1))
    LoadSynthetic(index=0)
    DelegateOrRestoreGeneratorState(StackMetadata(stack=1, variables=(), closure=(), synthetic_variables=1))
    Pop()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(2),

    def execute(self, frame: "Frame") -> None:
        generator = frame.stack.pop()
        generator._sub_generator = frame.stack.pop()


class GeneratorOperation(enum.Enum):
    NEXT = 0
    SEND = 1
    THROW = 2


@dataclass(frozen=True)
class DelegateOrRestoreGeneratorState(Opcode):
    """Pops generator from TOS, restores the frame from the generator, then
    replace TOS with the sent value stored on the generator (or raise an
    exception if throw was called on the generator).

    Notes
    -----
        | DelegateOrRestoreGeneratorState
        | Stack Effect: 0
        | Prior: ..., generator
        | After: ..., sent_value_or_yield_from_return

    Attributes
    ----------
    stack: int
        Size of the stack when this bytecode is executed.
    variables: tuple[str, ...]
        Local variables defined by the function
    closure: tuple[str, ...]
        Closure variables used by the function
    synthetic_variables: int
        Synthetic variables used by the function

    Examples
    --------
    >>> yield 10
    LoadConstant(constant=10)
    LoadSynthetic(index=0)
    SaveGeneratorState(stack=1, variables=(), closure=(), synthetic_variables=1)
    YieldValue()
    LoadSynthetic(index=0)
    RestoreGeneratorState(stack=1, variables=(), closure=(), synthetic_variables=1)
    Pop()
    """
    state_id: int
    stack_metadata: StackMetadata

    def next_stack_metadata(
            self,
            instruction: "Instruction",
            bytecode: "Bytecode",
            previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        if self.stack_metadata is None:
            return previous_stack_metadata.pop(2).push(ValueSource(sources=(instruction,), value_type=object)),
        else:
            return self.stack_metadata.pop(1).push(ValueSource(sources=(instruction,), value_type=object)),

    def execute(self, frame: "Frame") -> None:
        generator = frame.stack.pop()
        sent_value = generator._sent_value
        thrown_value = generator._thrown_value
        operation = generator._operation
        sub_generator = generator._sub_generator

        generator._sent_value = None
        generator._thrown_value = None
        generator._operation = GeneratorOperation.NEXT.value

        frame.stack = generator._saved_state.stack
        frame.variables = generator._saved_state.variables
        frame.synthetic_variables = generator._saved_state.synthetic_variables
        frame.closure = generator._saved_state.closure
        frame.current_exception = generator._saved_state.current_exception

        frame.stack.pop()
        if sub_generator is None:
            match operation:
                case int(GeneratorOperation.NEXT.value):
                    frame.stack.append(sent_value)
                case int(GeneratorOperation.SEND.value):
                    frame.stack.append(sent_value)
                case int(GeneratorOperation.THROW.value):
                    raise thrown_value
        else:
            try:
                match operation:
                    case int(GeneratorOperation.NEXT.value):
                        out = next(sub_generator)
                    case int(GeneratorOperation.SEND.value):
                        out = sub_generator.send(sent_value)
                    case int(GeneratorOperation.THROW.value):
                        out = sub_generator.throw(thrown_value)
                    case _:
                        raise SystemError(f'Unhandled operation {operation} for generator')
                frame.stack.append(out)
                YieldValue().execute(frame)
            except StopIteration as result:
                generator._sub_generator = None
                frame.stack.append(result.value)





@dataclass(frozen=True)
class YieldValue(Opcode):
    """Returns the value on top of stack and "pauses" execution.
    Acts identically to ReturnValue.

    Notes
    -----
        | YieldValue
        | Stack Effect: -1
        | Prior: ..., return_value
        | After: ...

    Examples
    --------
    >>> yield 10
    LoadConstant(constant=10)
    LoadSynthetic(index=0)
    SaveGeneratorState()
    YieldValue()
    LoadSynthetic(index=0)
    RestoreGeneratorState()
    Pop()
    """
    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def execute(self, frame: "Frame") -> None:
        out = frame.stack[-1]
        vm = frame.vm
        vm.frames.pop()
        vm.frames[-1].stack.append(out)


@dataclass(frozen=True)
class ReraiseLast(Opcode):
    """Re-raises the last exception raised.

    Notes
    -----
        | ReraiseLast
        | Stack Effect: N/A
        | Prior: ...
        | After: N/A

    Examples
    --------
    >>> try
    ...     raise TypeError
    ... except:
    ...     raise
    LoadGlobal(name="TypeError")
    Raise()
    label handler
    ReraiseLast()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return StackMetadata(
            stack=(),
            variables={},
            synthetic_variables=(),
        ),

    def next_bytecode_indices(self, instruction: "Instruction") -> tuple[int, ...]:
        return ()

    def execute(self, frame: "Frame") -> None:
        raise frame.current_exception


@dataclass(frozen=True)
class Raise(Opcode):
    """Raises the exception or exception type on the top of the stack.

    Notes
    -----
        | Raise
        | Stack Effect: N/A
        | Prior: ..., exception
        | After: N/A

    Examples
    --------
    >>> raise TypeError
    LoadGlobal(name="TypeError")
    Raise()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return StackMetadata(
            stack=(),
            variables={},
            synthetic_variables=(),
        ),

    def next_bytecode_indices(self, instruction: "Instruction") -> tuple[int, ...]:
        return ()

    def execute(self, frame: "Frame") -> None:
        raise frame.stack[-1]


@dataclass(frozen=True)
class RaiseWithCause(Opcode):
    """Raises the exception behind top of stack with top of stack as the cause.

    Notes
    -----
        | Raise
        | Stack Effect: N/A
        | Prior: ..., exception, cause
        | After: N/A

    Examples
    --------
    >>> raise TypeError from ValueError
    LoadGlobal(name="TypeError")
    LoadGlobal(name="ValueError")
    RaiseWithCause()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return StackMetadata(
            stack=(),
            variables={},
            synthetic_variables=(),
        ),

    def next_bytecode_indices(self, instruction: "Instruction") -> tuple[int, ...]:
        return ()

    def execute(self, frame: "Frame") -> None:
        raise frame.stack[-2] from frame.stack[-1]


@dataclass(frozen=True)
class IfTrue(Opcode):
    """Pops top of stack and jumps to target if it is truthy.

    Notes
    -----
        | IfTrue
        | Stack Effect: -1
        | Prior: ..., condition
        | After: ...

    Examples
    --------
    >>> not x
    LoadLocal(name="x")
    IfTrue(target=is_true)
    LoadConstant(constant=True)
    JumpTo(target=done)
    label is_true
    LoadConstant(constant=False)
    label done

    >>> a or b
    LoadLocal(name="a")
    Dup()
    IfTrue(target=done)
    Pop()
    LoadLocal(name="b")
    label done
    """

    target: Label

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def next_bytecode_indices(self, instruction: "Instruction") -> tuple[int, ...]:
        return instruction.bytecode_index + 1, self.target.index

    def execute(self, frame: "Frame") -> None:
        if frame.stack.pop():
            frame.bytecode_index = self.target.index - 1


@dataclass(frozen=True)
class IfFalse(Opcode):
    """Pops top of stack and jumps to target if it is falsey.

    Notes
    -----
        | IfFalse
        | Stack Effect: -1
        | Prior: ..., condition
        | After: ...

    Examples
    --------
    >>> a and b
    LoadLocal(name="a")
    Dup()
    IfFalse(target=done)
    Pop()
    LoadLocal(name="b")
    label done
    """

    target: Label

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def next_bytecode_indices(self, instruction: "Instruction") -> tuple[int, ...]:
        return instruction.bytecode_index + 1, self.target.index

    def execute(self, frame: "Frame") -> None:
        if not frame.stack.pop():
            frame.bytecode_index = self.target.index - 1


@dataclass(frozen=True)
class JumpTo(Opcode):
    """Jumps to target unconditionally.

    Notes
    -----
        | JumpTo
        | Stack Effect: 0
        | Prior: ...
        | After: ...

    Examples
    --------
    >>> not x
    LoadLocal(name="x")
    IfTrue(target=is_true)
    LoadConstant(constant=True)
    JumpTo(target=done)
    label is_true
    LoadConstant(constant=False)
    label done
    """

    target: Label

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata,

    def next_bytecode_indices(self, instruction: "Instruction") -> tuple[int, ...]:
        return (self.target.index,)

    def execute(self, frame: "Frame") -> None:
        frame.bytecode_index = self.target.index - 1


########################################
# Operations
########################################
@dataclass(frozen=True)
class UnaryOp(Opcode):
    """Performs a unary operation on the operand on the top of the stack.

    Notes
    -----
        | UnaryOp
        | Stack Effect: 0
        | Prior: ..., operand
        | After: ..., result

    Examples
    --------
    >>> -x
    LoadLocal(name="x")
    UnaryOp(operator=UnaryOperator.USub)
    """

    operator: UnaryOperator

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1).push(
            ValueSource(
                sources=(instruction,),
                value_type=object,  # TODO
            )
        ),

    def execute(self, frame: "Frame") -> None:
        top = frame.stack.pop()
        frame.stack.append(self.operator.value.function(top))


@dataclass(frozen=True)
class BinaryOp(Opcode):
    """Performs a binary operation on the two items on the top of the stack.

    Despite seemly simple, this is one of the most complex opcodes.
    First, get the types of the left and right operands. If the
    right operand is a more specific type than the left operand
    (i.e. is a subclass of the left operand's type), try the reflected
    operation first (ex: right.__radd__(left)), otherwise try the normal
    operation first (ex: left.__add__(right)). If the method corresponding
    to the operation is not present, the method returns `NotImplemented`,
    or the operand is a builtin type and raises `TypeError`,
    then try the other operation. Written as Python code, it looks like this:

    >>> def binary_op(forward_op, reverse_op, left, right):
    ...     left_type = type(left)
    ...     right_type = type(right)
    ...     def try_op(op, first, second):
    ...         method = getattr(type(first), op, None)
    ...         if method is None:
    ...             return NotImplemented
    ...         try:
    ...             return method(first, second)
    ...         except TypeError:
    ...             if type(first) in {int, float, str, ...}
    ...                 return NotImplemented
    ...             else:
    ...                 raise
    ...     if issubclass(right_type, left_type):
    ...         out = try_op(reverse_op, right, left)
    ...         if out is NotImplemented:
    ...             out = try_op(forward_op, left, right)
    ...         if out is NotImplemented:
    ...             raise TypeError
    ...         return out
    ...     else:
    ...         out = try_op(forward_op, left, right)
    ...         if out is NotImplemented:
    ...             out = try_op(reverse_op, right, left)
    ...         if out is NotImplemented:
    ...             raise TypeError
    ...         return out

    Notes
    -----
        | BinaryOp
        | Stack Effect: -1
        | Prior: ..., left, right
        | After: ..., result

        Comparisons are also BinaryOp, and can return any type
        (for instance, (a < b) can return an int).

    Examples
    --------
    >>> x + y
    LoadLocal(name="x")
    LoadLocal(name="y")
    BinaryOp(operator=BinaryOperator.Add)
    """

    operator: BinaryOperator

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(2).push(
            ValueSource(
                sources=(instruction,),
                value_type=object,  # TODO
            )
        ),

    def execute(self, frame: "Frame") -> None:
        right = frame.stack.pop()
        left = frame.stack.pop()
        frame.stack.append(self.operator.value.function(left, right))


# TODO: Inplace operations should be their own opcode, since the logic is
#       different.


@dataclass(frozen=True)
class IsSameAs(Opcode):
    """Pops off the two top items on the stack and check if they are the same reference.
    If they are the same reference, `True` is pushed to the stack; otherwise
    `False` is pushed to the stack. If `negate` is set, then the result
    is negated before being pushed to the stack.

    Notes
    -----
        | IsSameAs
        | Stack Effect: -1
        | Prior: ..., left, right
        | After: ..., result

    Examples
    --------
    >>> x is y
    LoadLocal(name="x")
    LoadLocal(name="y")
    IsSameAs(negate=False)

    >>> x is not y
    LoadLocal(name="x")
    LoadLocal(name="y")
    IsSameAs(negate=True)
    """

    negate: bool

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(2).push(
            ValueSource(sources=(instruction,), value_type=bool)
        ),

    def execute(self, frame: "Frame") -> None:
        right = frame.stack.pop()
        left = frame.stack.pop()
        if self.negate:
            frame.stack.append(left is not right)
        else:
            frame.stack.append(left is right)


class FormatConversion(Enum):
    """How a value should be converted before being formatted in a f-string."""

    # Values taken from
    # https://docs.python.org/3/library/ast.html#ast.FormattedValue
    NONE = -1
    """Do no conversion before formatting.
    """

    TO_STRING = 115
    """Call str on the value before formatting.
    """

    TO_REPR = 114
    """Call repr on the value before formatting.
    """

    TO_ASCII = 97
    """Call ascii on the value before formatting.
    """

    @staticmethod
    def from_int(value: int) -> "FormatConversion":
        """Gets a `FormatConversion` from the `conversion` attribute of an `ast.FormattedValue` object.

        Parameters
        ----------
        value: int
            The `conversion` attribute of an `ast.FormattedValue` object.

        Returns
        -------
        The `FormatConversion` corresponding to ast int `value`.
        """
        for conversion in FormatConversion:
            if conversion.value == value:
                return conversion
        raise ValueError(f"Invalid conversion: {value}")

    def convert(self, value: Any) -> Any:
        """Performs the conversion.

        Parameters
        ----------
        value
            The value to convert.

        Returns
        -------
        The converted value.
        """
        match self:
            case FormatConversion.NONE:
                return value
            case FormatConversion.TO_STRING:
                return str(value)
            case FormatConversion.TO_REPR:
                return repr(value)
            case FormatConversion.TO_ASCII:
                return ascii(value)
            case _:
                raise RuntimeError(f"Missing conversion: {self}")


@dataclass(frozen=True)
class FormatValue(Opcode):
    """Formats the value on the top of stack, performing a conversion if necessary.
    Raises `TypeError` if __format__ does not return a `str`.

    Notes
    -----
        | FormatValue
        | Stack Effect: 0
        | Prior: ..., value
        | After: ..., formatted_value

    Examples
    --------
    >>> f'{x}'
    LoadLocal(name="x")
    FormatValue(conversion=FormatConversion.NONE, format_spec='')

    >>> f'{x!s}'
    LoadLocal(name="x")
    FormatValue(conversion=FormatConversion.TO_STRING, format_spec='')

    >>> f'{x:spec}'
    LoadLocal(name="x")
    FormatValue(conversion=FormatConversion.NONE, format_spec='spec')

    >>> f'{x!s:spec}'
    LoadLocal(name="x")
    FormatValue(conversion=FormatConversion.TO_STRING, format_spec='spec')
    """

    conversion: FormatConversion
    format_spec: str

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1).push(
            ValueSource(sources=(instruction,), value_type=str)
        ),

    def execute(self, frame: "Frame") -> None:
        value = self.conversion.convert(frame.stack.pop())
        frame.stack.append(format(value, self.format_spec))


@dataclass(frozen=True)
class JoinStringValues(Opcode):
    """Joins the top count items on the stack into a single string.
    The items on the stack are guaranteed to be instances of `str`.

    Notes
    -----
        | JoinStringValues
        | Stack Effect: -count + 1
        | Prior: ..., str_1, str_2, ..., str_count
        | After: ..., combined_str

    Examples
    --------
    >>> f'{greetings} {noun}!'
    LoadLocal(name="greetings")
    FormatValue(conversion=FormatConversion.NONE, format_spec='')
    LoadConstant(constant=' ')
    LoadLocal(name="noun")
    FormatValue(conversion=FormatConversion.NONE, format_spec='')
    LoadConstant(constant='!')
    JoinStringValues(count=3)
    """

    count: int

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(self.count).push(
            ValueSource(sources=(instruction,), value_type=str)
        ),

    def execute(self, frame: "Frame") -> None:
        values = frame.stack[-self.count :]
        del frame.stack[-self.count :]
        out = "".join(values)
        frame.stack.append(out)


@dataclass
class PreparedCall:
    """Stores a function and its arguments; mutated by opcodes."""

    func: Union[Callable, 'Bytecode']
    args: tuple[object, ...] = ()
    kwargs: dict[str, object] = field(default_factory=dict)

    def invoke(self, vm: 'CDisVM'):
        from ._compiler import Bytecode
        from ._vm import Frame
        if isinstance(self.func, Bytecode):
            new_frame = Frame.new_frame(vm)
            new_frame.bind_bytecode_to_frame(self.func, *self.args, **self.kwargs)
            vm.frames.append(new_frame)
            return self.func
        else:
            return self.func(*self.args, **self.kwargs)


@dataclass(frozen=True)
class CreateCallBuilder(Opcode):
    """Creates a call builder for the item on the top of stack.

    Notes
    -----
        | CreateCallBuilder
        | Stack Effect: 0
        | Prior: ..., callable
        | After: ..., call_builder

    Examples
    --------
    >>> func()
    LoadLocal(name="func")
    CreateCallBuilder()
    CallWithBuilder()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        # TODO: CallBuilder type
        return previous_stack_metadata,

    def execute(self, frame: "Frame") -> None:
        func = frame.stack.pop()
        frame.stack.append(PreparedCall(func))


@dataclass(frozen=True)
class WithPositionalArg(Opcode):
    """Pops top of stack and inserts it as the given positional argument.

    Notes
    -----
        | WithPositionalArg
        | Stack Effect: -1
        | Prior: ..., call_builder, positional_arg
        | After: ..., call_builder

    Examples
    --------
    >>> func(1)
    LoadLocal(name="func")
    CreateCallBuilder()
    LoadConstant(constant=1)
    WithPositionalArg(index=0)
    CallWithBuilder()
    """

    index: int

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def execute(self, frame: "Frame") -> None:
        arg = frame.stack.pop()
        prepared_call = frame.stack[-1]
        prepared_call.args = prepared_call.args[: self.index] + (arg,)


@dataclass(frozen=True)
class AppendPositionalArg(Opcode):
    """Pops top of stack and appends it to the positional argument list.

    Notes
    -----
        | AppendPositionalArg
        | Stack Effect: -1
        | Prior: ..., call_builder, arg
        | After: ..., call_builder

    Examples
    --------
    >>> func(*args, 1)
    LoadLocal(name="func")
    CreateCallBuilder()
    LoadLocal(name="args")
    ExtendPositionalArgs()
    LoadConstant(constant=1)
    AppendPositionalArg()
    CallWithBuilder()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def execute(self, frame: "Frame") -> None:
        arg = frame.stack.pop()
        prepared_call = frame.stack[-1]
        prepared_call.args = prepared_call.args + (arg,)


@dataclass(frozen=True)
class WithKeywordArg(Opcode):
    """Pops top of stack and sets the corresponding keyword argument.

    Notes
    -----
        | WithKeywordArg
        | Stack Effect: -1
        | Prior: ..., call_builder, arg
        | After: ..., call_builder

    Examples
    --------
    >>> func(arg=1)
    LoadLocal(name="func")
    CreateCallBuilder()
    LoadConstant(constant=1)
    WithKeywordArg(name="arg")
    CallWithBuilder()
    """

    name: str

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def execute(self, frame: "Frame") -> None:
        arg = frame.stack.pop()
        prepared_call = frame.stack[-1]
        prepared_call.kwargs[self.name] = arg


@dataclass(frozen=True)
class ExtendPositionalArgs(Opcode):
    """Pops top of stack and unpacks it into the positional argument list.

    Notes
    -----
        | ExtendPositionalArgs
        | Stack Effect: -1
        | Prior: ..., call_builder, iterable
        | After: ..., call_builder

    Examples
    --------
    >>> func(*args)
    LoadLocal(name="func")
    CreateCallBuilder()
    LoadLocal(name="args")
    ExtendPositionalArgs()
    CallWithBuilder()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def execute(self, frame: "Frame") -> None:
        arg = frame.stack.pop()
        prepared_call = frame.stack[-1]
        prepared_call.args = prepared_call.args + (*arg,)


@dataclass(frozen=True)
class ExtendKeywordArgs(Opcode):
    """Pops top of stack and unpacks it into the keyword argument dict.

    Notes
    -----
        | ExtendKeywordArgs
        | Stack Effect: -1
        | Prior: ..., call_builder, mapping
        | After: ..., call_builder

    Examples
    --------
    >>> func(**args)
    LoadLocal(name="func")
    CreateCallBuilder()
    LoadLocal(name="args")
    ExtendKeywordArgs()
    CallWithBuilder()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def execute(self, frame: "Frame") -> None:
        arg = frame.stack.pop()
        prepared_call = frame.stack[-1]
        expected_length = len(prepared_call.kwargs) + len(arg)
        result = prepared_call.kwargs.update(arg)
        if expected_length != len(result):
            raise ValueError("Duplicate keyword arguments")
        prepared_call.kwargs = result


@dataclass(frozen=True)
class CallWithBuilder(Opcode):
    """Pops top of stack and call it.
    Top of stack is a call builder object that
    was mutated in prior opcodes to contain the callable
    and its arguments.

    Notes
    -----
        | CallWithBuilder
        | Stack Effect: 0
        | Prior: ..., call_builder
        | After: ..., result

    Examples
    --------
    >>> func()
    LoadLocal(name="func")
    CreateCallBuilder()
    CallWithBuilder()

    >>> func(1)
    LoadLocal(name="func")
    CreateCallBuilder()
    LoadConstant(constant=1)
    WithPositionalArg(index=0)
    CallWithBuilder()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1).push(
            ValueSource(
                sources=(instruction,),
                value_type=object,  # TODO
            )
        ),

    def execute(self, frame: "Frame") -> None:
        vm = frame.vm
        prepared_call = frame.stack.pop()
        old_frame_size = len(vm.frames)
        result = prepared_call.invoke(vm)
        if old_frame_size == len(vm.frames):
            # PreparedCall was for Callable/C Function
            frame.stack.append(result)
        else:
            # PreparedCall was for Bytecode
            while old_frame_size != len(vm.frames):
                vm.step(result)
            #  ReturnValue of the last frame appended the result to our frame's stack



########################################
# Collections
########################################
@dataclass(frozen=True)
class NewList(Opcode):
    """Push a new list into the stack.

    Notes
    -----
        | NewList
        | Stack Effect: +1
        | Prior: ...
        | After: ..., new_list

    Examples
    --------
    >>> []
    NewList()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.push(
            ValueSource(sources=(instruction,), value_type=list)
        ),

    def execute(self, frame: "Frame") -> None:
        frame.stack.append([])


@dataclass(frozen=True)
class NewSet(Opcode):
    """Push a new set into the stack.

    Notes
    -----
        | NewSet
        | Stack Effect: +1
        | Prior: ...
        | After: ..., new_set

    Examples
    --------
    >>> {0}
    NewSet()
    LoadConstant(constant=0)
    SetAdd()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.push(
            ValueSource(sources=(instruction,), value_type=set)
        ),

    def execute(self, frame: "Frame") -> None:
        frame.stack.append(set())


@dataclass(frozen=True)
class NewDict(Opcode):
    """Push a new dict into the stack.

    Notes
    -----
        | NewDict
        | Stack Effect: +1
        | Prior: ...
        | After: ..., new_dict

    Examples
    --------
    >>> {}
    NewDict()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.push(
            ValueSource(sources=(instruction,), value_type=dict)
        ),

    def execute(self, frame: "Frame") -> None:
        frame.stack.append(dict())


@dataclass(frozen=True)
class ListAppend(Opcode):
    """Pop top of stack and append it to the list before it in the stack.
    The list remains on the stack.

    Notes
    -----
        | ListAppend
        | Stack Effect: -1
        | Prior: ..., list, item
        | After: ..., list

    Examples
    --------
    >>> [0]
    NewList()
    LoadConstant(constant=0)
    ListAppend()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def execute(self, frame: "Frame") -> None:
        item = frame.stack.pop()
        frame.stack[-1].append(item)


@dataclass(frozen=True)
class ListExtend(Opcode):
    """Pop top of stack and use it to extend the list before it in the stack.
    The list remains on the stack.

    Notes
    -----
        | ListExtend
        | Stack Effect: -1
        | Prior: ..., list, iterable
        | After: ..., list

    Examples
    --------
    >>> [*items]
    NewList()
    LoadLocal(name="items")
    ListExtend()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def execute(self, frame: "Frame") -> None:
        item = frame.stack.pop()
        frame.stack[-1].extend(item)


@dataclass(frozen=True)
class SetAdd(Opcode):
    """Pop top of stack and adds it to the set before it in the stack.
    The set remains on the stack.

    Notes
    -----
        | SetAdd
        | Stack Effect: -1
        | Prior: ..., set, item
        | After: ..., set

    Examples
    --------
    >>> {0}
    NewSet()
    LoadConstant(constant=0)
    SetAdd()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def execute(self, frame: "Frame") -> None:
        item = frame.stack.pop()
        frame.stack[-1].add(item)


@dataclass(frozen=True)
class SetUpdate(Opcode):
    """Pop top of stack and merge it into the set before it in the stack.
    The set remains on the stack.

    Notes
    -----
        | SetUpdate
        | Stack Effect: -1
        | Prior: ..., set, iterable
        | After: ..., set

    Examples
    --------
    >>> {*items}
    NewSet()
    LoadLocal(name="items")
    SetUpdate()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def execute(self, frame: "Frame") -> None:
        item = frame.stack.pop()
        frame.stack[-1].update(item)


@dataclass(frozen=True)
class DictPut(Opcode):
    """Pops the top two items off the stack and put it in the dict prior to them.
    The dict remains on the stack.
    The top of stack is the value, and the item before it is the key.

    Notes
    -----
        | SetAdd
        | Stack Effect: -2
        | Prior: ..., dict, key, value
        | After: ..., dict

    Examples
    --------
    >>> {"key": "value"}
    NewDict()
    LoadConstant(constant="key")
    LoadConstant(constant="value")
    DictPut()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(2),

    def execute(self, frame: "Frame") -> None:
        value = frame.stack.pop()
        key = frame.stack.pop()
        frame.stack[-1][key] = value


@dataclass(frozen=True)
class DictUpdate(Opcode):
    """Pop top of stack and merge it into the dict before it in the stack.
    The dict remains on the stack.

    Notes
    -----
        | DictUpdate
        | Stack Effect: -1
        | Prior: ..., dict, mapping
        | After: ..., dict

    Examples
    --------
    >>> {**items}
    NewDict()
    LoadLocal(name="items")
    DictUpdate()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1),

    def execute(self, frame: "Frame") -> None:
        value = frame.stack.pop()
        frame.stack[-1].update(value)


@dataclass(frozen=True)
class ListToTuple(Opcode):
    """Unpacks the list at the top of the stack into a tuple and push that tuple to the stack.

    Notes
    -----
        | ListToTuple
        | Stack Effect: 0
        | Prior: ..., list
        | After: ..., tuple

    Examples
    --------
    >>> 0, 1
    NewList()
    LoadConstant(constant=0)
    ListAppend()
    LoadConstant(constant=1)
    ListAppend()
    ListToTuple()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1).push(
            ValueSource(previous_stack_metadata.stack[-1].sources, tuple)
        ),

    def execute(self, frame: "Frame") -> None:
        item = frame.stack.pop()
        frame.stack.append(tuple(item))


@dataclass(frozen=True)
class GetItem(Opcode):
    """Pops off the top two items on the stack to get an item.
    The top of stack is the index, and the item before it is the collection.

    Notes
    -----
        | GetItem
        | Stack Effect: -1
        | Prior: ..., collection, index
        | After: ..., item

    Examples
    --------
    >>> items[0]
    LoadLocal(name="items")
    LoadConstant(constant=0)
    GetItem()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(2).push(
            ValueSource(sources=(instruction,), value_type=object)
        ),

    def execute(self, frame: "Frame") -> None:
        index, items = frame.stack.pop(), frame.stack.pop()
        frame.stack.append(items[index])


@dataclass(frozen=True)
class SetItem(Opcode):
    """Pops off the top three items on the stack to set an item in the collection.
    The top of stack is the index, and the item before it is the collection,
    and the item before the collection is the value the index is set to.

    Notes
    -----
        | SetItem
        | Stack Effect: -3
        | Prior: ..., value, collection, index
        | After: ...

    Examples
    --------
    >>> items[0] = 10
    LoadConstant(constant=10)
    LoadLocal(name="items")
    LoadConstant(constant=0)
    SetItem()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(3),

    def execute(self, frame: "Frame") -> None:
        index, items, value = frame.stack.pop(), frame.stack.pop(), frame.stack.pop()
        items[index] = value


@dataclass(frozen=True)
class DeleteItem(Opcode):
    """Pops off the top two items on the stack to delete an item.
    The top of stack is the index, and the item before it is the collection.

    Notes
    -----
        | DeleteItem
        | Stack Effect: -2
        | Prior: ..., collection, index
        | After: ...

    Examples
    --------
    >>> del items[0]
    LoadLocal(name="items")
    LoadConstant(constant=0)
    DeleteItem()
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(2),

    def execute(self, frame: "Frame") -> None:
        index, items = frame.stack.pop(), frame.stack.pop()
        del items[index]


@dataclass(frozen=True)
class GetIterator(Opcode):
    """Pops off the top of stack and gets its iterator.

    Notes
    -----
        | GetIterator
        | Stack Effect: -1
        | Prior: ..., iterable
        | After: ..., iterator

    Examples
    --------
    >>> for item in items:
    ...     pass
    LoadLocal(name="items")
    GetIterator()
    StoreSynthetic(index=0)
    ...
    """

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1).push(
            ValueSource(sources=(instruction,), value_type=Iterator)
        ),

    def execute(self, frame: "Frame") -> None:
        item = frame.stack.pop()
        frame.stack.append(iter(item))


@dataclass(frozen=True)
class GetNextElseJumpTo(Opcode):
    """Gets the next element of the iterator at top of stack. If next raises `StopIteration`, jump to target instead.

    Notes
    -----
        | GetNextElseJumpTo
        | Stack Effect: 0 if iterator has next element, -1 otherwise
        | Prior: ..., iterator
        | After (has next element): ..., next_element
        | After (iterator exhausted): ...

    Attributes
    ----------
    target: Label
        Where to jump to if the iterator is exhausted.

    Examples
    --------
    >>> for item in collection:
    ...     pass
    LoadLocal(name="collection")
    GetIterator()
    StoreSynthetic(index=0)

    label loop_start

    LoadSynthetic(index=0)
    GetNextElseJumpTo(target=loop_end)
    StoreLocal(name="item")
    JumpTo(target=loop_start)

    label loop_end
    """
    target: Label

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        # TODO: Get type of iterable
        return previous_stack_metadata.pop(1).push(
            ValueSource(sources=(instruction,), value_type=object)
        ), previous_stack_metadata.pop(1)

    def execute(self, frame: "Frame") -> None:
        iterator = frame.stack.pop()
        try:
            frame.stack.append(next(iterator))
        except StopIteration:
            frame.bytecode_index = self.target.index - 1


@dataclass(frozen=True)
class UnpackElements(Opcode):
    """Pops off the top of stack, and push its elements onto the stack in reversed order.
    If `has_extras` is False, before_count is the exact number of elements expected in the iterable,
    and after_count is 0.
    If `has_extras` is True, the first before_count elements of the iterable are added last to the
    stack, then a list containing the items that are not first before_count elements of the
    iterable or the last after_count elements of the iterable is put between, and finally
    the last after_count elements of the iterable are put before the list.

    Notes
    -----
        | UnpackElements
        | Stack Effect: before_count + after_count + (1 if has_extras else 0) - 1
        | Prior: ..., iterable
        | After: ..., last_after_count, ..., last_after_1, extras_list, first_before_count, ..., first_1

    Examples
    --------
    >>> a, b = 1, 2
    NewList()
    LoadConstant(constant=1)
    ListAppend()
    LoadConstant(constant=2)
    ListAppend()
    UnpackElements(before_count=2)
    StoreLocal(name="a")  # 1
    StoreLocal(name="b")  # 2

    >>> a, *b, c = 1, 2, 3, 4
    NewList()
    LoadConstant(constant=1)
    ListAppend()
    LoadConstant(constant=2)
    ListAppend()
     LoadConstant(constant=3)
    ListAppend()
    LoadConstant(constant=4)
    ListAppend()
    UnpackElements(before_count=1, has_extras=True, after_count=1)
    StoreLocal(name="a")  # 1
    StoreLocal(name="b")  # [2, 3]
    StoreLocal(name="c")  # 4
    """

    before_count: int
    has_extras: bool
    after_count: int

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        values = []

        for i in range(self.before_count):
            values.append(ValueSource(sources=(instruction,), value_type=object))

        if self.has_extras:
            values.append(ValueSource(sources=(instruction,), value_type=list))

        for i in range(self.after_count):
            values.append(ValueSource(sources=(instruction,), value_type=object))

        return previous_stack_metadata.pop(1).push(*values),

    def execute(self, frame: "Frame") -> None:
        items = frame.stack.pop()
        iterator = iter(items)
        index = 0
        elements = []
        while index < self.before_count:
            try:
                elements.append(next(iterator))
            except StopIteration:
                raise ValueError(
                    f"not enough values to unpack (expected {'at least ' if self.has_extras else ''}{self.before_count + self.after_count}, "
                    f"got {index})"
                )
            index += 1

        if self.has_extras:
            extras = []

            for item in iterator:
                extras.append(item)
                index += 1

            if len(extras) < self.after_count:
                raise ValueError(
                    f"not enough values to unpack (expected {self.before_count + self.after_count},"
                    f"got {index})."
                )

            elements.append(extras)
            elements.extend(extras[-self.after_count :])
            del extras[-self.after_count :]
        else:
            try:
                # after_count is 0 if has_extras is False
                next(iterator)
                raise ValueError(
                    f"too many values to unpack (expected {self.before_count})"
                )
            except StopIteration:
                pass

        for element in reversed(elements):
            frame.stack.append(element)


@dataclass(frozen=True)
class IsContainedIn(Opcode):
    """Pops the two top items off the stack and check if the second item is contained by the first.
    If `negate` is True, the result is negated.

    Notes
    -----
        | IsContainedIn
        | Stack Effect: -1
        | Prior: ..., item, collection
        | After: ..., is_contained

    Examples
    --------
    >>> a in b
    LoadLocal(name="a")
    LoadLocal(name="b")
    IsContainedIn(negate=False)

    >>> a not in b
    LoadLocal(name="a")
    LoadLocal(name="b")
    IsContainedIn(negate=True)
    """

    negate: bool

    def next_stack_metadata(
        self,
        instruction: "Instruction",
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(2).push(
            ValueSource(sources=(instruction,), value_type=bool)
        ),

    def execute(self, frame: "Frame") -> None:
        right = frame.stack.pop()
        left = frame.stack.pop()
        if self.negate:
            frame.stack.append(left not in right)
        else:
            frame.stack.append(left in right)


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
        return isinstance(other, Instruction) and self.bytecode_index == other.bytecode_index

    def __hash__(self) -> int:
        return hash(self.bytecode_index)


class MethodType(Enum):
    VIRTUAL = 'virtual'
    STATIC = 'static'
    CLASS = 'class'
    GENERATOR = 'generator'

    @staticmethod
    def for_function(function: types.FunctionType) -> "MethodType":
        if isinstance(function, classmethod):
            return MethodType.CLASS
        elif isinstance(function, staticmethod):
            return MethodType.STATIC
        else:
            return MethodType.VIRTUAL

    @staticmethod
    def for_function_ast(function_ast: ast.FunctionDef) -> "MethodType":
        # Note: this does assume people don't do something like
        # have another decorator call classmethod indirectly or
        # override classmethod/staticmethod
        for decorator in function_ast.decorator_list:
            match decorator:
                case ast.Name(id=name):
                    match name:
                        case 'classmethod':
                            return MethodType.CLASS
                        case 'staticmethod':
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
    def methods(self) -> dict[str, 'Bytecode']:
        from ._compiler import Bytecode
        out: dict[str, Bytecode] = {}

        for method_name, method in self.class_attribute_defaults.items():
            if isinstance(method, Bytecode):
                out[method_name] = method

        return out

    def as_class(self, *, vm=None, **vm_kwargs) -> type:
        from ._compiler import Bytecode
        from ._vm import CDisVM

        class Out:
            def __init__(self, *args, **kwargs):
                pass

        if vm is None:
            vm = CDisVM()

        Out.__name__ = self.name
        Out.__qualname__ = self.qualname

        for attribute_name, attribute_value in self.class_attribute_defaults.items():
            if isinstance(attribute_value, Bytecode):
                def _run(owner, *args, _method_bytecode: Bytecode=attribute_value, **kwargs):
                    return vm.run(_method_bytecode, owner, *args, **kwargs, **vm_kwargs)

                setattr(Out, attribute_name, _run)
            else:
                setattr(Out, attribute_name, attribute_value)

        def instance_getter(*vm_args, **vm_kwargs):
            def outer(*args, **kwargs):
                return Out(*args, **kwargs)
            return outer

        return Out
