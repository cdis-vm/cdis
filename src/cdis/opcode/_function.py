from ._api import StackMetadata, Opcode, ValueSource, Instruction, InnerFunction
from dataclasses import dataclass, field
from typing import Union, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..compiler._api import Bytecode, BytecodeDescriptor
    from .._vm import Frame, CDisVM


@dataclass
class PreparedCall:
    """Stores a function and its arguments; mutated by opcodes."""

    func: Union[Callable, "Bytecode"]
    args: tuple[object, ...] = ()
    kwargs: dict[str, object] = field(default_factory=dict)

    @staticmethod
    def _wrap_bytecode(vm: "CDisVM", bytecode: "Bytecode") -> Callable:
        def wrapper(*args, **kwargs):
            return vm.run(bytecode, *args, **kwargs)

        return wrapper

    @staticmethod
    def _convert_args(vm: "CDisVM", args: tuple[object, ...]) -> tuple[object, ...]:
        from ..compiler import Bytecode

        out = []
        for arg in args:
            if isinstance(arg, Bytecode):
                out.append(PreparedCall._wrap_bytecode(vm, arg))
            else:
                out.append(arg)
        return tuple(out)

    @staticmethod
    def _convert_kwargs(vm: "CDisVM", kwargs: dict[str, object]) -> dict[str, object]:
        from ..compiler import Bytecode

        out = {}

        for key, value in kwargs.items():
            if isinstance(value, Bytecode):
                out[key] = PreparedCall._wrap_bytecode(vm, value)
            else:
                out[key] = value

        return out

    def invoke(self, vm: "CDisVM"):
        from ..compiler import Bytecode

        args = PreparedCall._convert_args(vm, self.args)
        kwargs = PreparedCall._convert_kwargs(vm, self.kwargs)

        if isinstance(self.func, Bytecode):
            return vm.run(self.func, *args, **kwargs)
        else:
            return self.func(*args, **kwargs)


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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        # TODO: CallBuilder type
        return (previous_stack_metadata,)

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(1),)

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(1),)

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(1),)

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(1),)

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(1),)

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(1).push(
                ValueSource(
                    sources=(instruction,),
                    value_type=object,  # TODO
                )
            ),
        )

    def execute(self, frame: "Frame") -> None:
        vm = frame.vm
        prepared_call = frame.stack.pop()
        result = prepared_call.invoke(vm)
        frame.stack.append(result)


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

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(
                len(self.inner_function.parameters_with_defaults)
            ).push(
                ValueSource((instruction,), object)  # TODO: Typing
            ),
        )

    def execute(self, frame: "Frame") -> None:
        default_parameters = self.inner_function.parameters_with_defaults
        default_parameters_values = frame.stack[
            len(frame.stack) - len(default_parameters) :
        ]
        frame.stack[len(frame.stack) - len(default_parameters) :] = []
        frame.stack.append(self.inner_function.bind(frame, *default_parameters_values))


__all__ = (
    "PreparedCall",
    "CreateCallBuilder",
    "WithPositionalArg",
    "AppendPositionalArg",
    "WithKeywordArg",
    "ExtendPositionalArgs",
    "ExtendKeywordArgs",
    "CallWithBuilder",
    "LoadAndBindInnerFunction",
)
