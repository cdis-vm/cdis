from ._api import StackMetadata, Opcode, ValueSource, Instruction, ClassInfo
from dataclasses import dataclass, replace
from copy import copy
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._compiler import Bytecode
    from .._vm import Frame


@dataclass(frozen=True)
class LoadAndBindInnerGenerator(Opcode):
    """Loads and binds an inner generator.

    Notes
    -----
        | LoadAndBindInnerGenerator
        | Stack Effect: +1
        | Prior: ...
        | After: ..., bound_inner_generator

    """

    inner_generator: ClassInfo

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(1).push(
                ValueSource((instruction,), object)  # TODO: Typing
            ),
        )

    def execute(self, frame: "Frame") -> None:
        new_closure = frame.closure
        new_class_attributes = copy(self.inner_generator.class_attribute_defaults)

        for method, bytecode in self.inner_generator.methods.items():
            new_class_attributes[method] = replace(bytecode, closure=new_closure)

        generator_copy = replace(
            self.inner_generator, class_attribute_defaults=new_class_attributes
        )
        frame.stack.append(generator_copy.as_class())


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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        if self.stack_metadata is None:
            return (previous_stack_metadata.pop(1),)
        return (self.stack_metadata,)

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(2),)

    def execute(self, frame: "Frame") -> None:
        generator = frame.stack.pop()
        generator._sub_generator = frame.stack.pop()


class GeneratorOperation(Enum):
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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        if self.stack_metadata is None:
            return (
                previous_stack_metadata.pop(2).push(
                    ValueSource(sources=(instruction,), value_type=object)
                ),
            )
        else:
            return (
                self.stack_metadata.pop(1).push(
                    ValueSource(sources=(instruction,), value_type=object)
                ),
            )

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
                        raise SystemError(
                            f"Unhandled operation {operation} for generator"
                        )

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(1),)

    def execute(self, frame: "Frame") -> None:
        out = frame.stack[-1]
        vm = frame.vm
        vm.frames.pop()
        vm.frames[-1].stack.append(out)
