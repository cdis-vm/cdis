from ._api import StackMetadata, Opcode, Instruction
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..compiler._api import BytecodeDescriptor
    from .._vm import Frame


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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        top = previous_stack_metadata.stack[-1]
        return (
            replace(
                previous_stack_metadata, stack=previous_stack_metadata.stack + (top,)
            ),
        )

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        top = previous_stack_metadata.stack[-1]
        return (
            replace(
                previous_stack_metadata,
                stack=previous_stack_metadata.stack[:-2]
                + (top,)
                + previous_stack_metadata.stack[-2:],
            ),
        )

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(1),)

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            replace(
                previous_stack_metadata,
                stack=previous_stack_metadata.stack[:-2]
                + (
                    previous_stack_metadata.stack[-1],
                    previous_stack_metadata.stack[-2],
                ),
            ),
        )

    def execute(self, frame: "Frame") -> None:
        frame.stack = frame.stack[:-2] + [frame.stack[-1], frame.stack[-2]]


__all__ = ("Dup", "DupX1", "Pop", "Swap")
