from ._api import StackMetadata, Opcode, Instruction, Label
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._compiler import Bytecode
    from .._vm import Frame


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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return ()

    def next_bytecode_indices(self, instruction: Instruction) -> tuple[int, ...]:
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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return ()

    def next_bytecode_indices(self, instruction: Instruction) -> tuple[int, ...]:
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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return ()

    def next_bytecode_indices(self, instruction: Instruction) -> tuple[int, ...]:
        return ()

    def execute(self, frame: "Frame") -> None:
        raise frame.stack[-2] from frame.stack[-1]


@dataclass(frozen=True)
class JumpIfNotMatchExceptType(Opcode):
    """Top of stack is an exception type and the item below it is an exception.
    If the exception is not an instance of the exception type, jump to target.
    If the exception type is not a subclass of BaseException, raise TypeError.

    Notes
    -----
        | JumpIfNotMatchExceptType
        | Stack Effect: -2
        | Prior: ..., exception, exception_type
        | After: ...

    Examples
    --------
    >>> try:
    ...     pass
    ... except ValueError:
    ...     pass
    StoreSynthetic(index=0)
    LoadSynthetic(index=0)
    LoadGlobal(name="ValueError")
    JumpIfNotMatchExceptType(target=reraise)
    Nop()
    JumpTo(target=continue)

    label reraise
    Reraise()

    label continue
    """

    target: Label

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(2), previous_stack_metadata.pop(2)

    def next_bytecode_indices(self, instruction: Instruction) -> tuple[int, ...]:
        return instruction.bytecode_index + 1, self.target.index

    def execute(self, frame: "Frame") -> None:
        exception_type, exception = frame.stack.pop(), frame.stack.pop()
        if not isinstance(exception_type, type):
            raise TypeError(
                "catching classes that do not inherit from BaseException is not allowed"
            )
        elif not issubclass(exception_type, BaseException):
            raise TypeError(
                "catching classes that do not inherit from BaseException is not allowed"
            )
        elif not isinstance(exception, exception_type):
            frame.bytecode_index = self.target.index - 1
            return


__all__ = ("ReraiseLast", "Raise", "RaiseWithCause", "JumpIfNotMatchExceptType")
