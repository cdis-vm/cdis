from ._api import StackMetadata, Opcode, ValueSource, Instruction
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..compiler._api import BytecodeDescriptor
    from .._vm import Frame


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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.push(
                ValueSource((instruction,), type(self.constant))
            ),
        )

    def execute(self, frame: "Frame") -> None:
        frame.stack.append(self.constant)


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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata,)

    def execute(self, frame: "Frame") -> None:
        pass


@dataclass(frozen=True)
class ImportModule(Opcode):
    """Push the module with the given name to the stack.
    The module is loaded and executed if it is not loaded yet.
    Raises ImportError if the module cannot be found.

    Used to implement import statements.

    Notes
    -----
        | ImportModule
        | Stack Effect: +1
        | Prior: ...
        | After: ..., module

    Examples
    --------
    >>> import cdis
    ImportModule(name='cdis', level=0, from_list=())
    """

    name: str
    level: int
    from_list: tuple[str, ...]

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        from types import ModuleType

        return (
            previous_stack_metadata.push(
                ValueSource(sources=(instruction,), value_type=ModuleType)
            ),
        )

    def execute(self, frame: "Frame") -> None:
        frame.stack.append(
            __import__(
                self.name, frame.globals, frame.locals, self.from_list, self.level
            )
        )
