from ._api import StackMetadata, Opcode, ValueSource, Instruction
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..compiler._api import BytecodeDescriptor
    from .._vm import Frame


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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.push(
                ValueSource(
                    (instruction,), type(bytecode.globals.get(self.name, object()))
                )
            ),
        )

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        local_metadata = previous_stack_metadata.variables.get(
            self.name, ValueSource((), object)
        )
        return (previous_stack_metadata.push(local_metadata),)

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.push(
                ValueSource(
                    (instruction,),
                    object,  # TODO
                )
            ),
        )

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        synthetic_metadata = previous_stack_metadata.synthetic_variables[self.index]
        return (previous_stack_metadata.push(synthetic_metadata),)

    def execute(self, frame: "Frame") -> None:
        frame.stack.append(frame.synthetic_variables[self.index])


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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(1),)

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        top = previous_stack_metadata.stack[-1]
        return (previous_stack_metadata.pop(1).set_variable(self.name, top),)

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(1),)

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        top = previous_stack_metadata.stack[-1]
        return (previous_stack_metadata.pop(1).set_synthetic(self.index, top),)

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata,)

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        new_metadata = previous_stack_metadata.pop(0)
        new_metadata.variables.pop(self.name)
        return (new_metadata,)

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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.push(
                ValueSource(
                    (instruction,),
                    object,  # TODO
                )
            ),
        )

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


__all__ = (
    "LoadGlobal",
    "LoadLocal",
    "LoadCell",
    "LoadSynthetic",
    "StoreGlobal",
    "StoreLocal",
    "StoreCell",
    "StoreSynthetic",
    "DeleteGlobal",
    "DeleteLocal",
    "DeleteCell",
)
