from ._api import StackMetadata, Opcode, ValueSource, Instruction
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._compiler import Bytecode
    from .._vm import Frame


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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(1).push(
                ValueSource(
                    sources=(instruction,),
                    value_type=bool,
                )
            ),
        )

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(1).push(
                ValueSource(
                    sources=(instruction,),
                    value_type=type[previous_stack_metadata.stack[-1].value_type],
                )
            ),
        )

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
        instruction: Instruction,
        bytecode: "Bytecode",
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
        obj = frame.stack.pop()
        obj_type = type(obj)
        try:
            frame.stack.append(obj_type.__getattribute__(obj, self.name))
        except AttributeError:
            if not hasattr(obj_type, "__getattr__"):
                raise
            frame.stack.append(obj_type.__getattr__(obj, self.name))


@dataclass(frozen=True)
class LoadObjectTypeAttr(Opcode):
    """Replaces top of stack with the result of an attribute lookup from its type.

    Notes
    -----
        | LoadTypeAttr
        | Stack Effect: 0
        | Prior: ..., object
        | After: ..., type_attribute

    Attributes
    ----------
    name: str
        The name of the attribute.

    Examples
    --------
    >>> with ctx:
    LoadLocal(name="ctx")
    LoadTypeAttr(name="__enter__")
    """

    name: str

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "Bytecode",
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
        obj = frame.stack.pop()
        obj_type = type(obj)
        obj_type_type = type(obj_type)
        try:
            frame.stack.append(obj_type_type.__getattribute__(obj_type, self.name))
        except AttributeError:
            if not hasattr(obj_type_type, "__getattr__"):
                raise
            frame.stack.append(obj_type_type.__getattr__(obj_type, self.name))


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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(2),)

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(1),)

    def execute(self, frame: "Frame") -> None:
        obj = frame.stack.pop()
        type(obj).__delattr__(obj, self.name)


__all__ = (
    "AsBool",
    "GetType",
    "LoadAttr",
    "LoadObjectTypeAttr",
    "StoreAttr",
    "DeleteAttr",
)
