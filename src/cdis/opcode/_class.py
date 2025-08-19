from ._api import StackMetadata, Opcode, ValueSource, Instruction
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._compiler import Bytecode
    from .._vm import Frame


@dataclass(frozen=True)
class LoadAndBindInnerClass(Opcode):
    """Top of stack is keyword arguments, and the item below it are the tuple of base types.
    Loads and binds an inner class.

    Notes
    -----
        | LoadAndBindInnerClass
        | Stack Effect: -1
        | Prior: ..., bases, keyword_args
        | After: ..., bound_inner_class
    """

    class_name: str
    class_body: "Bytecode"

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(2).push(ValueSource((instruction,), type)),)

    def execute(self, frame: "Frame") -> None:
        keyword_args, bases = frame.stack.pop(), frame.stack.pop()
        metaclass = keyword_args.pop("metaclass", type)
        if isinstance(metaclass, type):
            for cls in bases:
                cls_meta = type(cls)
                if issubclass(metaclass, cls_meta):
                    pass  # metaclass is a more derived than cls_meta
                elif issubclass(cls_meta, metaclass):
                    metaclass = cls_meta  # cls_meta is more derived than metaclass
                    # so it should be used as the metaclass instead
                else:
                    # The metaclasses are unrelated
                    raise TypeError(
                        "metaclass conflict: "
                        "the metaclass of a derived class "
                        "must be a (non-strict) subclass "
                        "of the metaclasses of all its bases"
                    )

        new_bases = []
        sentinel = object()
        for base in bases:
            if isinstance(base, type):
                new_bases.append(base)
            elif (
                mro_entries := getattr(base, "__mro_entries__", sentinel)
            ) is not sentinel:
                new_bases.extend(mro_entries(bases))
            else:
                new_bases.append(base)

        new_bases = tuple(new_bases)
        prepared_locals = getattr(metaclass, "__prepare__", lambda _, __: {})(
            self.class_name, new_bases
        )
        new_class_body = replace(self.class_body, closure=frame.closure)
        frame.vm.run(new_class_body, prepared_locals)
        new_class = metaclass(
            self.class_name, new_bases, prepared_locals, **keyword_args
        )
        frame.stack.append(new_class)


@dataclass(frozen=True)
class LoadTypeAttrOrGlobal(Opcode):
    """TOS is a mapping. If the specified variable exists in TOS, load it.
    Otherwise load the global variable or builtin with that name onto the stack.

    Notes
    -----
        | LoadLocalOrGlobal
        | Stack Effect: 0
        | Prior: ..., mapping
        | After: ..., local_or_global

    Attributes
    ----------
    name: str
        The name of the type attribute.

    Examples
    --------
    >>> class A:
    ...     a = id
    LoadSynthentic(index=0)
    LoadLocalOrGlobal(name="id")
    StoreLocal(name="a")
    """

    name: str

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            (previous_stack_metadata.pop(1).push(ValueSource((instruction,), object))),
        )

    def execute(self, frame: "Frame") -> None:
        from ._variable import LoadGlobal

        mapping = frame.stack.pop()
        sentinel = object()
        if (out := mapping.get(self.name, sentinel)) is not sentinel:
            frame.stack.append(out)
        else:
            LoadGlobal(name=self.name).execute(frame)


@dataclass(frozen=True)
class StoreTypeAttr(Opcode):
    """TOS is a mapping, and the item below it is a value.
    Stores the value into the mapping..

    Notes
    -----
        | StoreTypeAttr
        | Stack Effect: -2
        | Prior: ..., value, mapping
        | After: ...

    Attributes
    ----------
    name: str
        The name of the local variable.

    Examples
    --------
    >>> class A:
    ...     x = 0
    LoadConstant(constant=0)
    LoadSynthetic(index=0)
    StoreTypeAttr(name="x")
    """

    name: str

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        top = previous_stack_metadata.stack[-1]
        # Note: technically, the mapping can modify the set value.
        return (previous_stack_metadata.pop(1).set_variable(self.name, top),)

    def execute(self, frame: "Frame") -> None:
        mapping, value = frame.stack.pop(), frame.stack.pop()
        mapping[self.name] = value


@dataclass(frozen=True)
class DeleteTypeAttr(Opcode):
    """TOS is a mapping.
    Delete the given key from the mapping.

    Notes
    -----
        | DeleteTypeAttr
        | Stack Effect: -1
        | Prior: ..., mapping
        | After: ...

    Attributes
    ----------
    name: str
        The name of the local variable.

    Examples
    --------
    >>> class A:
    ...     x = 0
    ...     del x
    LoadConstant(constant=0)
    LoadSynthetic(index=0)
    StoreTypeAttr(name="x")
    LoadSynthetic(index=0)
    DeleteTypeAttr(name="x")
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
        mapping = frame.stack.pop()
        del mapping[self.name]


__all__ = (
    "LoadAndBindInnerClass",
    "LoadTypeAttrOrGlobal",
    "StoreTypeAttr",
    "DeleteTypeAttr",
)
