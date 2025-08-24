from ._api import StackMetadata, Opcode, ValueSource, Instruction, Label
from dataclasses import dataclass
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from ..compiler._api import BytecodeDescriptor
    from .._vm import Frame


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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            StackMetadata(
                stack=(),
                variables={},
                synthetic_variables=(),
            ),
        )

    def next_bytecode_indices(self, instruction: Instruction) -> tuple[int, ...]:
        return ()

    def execute(self, frame: "Frame") -> None:
        out = frame.stack[-1]
        vm = frame.vm
        vm.frames.pop()
        vm.frames[-1].stack.append(out)


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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1), previous_stack_metadata.pop(1)

    def next_bytecode_indices(self, instruction: Instruction) -> tuple[int, ...]:
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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata.pop(1), previous_stack_metadata.pop(1)

    def next_bytecode_indices(self, instruction: Instruction) -> tuple[int, ...]:
        return instruction.bytecode_index + 1, self.target.index

    def execute(self, frame: "Frame") -> None:
        if not frame.stack.pop():
            frame.bytecode_index = self.target.index - 1


@dataclass(frozen=True)
class MatchClass(Opcode):
    """Top of stack is the checked type, and the item below it is the quried object.
    Pop only the checked type off the stack. Jump to target if the object is not an instance of
    the checked type, or does not have the specified attributes. If  positional_count,
    read __match_args__ from the popped type, and raise TypeError if positional_count is
    greater than len(__match_args__), or if __match_args__ is missing from the type.
    If the queried object is an instance of the type and has the specified attributes,
    push the values of the specified attributes to the stack.

    Notes
    -----
        | MatchClass
        | Stack Effect: len(attributes) + positional_count - 1 if matched else -1
        | Prior: ..., query, type
        | After (matched): ..., query, positional_0, ..., positional_{positional_count - 1}, attribute_0, ..., attribute_(len(attributes) - 1)
        | After (not matched): ..., query

    Examples
    --------
    >>> match query:
    ...     case MyType(positional_arg, my_attr=value):
    ...         pass
    LoadLocal(name="query")
    MatchClass(target=no_match, positional_count=1, attributes=('my_attr',))
    StoreSynthetic(index=0)  # my_attr
    StoreSynthetic(index=1)  # positional_arg
    LoadSynthetic(index=0)
    StoreLocal(name='value')
    LoadSynthetic(index=1)
    StoreLocal(name='positional_arg')
    JumpTo(target=end_match)
    label no_match
    Pop()
    label end_match
    """

    attributes: tuple[str, ...]
    positional_count: int
    target: Label
    # Types that get special handling; see https://peps.python.org/pep-0634/#class-patterns
    literal_types: ClassVar[tuple[type, ...]] = (
        bool,
        bytearray,
        bytes,
        dict,
        float,
        frozenset,
        int,
        list,
        set,
        str,
        tuple,
    )

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        pushed_items = tuple(
            [ValueSource(sources=(instruction,), value_type=object)]
            * (len(self.attributes) + self.positional_count)
        )
        return previous_stack_metadata.pop(1).push(
            *pushed_items
        ), previous_stack_metadata.pop(1)

    def next_bytecode_indices(self, instruction: Instruction) -> tuple[int, ...]:
        return instruction.bytecode_index + 1, self.target.index

    def execute(self, frame: "Frame") -> None:
        checked_type = frame.stack.pop()
        query = frame.stack[-1]
        if not isinstance(query, checked_type):
            frame.bytecode_index = self.target.index - 1
            return
        sentinel = object()
        out = []
        matched_names = set()
        if self.positional_count > 0:
            matched_args = getattr(checked_type, "__match_args__", sentinel)
            if matched_args is sentinel:
                if issubclass(checked_type, MatchClass.literal_types):
                    matched_args = (sentinel,)
                else:
                    raise TypeError(
                        f"{checked_type}() accepts 0 positional sub-patterns ({self.positional_count} given)"
                    )
            if self.positional_count > len(matched_args):
                raise TypeError(
                    f"{checked_type}() accepts {len(matched_args)} positional sub-patterns ({self.positional_count} given)"
                )
            for attribute in matched_args[: self.positional_count]:
                # handle literal
                if attribute is sentinel:
                    out.append(query)
                    continue
                if attribute in matched_names:
                    raise TypeError(
                        f"{checked_type}() got multiple sub-patterns for attribute '{attribute}'"
                    )
                value = getattr(query, attribute, sentinel)  # type: ignore
                if value is sentinel:
                    frame.bytecode_index = self.target.index - 1
                    return
                out.append(value)
                matched_names.add(attribute)
        for attribute in self.attributes:
            if attribute in matched_names:
                raise TypeError(
                    f"{checked_type}() got multiple sub-patterns for attribute '{attribute}'"
                )
            value = getattr(query, attribute, sentinel)
            if value is sentinel:
                frame.bytecode_index = self.target.index - 1
                return
            out.append(value)
            matched_names.add(attribute)
        frame.stack.extend(out)


@dataclass(frozen=True)
class MatchSequence(Opcode):
    """Top of stack is the queried object.
    Do not pop it off the check, and check if it is a sequence with at least
    length elements (exact if is_exact is True).
    If it not a sequence of at least the specified length, jump to target.

    Notes
    -----
        | MatchSequence
        | Stack Effect: 0
        | Prior: ..., query
        | After: ..., query

    Examples
    --------
    >>> match query:
    ...     case [x, y]:
    ...         pass
    LoadLocal(name="query")
    MatchSequence(length=2, is_exact=True, target=no_match)
    UnpackElements(before_count=2, after_count=0, has_extras=False, target=no_match)
    StoreLocal(name="x")
    StoreLocal(name="y")
    JumpTo(target=end_match)
    label no_match
    Pop()
    label end_match
    """

    length: int
    is_exact: bool
    target: Label

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata, previous_stack_metadata

    def next_bytecode_indices(self, instruction: Instruction) -> tuple[int, ...]:
        return instruction.bytecode_index + 1, self.target.index

    def execute(self, frame: "Frame") -> None:
        from collections.abc import Sequence

        query = frame.stack[-1]
        if (
            not isinstance(query, Sequence)
            or (query_length := len(query)) < self.length
        ):
            frame.bytecode_index = self.target.index - 1
        elif self.is_exact and query_length != self.length:
            frame.bytecode_index = self.target.index - 1


@dataclass(frozen=True)
class MatchMapping(Opcode):
    """Top of stack is the queried object.
    Do not pop it off the check, and check if it is a mapping with the given keys.
    If it not a mapping with the given keys, jump to target.

    Notes
    -----
        | MatchMapping
        | Stack Effect: 0
        | Prior: ..., query
        | After: ..., query

    Examples
    --------
    >>> match query:
    ...     case {'a': x, 'b': y}:
    ...         pass
    LoadLocal(name="query")
    MatchMapping(keys=("a", "b"), target=no_match)
    UnpackMapping(keys=("a", "b"), has_extras=False, target=no_match)
    StoreLocal(name="x")
    StoreLocal(name="y")
    JumpTo(target=end_match)
    label no_match
    Pop()
    label end_match
    """

    keys: tuple[object, ...]
    target: Label

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return previous_stack_metadata, previous_stack_metadata

    def next_bytecode_indices(self, instruction: Instruction) -> tuple[int, ...]:
        return instruction.bytecode_index + 1, self.target.index

    def execute(self, frame: "Frame") -> None:
        from collections.abc import Mapping

        query = frame.stack[-1]
        if isinstance(query, Mapping):
            mapping_keys = query.keys()
            for key in self.keys:
                if key not in mapping_keys:
                    frame.bytecode_index = self.target.index - 1
                    return
        else:
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
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata,)

    def next_bytecode_indices(self, instruction: Instruction) -> tuple[int, ...]:
        return (self.target.index,)

    def execute(self, frame: "Frame") -> None:
        frame.bytecode_index = self.target.index - 1


__all__ = (
    "ReturnValue",
    "IfTrue",
    "IfFalse",
    "MatchClass",
    "MatchSequence",
    "MatchMapping",
    "JumpTo",
)
