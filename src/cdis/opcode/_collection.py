from ._api import StackMetadata, Opcode, ValueSource, Instruction, Label
from dataclasses import dataclass
from typing import Iterator, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from .._compiler import Bytecode
    from .._vm import Frame


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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.push(
                ValueSource(sources=(instruction,), value_type=list)
            ),
        )

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.push(
                ValueSource(sources=(instruction,), value_type=set)
            ),
        )

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.push(
                ValueSource(sources=(instruction,), value_type=dict)
            ),
        )

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(1),)

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(1),)

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(1),)

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(1),)

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(2),)

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(1),)

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(1).push(
                ValueSource(previous_stack_metadata.stack[-1].sources, tuple)
            ),
        )

    def execute(self, frame: "Frame") -> None:
        item = frame.stack.pop()
        frame.stack.append(tuple(item))


@dataclass(frozen=True)
class BuildSlice(Opcode):
    """The top item on the stack is the step, the item below it the stop, and the item
    below that the start.
    Build a new slice from the start, end, and step. Equivalent to slice(start, stop, step)

    Notes
    -----
        | BuildSlice
        | Stack Effect: -3
        | Prior: ..., start, end, step
        | After: ..., slice

    Examples
    --------
    >>> items[1:3]
    LoadLocal(name="items")
    LoadConstant(constant=1)
    LoadConstant(constant=3)
    LoadConstant(constant=None)
    BuildSlice()
    GetItem()
    """

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(3).push(
                ValueSource(sources=(instruction,), value_type=slice)
            ),
        )

    def execute(self, frame: "Frame") -> None:
        step, stop, start = frame.stack.pop(), frame.stack.pop(), frame.stack.pop()
        frame.stack.append(slice(start, stop, step))


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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(2).push(
                ValueSource(sources=(instruction,), value_type=object)
            ),
        )

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(3),)

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (previous_stack_metadata.pop(2),)

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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(1).push(
                ValueSource(sources=(instruction,), value_type=Iterator)
            ),
        )

    def execute(self, frame: "Frame") -> None:
        item = frame.stack.pop()
        frame.stack.append(iter(item))


@dataclass(frozen=True)
class GetAwaitableIterator(Opcode):
    """Pops off the top of stack and gets its awaitable iterator.

    Notes
    -----
        | GetAwaitableIterator
        | Stack Effect: -1
        | Prior: ..., awaitable
        | After: ..., iterator

    Examples
    --------
    >>> await task
    LoadLocal(name="task")
    GetAwaitableIterator()
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
        return (
            previous_stack_metadata.pop(1).push(
                ValueSource(sources=(instruction,), value_type=Iterator)
            ),
        )

    def execute(self, frame: "Frame") -> None:
        item = frame.stack.pop()
        await_function = getattr(type(item), "__await__", None)
        if await_function is None:
            from types import GeneratorType
            from inspect import CO_ITERABLE_COROUTINE

            # CPython generator-based coroutines do not have an __await__ attribute!
            # need to manually check their flags
            if (
                isinstance(item, GeneratorType)
                and item.gi_code.co_flags & CO_ITERABLE_COROUTINE
            ):
                frame.stack.append(item)
            else:
                raise TypeError(f"object {item} can't be used in 'await' expression")
        else:
            frame.stack.append(await_function(item))


@dataclass(frozen=True)
class GetAsyncIterator(Opcode):
    """Pops off the top of stack and gets its async for iterator.

    Notes
    -----
        | GetAsyncIterator
        | Stack Effect: -1
        | Prior: ..., async_iterable
        | After: ..., asyc_iterator

    Examples
    --------
    >>> async for item in items:
    ...     pass
    LoadLocal(name="items")
    GetAsyncIterator()
    LoadSynthetic(index=0)
    SetGeneratorDelegate()

    label loop_start
    LoadSynthetic(index=0)
    SaveGeneratorState(StackMetadata(stack=1, variables=('item', 'items'), closure=(), synthetic_variables=1))
    LoadSynthetic(index=0)

    try:
        DelegateOrRestoreGeneratorState(StackMetadata(stack=1, variables=(), closure=(), synthetic_variables=1))
        StoreLocal(name="item")
    except AsyncStopIteration:
        JumpTo(target=loop_end)

    Nop()
    JumpTo(target=loop_start)
    label loop_end
    """

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(1).push(
                ValueSource(sources=(instruction,), value_type=AsyncIterator)
            ),
        )

    def execute(self, frame: "Frame") -> None:
        item = frame.stack.pop()
        frame.stack.append(aiter(item))


@dataclass(frozen=True)
class GetAsyncNext(Opcode):
    """Pops off the top of stack and gets its async next.

    Notes
    -----
        | GetAsyncIterator
        | Stack Effect: -1
        | Prior: ..., asyc_iterator
        | After: ..., async_next

    Examples
    --------
    >>> async for item in items:
    ...     pass
    LoadLocal(name="items")
    GetAsyncIterator()
    LoadSynthetic(index=0)
    SetGeneratorDelegate()

    label loop_start
    LoadSynthetic(index=0)
    SaveGeneratorState(StackMetadata(stack=1, variables=('item', 'items'), closure=(), synthetic_variables=1))
    LoadSynthetic(index=0)

    try:
        DelegateOrRestoreGeneratorState(StackMetadata(stack=1, variables=(), closure=(), synthetic_variables=1))
        StoreLocal(name="item")
    except AsyncStopIteration:
        JumpTo(target=loop_end)

    Nop()
    JumpTo(target=loop_start)
    label loop_end
    """

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(1).push(
                ValueSource(sources=(instruction,), value_type=AsyncIterator)
            ),
        )

    def execute(self, frame: "Frame") -> None:
        async_iterator = frame.stack.pop()
        frame.stack.append(anext(async_iterator))


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
        instruction: Instruction,
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
        instruction: Instruction,
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

        return (previous_stack_metadata.pop(1).push(*values),)

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
class UnpackMapping(Opcode):
    """Pops off the top of stack (which is a mapping), and push the values of the given keys
    onto the stack in reversed order.
    If `has_extras` is True, push all items in the mapping not specified by the given keys into
    a new dict at the top of the stack

    Notes
    -----
        | UnpackElements
        | Stack Effect: len(keys) + (1 if has_extras else 0) - 1
        | Prior: ..., mapping
        | After: ..., value_(len(keys) - 1), ..., value_1, value_0, (extras_dict if has_extras)

    Examples
    --------
    >>> match mapping:
    ...     case {'a': x, 'b': y}
    LoadLocal(name="mapping")
    MatchMapping(keys=("a", "b"))
    UnpackMapping(keys=("a", "b"), has_extras=False)
    StoreLocal(name="x")
    StoreLocal(name="y")
    """

    keys: tuple[object, ...]
    has_extras: bool

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        values = [ValueSource(sources=(instruction,), value_type=object)] * len(
            self.keys
        )
        if self.has_extras:
            values.append(ValueSource(sources=(instruction,), value_type=dict))
        return (previous_stack_metadata.pop(1).push(*values),)

    def execute(self, frame: "Frame") -> None:
        mapping = frame.stack.pop()
        if self.has_extras:
            extras = dict()
            for name, value in mapping.items():
                if name not in self.keys:
                    extras[name] = value
            for name in reversed(self.keys):
                frame.stack.append(mapping[name])
            frame.stack.append(extras)
        else:
            for name in reversed(self.keys):
                frame.stack.append(mapping[name])


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
        instruction: Instruction,
        bytecode: "Bytecode",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(2).push(
                ValueSource(sources=(instruction,), value_type=bool)
            ),
        )

    def execute(self, frame: "Frame") -> None:
        right = frame.stack.pop()
        left = frame.stack.pop()
        if self.negate:
            frame.stack.append(left not in right)
        else:
            frame.stack.append(left in right)


__all__ = (
    "NewList",
    "NewSet",
    "NewDict",
    "ListAppend",
    "ListExtend",
    "SetAdd",
    "SetUpdate",
    "DictPut",
    "DictUpdate",
    "ListToTuple",
    "BuildSlice",
    "GetItem",
    "SetItem",
    "DeleteItem",
    "GetIterator",
    "GetAwaitableIterator",
    "GetAsyncIterator",
    "GetAsyncNext",
    "GetNextElseJumpTo",
    "UnpackElements",
    "UnpackMapping",
    "IsContainedIn",
)
