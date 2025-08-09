from cdis import CDisVM, to_bytecode
from types import FunctionType
from typing import Callable, Generator, cast, Sequence
from dataclasses import dataclass


def assert_bytecode_for_args(
    function: Callable, *args, trace=False, timeout=3,
        check_exception_args=True, **kwargs
):
    import inspect

    vm = CDisVM()
    bytecode = to_bytecode(cast(FunctionType, function))
    expected_error = None
    try:
        expected = function(*args, **kwargs)
    except Exception as e:
        expected = None
        expected_error = e
    try:
        actual = vm.run(bytecode, trace=trace, timeout=timeout, *args, **kwargs)
    except Exception as e:
        if expected_error is not None:
            if expected_error.__class__ != e.__class__ or (check_exception_args and expected_error.args != e.args):
                raise AssertionError(
                    f"Expected error {expected_error!r} but a different exception was raised {e!r}\n"
                    f"Stack Trace: {vm.stack_trace}\n"
                    f"Source:\n{inspect.getsource(function)}\n"
                    f"Bytecode:\n{bytecode}\n\n"
                )
            else:
                return
        raise AssertionError(
            f"Expected {expected!r} but an exception was raised {e!r}\n"
            f"Stack Trace: {vm.stack_trace}\n"
            f"Source:\n{inspect.getsource(function)}\n"
            f"Bytecode:\n{bytecode}\n\n"
        ) from e

    if expected_error is not None:
        raise AssertionError(
            f"Expected error {expected_error!r} but got result {actual!r}\n"
            f"Stack Trace: {vm.stack_trace}\n"
            f"Source:\n{inspect.getsource(function)}\n"
            f"Bytecode:\n{bytecode}\n\n"
        )
    elif expected != actual:
        raise AssertionError(
            f"Expected {expected!r} but got {actual!r}\n"
            f"Stack Trace: {vm.stack_trace}\n"
            f"Source:\n{inspect.getsource(function)}\n"
            f"Bytecode:\n{bytecode}\n\n"
        )


def assert_async_bytecode_for_args(
    function: Callable, *args, trace=False, timeout=3,
        check_exception_args=True, **kwargs
):
    import inspect
    from asyncio import run

    vm = CDisVM()
    bytecode = to_bytecode(cast(FunctionType, function))
    async_class = bytecode.as_class(vm=vm, trace=trace, timeout=timeout)

    expected_error = None
    try:
        expected = run(function(*args, **kwargs))
    except Exception as e:
        expected = None
        expected_error = e
    try:
        async def actual_wrapper():
            return await async_class(*args, **kwargs)

        actual = run(actual_wrapper())
    except Exception as e:
        if expected_error is not None:
            if expected_error.__class__ != e.__class__ or (check_exception_args and expected_error.args != e.args):
                raise AssertionError(
                    f"Expected error {expected_error!r} but a different exception was raised {e!r}\n"
                    f"Stack Trace: {vm.stack_trace}\n"
                    f"Source:\n{inspect.getsource(function)}\n"
                    f"Bytecode:\n{bytecode}\n\n"
                )
            else:
                return
        raise AssertionError(
            f"Expected {expected!r} but an exception was raised {e!r}\n"
            f"Stack Trace: {vm.stack_trace}\n"
            f"Source:\n{inspect.getsource(function)}\n"
            f"Bytecode:\n{bytecode}\n\n"
        ) from e

    if expected_error is not None:
        raise AssertionError(
            f"Expected error {expected_error!r} but got result {actual!r}\n"
            f"Stack Trace: {vm.stack_trace}\n"
            f"Source:\n{inspect.getsource(function)}\n"
            f"Bytecode:\n{bytecode}\n\n"
        )
    elif expected != actual:
        raise AssertionError(
            f"Expected {expected!r} but got {actual!r}\n"
            f"Stack Trace: {vm.stack_trace}\n"
            f"Source:\n{inspect.getsource(function)}\n"
            f"Bytecode:\n{bytecode}\n\n"
        )


@dataclass(frozen=True)
class Sent:
    value: object


@dataclass(frozen=True)
class Thrown:
    exception: BaseException


@dataclass(frozen=True)
class Skip:
    count: int = 1


def assert_generator_bytecode_for_args(
    generator: Callable, *args, trace=False, timeout=3, sequence: Sequence[Sent | Thrown | Skip] | None = None,
        **kwargs
):
    import inspect

    vm = CDisVM()
    generator_bytecode = to_bytecode(cast(FunctionType, generator))
    generator_class = generator_bytecode.as_class(vm=vm, trace=trace, timeout=timeout)
    python_generator = generator(*args, **kwargs)
    vm_generator = generator_class(*args, **kwargs)

    def assert_code(func: Callable[[Generator], object]) -> bool:
        expected_error = None
        try:
            expected = func(python_generator)
        except Exception as expected:
            expected_error = expected

        actual_error = None
        try:
            actual = func(vm_generator)
        except Exception as actual:
            actual_error = actual

        if expected_error is None:
            if actual_error is not None:
                raise AssertionError(
                    f"Expected {expected!r} but got exception {actual_error!r}\n"
                    f"Stack Trace: {vm.stack_trace}\n"
                    f"Source:\n{inspect.getsource(generator)}\n"
                    f"Bytecode:\n{generator_bytecode.methods['_next_0']}"
                )
            elif expected != actual:
                raise AssertionError(
                    f"Expected {expected!r} but got {actual!r}\n"
                    f"Stack Trace: {vm.stack_trace}\n"
                    f"Source:\n{inspect.getsource(generator)}\n"
                    f"Bytecode:\n{generator_bytecode.methods['_next_0']}"
                )
            else:
                return True
        else:
            if actual_error is None:
                raise AssertionError(
                    f"Expected error {expected_error!r} but got result {actual!r}\n"
                    f"Stack Trace: {vm.stack_trace}\n"
                    f"Source:\n{inspect.getsource(generator)}\n"
                    f"Bytecode:\n{generator_bytecode.methods['_next_0']}"
                )
            elif actual_error.__class__ != expected_error.__class__ or actual_error.args != expected_error.args:
                raise AssertionError(
                    f"Expected error {expected_error!r} but got error {actual_error!r}\n"
                    f"Stack Trace: {vm.stack_trace}\n"
                    f"Source:\n{inspect.getsource(generator)}\n"
                    f"Bytecode:\n{generator_bytecode.methods['_next_0']}"
                ) from actual_error
            else:
                return not isinstance(expected_error, StopIteration)

    if sequence is None:
        while assert_code(lambda _generator: next(_generator)):
            pass
    else:
        for operation in sequence:
            match operation:
                case Sent(value):
                    assert_code(lambda _generator: _generator.send(value))
                case Skip(count):
                    for i in range(count):
                        assert_code(lambda _generator: next(_generator))
                case Thrown(exception):
                    assert_code(lambda _generator: _generator.throw(exception))
