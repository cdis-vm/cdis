from cdis import CDisVM, to_bytecode
from types import FunctionType
from typing import Callable, cast


def assert_bytecode_for_args(
    function: Callable, *args, trace=False, timeout=3, **kwargs
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
            if type(expected_error) != type(e):
                raise AssertionError(
                    f"Expected error {expected_error!r} but a different exception was raised {e!r}\n"
                    f"Source:\n{inspect.getsource(function)}\n"
                    f"Bytecode:\n{bytecode}\n\n"
                )
            else:
                return
        raise AssertionError(
            f"Expected {expected!r} but an exception was raised {e!r}\n"
            f"Source:\n{inspect.getsource(function)}\n"
            f"Bytecode:\n{bytecode}\n\n"
        ) from e

    if expected_error is not None:
        raise AssertionError(
            f"Expected error {expected_error!r} but got result {actual!r}\n"
            f"Source:\n{inspect.getsource(function)}\n"
            f"Bytecode:\n{bytecode}\n\n"
        )
    elif expected != actual:
        raise AssertionError(
            f"Expected {expected!r} but got {actual!r}\n"
            f"Source:\n{inspect.getsource(function)}\n"
            f"Bytecode:\n{bytecode}\n\n"
        )
