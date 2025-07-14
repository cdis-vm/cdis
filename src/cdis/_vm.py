from dataclasses import dataclass
from types import CellType

from ._compiler import Bytecode, ExceptionHandler

import time


@dataclass
class Frame:
    vm: "CDisVM"
    bytecode_index: int
    stack: list
    current_exception: BaseException | None
    variables: dict[str, object]
    closure: dict[str, CellType]
    globals: dict[str, object]
    exception_handlers: tuple[ExceptionHandler, ...]
    synthetic_variables: list[object]

    @staticmethod
    def new_frame(vm: "CDisVM") -> "Frame":
        return Frame(
            vm=vm,
            bytecode_index=0,
            stack=[],
            current_exception=None,
            variables={},
            globals={},
            closure={},
            exception_handlers=(),
            synthetic_variables=[],
        )

    def bind_bytecode_to_frame(self, bytecode: Bytecode, *args, **kwargs) -> None:
        self.globals = bytecode.globals
        self.closure = bytecode.closure
        self.exception_handlers = bytecode.exception_handlers
        self.synthetic_variables = [None] * bytecode.synthetic_count
        bound = bytecode.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, value in bound.arguments.items():
            self.variables[name] = value


class CDisVM:
    frames: list[Frame]
    builtins: dict[str, object]
    _start: float
    _timeout: float
    _trace: bool

    def __init__(self, builtins: dict[str, object] = __builtins__):
        self.frames = []
        self.builtins = builtins

    def run(
        self, bytecode: Bytecode, *args, trace=False, timeout=float("inf"), **kwargs
    ) -> object:
        # Bottom frame for return value, top frame for function
        self.frames = [Frame.new_frame(self), Frame.new_frame(self)]
        self._start = time.time()
        self._timeout = timeout
        self._trace = trace
        self.frames[-1].bind_bytecode_to_frame(bytecode, *args, **kwargs)

        while len(self.frames) > 1:
            self.step(bytecode)

        out = self.frames[0].stack[0]
        self.frames = []
        return out

    def step(self, bytecode: Bytecode) -> None:
        if time.time() - self._start > self._timeout:
            raise TimeoutError(f"Timeout of {self._timeout}s exceeded")
        top_frame = self.frames[-1]
        instruction = bytecode.instructions[top_frame.bytecode_index]
        if self._trace:
            print(f'''
            stack={top_frame.stack}
            variables={top_frame.variables}
            synthetics={top_frame.synthetic_variables}
            current_exception={top_frame.current_exception}
            {instruction}
            ''')
        try:
            instruction.opcode.execute(top_frame)
            top_frame.bytecode_index += 1
        except BaseException as e:
            while len(self.frames) > 1:
                top_frame = self.frames[-1]
                top_frame.current_exception = e
                top_frame.stack = [e]
                for exception_handler in top_frame.exception_handlers:
                    if (
                            exception_handler.from_label._bytecode_index
                            <= top_frame.bytecode_index
                            < exception_handler.to_label._bytecode_index
                    ):
                        if isinstance(e, exception_handler.exception_class):
                            top_frame.bytecode_index = (
                                exception_handler.handler_label._bytecode_index
                            )
                            break
                else:
                    self.frames.pop()
                    continue
                break
            else:
                raise e
