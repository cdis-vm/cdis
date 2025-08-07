from dataclasses import dataclass
from types import CellType

from ._compiler import Bytecode, ExceptionHandler

import time


@dataclass
class Frame:
    vm: "CDisVM"
    function_name: str
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
            function_name='<unknown>',
            stack=[],
            current_exception=None,
            variables={},
            globals={},
            closure={},
            exception_handlers=(),
            synthetic_variables=[],
        )

    def bind_bytecode_to_frame(self, bytecode: Bytecode, *args, **kwargs) -> None:
        self.function_name = bytecode.function_name
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
    stack_trace: list[(str, int)] | None
    _start: float
    _timeout: float
    _trace: bool

    def __init__(self, builtins: dict[str, object] = __builtins__):
        self.frames = []
        self.builtins = builtins
        self.stack_trace = None

    def run(
        self, bytecode: Bytecode, *args, trace=False, timeout=float("inf"), **kwargs
    ) -> object:
        target_frame: int = 1
        if self.frames:
            target_frame = len(self.frames)
            self.frames.append(Frame.new_frame(self))
        else:
            # Bottom frame for return value, top frame for function
            self.frames = [Frame.new_frame(self), Frame.new_frame(self)]
            self._start = time.time()
            self._timeout = timeout
            self._trace = trace

        self.frames[-1].bind_bytecode_to_frame(bytecode, *args, **kwargs)

        while len(self.frames) > target_frame:
            self.step(bytecode)

        out = self.frames[-1].stack[-1]
        if target_frame == 1:
            self.frames = []
        else:
            self.frames[-1].stack.pop()
        return out

    def step(self, bytecode: Bytecode) -> None:
        if time.time() - self._start > self._timeout:
            raise TimeoutError(f"Timeout of {self._timeout}s exceeded")
        top_frame = self.frames[-1]
        instruction = bytecode.instructions[top_frame.bytecode_index]
        self.stack_trace = None
        if self._trace:
            print(f'''
            function={top_frame.function_name}
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
            self.stack_trace = [(_frame.function_name, _frame.bytecode_index) for _frame in self.frames]
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

    def __repr__(self):
        return '>'.join(f'{_frame.function_name}:{_frame.bytecode_index}' for _frame in self.frames)
