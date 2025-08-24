import copy
from functools import lru_cache, cached_property


import cdis.opcode as opcode
from cdis.opcode import InnerFunction
from .._type_analysis import resolve_stack_metadata

import ast
import abc
from contextlib import contextmanager
import inspect
from dataclasses import dataclass, field
from types import CellType, FunctionType
from typing import (
    Union,
    Callable,
    TypeVar,
    Iterable,
    MutableSequence,
    Generator,
    ContextManager,
    Sequence,
    cast,
)

from ..opcode import StackMetadata, Instruction


@dataclass(frozen=True)
class ExceptionHandler:
    exception_class: type
    from_label: opcode.Label
    to_label: opcode.Label
    handler_label: opcode.Label


@dataclass(frozen=True)
class BytecodeDescriptor:
    signature: inspect.Signature
    function_type: opcode.FunctionType
    instructions: tuple[Instruction, ...]
    labels: tuple[opcode.Label, ...]
    exception_handlers: tuple[ExceptionHandler, ...]
    synthetic_count: int
    closure: dict[str, CellType] = field(default_factory=dict)
    globals: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Bytecode:
    function_name: str
    signature: inspect.Signature
    function_type: opcode.FunctionType
    method_type: opcode.MethodType
    synthetic_count: int
    instructions: tuple[opcode.Instruction, ...]
    stack_metadata: tuple[opcode.StackMetadata, ...]
    exception_handlers: tuple[ExceptionHandler, ...]
    annotate_function: Union["Bytecode", None] = None
    closure: dict[str, CellType] = field(default_factory=dict)
    globals: dict[str, object] = field(default_factory=dict)
    free_names: frozenset[str] = field(default_factory=frozenset)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        out = ""
        out += f"function_name: {self.function_name}\n"
        out += "instructions:\n"
        for instruction in self.instructions:
            out += f"{instruction.bytecode_index:<16}{str(instruction.opcode)}\n"
        out += f"exception handlers: {self.exception_handlers}\n"
        return out


def _bytecode_annotations(self) -> dict[str, object]:
    from .._vm import CDisVM

    if self.annotate_function is None:
        return {}
    return cast(dict[str, object], CDisVM().run(self.annotate_function, 1))


_cached_bytecode_annotations = cached_property(_bytecode_annotations)
_cached_bytecode_annotations.__set_name__(Bytecode, "__annotations__")
Bytecode.__annotations__ = _cached_bytecode_annotations


class AstPassSharedMetadata:
    def get_id(self) -> object:
        return type(self)


_PrivateMetadata = TypeVar("_PrivateMetadata")
_SharedMetadata = TypeVar("_SharedMetadata", bound=AstPassSharedMetadata)


def code_to_bytecode(
    code: Iterable["Code"],
    exception_handlers: Iterable[ExceptionHandler] | None,
    function_name: str,
    signature: inspect.Signature,
    compile_metadata: "CompileMetadata",
    function_type: opcode.FunctionType = opcode.FunctionType.FUNCTION,
) -> Bytecode:
    if exception_handlers is None:
        exception_handlers = ()

    out: list[Instruction] = []
    labels: list[opcode.Label] = []
    bytecode_index = 0
    for instruction in code:
        match instruction:
            case opcode.Label():
                instruction._bytecode_index = bytecode_index
                labels.append(instruction)
            case opcode.Opcode():
                out.append(
                    Instruction(
                        opcode=instruction, bytecode_index=bytecode_index, lineno=0
                    )
                )
                bytecode_index += 1
            case Compilable():
                to_compile = [instruction]
                compile_metadata.compiler._inner_compile(
                    compile_metadata, to_compile, labels, 0, 1
                )
                for compiled in to_compile:
                    out.append(
                        Instruction(
                            opcode=cast(opcode.Opcode, compiled),
                            bytecode_index=bytecode_index,
                            lineno=0,
                        )
                    )
                    bytecode_index += 1
            case _:
                raise ValueError(f"Unexpected instruction type {type(instruction)}")

    instructions = tuple(out)
    labels_tuple = tuple(labels)
    exception_handlers_tuple = tuple(exception_handlers)

    bytecode_descriptor = BytecodeDescriptor(
        signature=signature,
        function_type=function_type,
        instructions=instructions,
        labels=labels_tuple,
        exception_handlers=exception_handlers_tuple,
        closure=compile_metadata.closure_dict,
        globals=compile_metadata.globals_dict,
        synthetic_count=compile_metadata._max_synthetics,
    )
    stack_metadata = resolve_stack_metadata(bytecode_descriptor)
    for deferred in compile_metadata.get_shared(DeferredProcessors).deferred_processors:
        deferred(bytecode_descriptor, stack_metadata)
    compile_metadata.get_shared(DeferredProcessors).deferred_processors.clear()

    return Bytecode(
        function_name=function_name,
        signature=signature,
        function_type=function_type,
        method_type=opcode.MethodType.VIRTUAL,
        synthetic_count=compile_metadata._max_synthetics,
        exception_handlers=exception_handlers_tuple,
        instructions=instructions,
        stack_metadata=stack_metadata,
        globals=compile_metadata.globals_dict,
        closure=compile_metadata.closure_dict,
        free_names=compile_metadata.free_names,
    )


@dataclass
class CompileMetadata:
    compiler: "AstCompiler"
    function_name: str
    function_type: opcode.FunctionType
    method_type: opcode.MethodType
    signature: inspect.Signature
    local_names: frozenset[str]
    cell_names: frozenset[str]
    global_names: frozenset[str]
    free_names: frozenset[str]
    globals_dict: dict[str, object]
    closure_dict: dict[str, CellType]
    _synthetic_count: int = 0
    _max_synthetics: int = 0
    _shared_data: dict[
        tuple[type[AstPassSharedMetadata], object], AstPassSharedMetadata
    ] = field(default_factory=dict)
    _private_data: dict[int, object] = field(default_factory=dict)

    def blank_copy(self) -> "CompileMetadata":
        return CompileMetadata(
            compiler=self.compiler,
            function_name=self.function_name,
            function_type=self.function_type,
            method_type=self.method_type,
            signature=self.signature,
            local_names=self.local_names,
            cell_names=self.cell_names,
            global_names=self.global_names,
            free_names=self.free_names,
            globals_dict=self.globals_dict,
            closure_dict=self.closure_dict,
        )

    def compile_inner_function(
        self,
        function_node: ast.FunctionDef
        | ast.AsyncFunctionDef
        | ast.ClassDef
        | ast.Lambda,
    ) -> "InnerFunction":
        from ._function import (
            SignatureAndParameters,
            get_signature_from_arguments,
            find_used_variables,
            create_annotation_function,
        )

        match function_node:
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name):
                function_name = name
                signature_and_parameters = get_signature_from_arguments(
                    function_node.args
                )
            case ast.Lambda():
                function_name = "<lambda>"
                signature_and_parameters = get_signature_from_arguments(
                    function_node.args
                )
            case ast.ClassDef(name=name):
                function_name = name
                signature_and_parameters = SignatureAndParameters(
                    signature=inspect.Signature(
                        parameters=(
                            inspect.Parameter(
                                "_attribute_mapping",
                                kind=inspect.Parameter.POSITIONAL_ONLY,
                            ),
                        )
                    ),
                    parameters_with_defaults=(),
                )
            case _:
                raise ValueError(f"Unexpected AST Type: {type(function_node)}")

        function_type = opcode.FunctionType.for_function_ast(function_node)
        method_type = opcode.MethodType.for_function_ast(function_node)
        used_variables = find_used_variables(function_node, self.local_names)
        annotate_function = create_annotation_function(
            func_def=function_node, compile_metadata=self
        )

        inner_bytecode = self.compiler.compile(
            ast_node=function_node,
            function_name=f"{self.function_name}.{function_name}",
            signature=signature_and_parameters.signature,
            function_type=function_type,
            method_type=method_type,
            local_names=used_variables.variable_names,
            cell_names=used_variables.cell_names,
            global_names=self.global_names,
            free_names=(self.local_names & used_variables.cell_names),
            globals_dict=self.globals_dict,
            closure_dict=copy.copy(self.closure_dict),
        )

        return InnerFunction(
            bytecode=inner_bytecode,
            annotate_function=annotate_function,
            parameters_with_defaults=signature_and_parameters.parameters_with_defaults,
        )

    @property
    def next_synthetic(self):
        return self._synthetic_count

    def get(
        self, ast_pass: "AstPass", initializer: Callable[[], _PrivateMetadata]
    ) -> _PrivateMetadata:
        key = id(ast_pass)
        if key not in self._private_data:
            self._private_data[key] = initializer()
        return self._private_data[key]

    def get_shared(
        self, shared: type[_SharedMetadata] | _SharedMetadata
    ) -> _SharedMetadata:
        if isinstance(shared, type):
            shared = shared()

        key = (type(shared), shared.get_id())
        if key not in self._shared_data:
            self._shared_data[key] = shared
        return self._shared_data[key]


@dataclass
class DeferredProcessors(AstPassSharedMetadata):
    deferred_processors: list[
        Callable[[BytecodeDescriptor, tuple[StackMetadata, ...]], None]
    ] = field(default_factory=list)

    def add_deferred_processor(
        self, processor: Callable[[BytecodeDescriptor, tuple[StackMetadata, ...]], None]
    ) -> None:
        self.deferred_processors.append(processor)


@dataclass
class ExceptionHandlers(AstPassSharedMetadata):
    exception_handlers: list[ExceptionHandler] = field(default_factory=list)
    finally_blocks: list[Sequence["Code"]] = field(default_factory=list)

    def push_finally_block(self, block: Sequence["Code"]) -> None:
        self.finally_blocks.append(block)

    def pop_finally_block(self) -> None:
        self.finally_blocks.pop()

    def add_exception_handler(self, exception_handler: ExceptionHandler) -> None:
        self.exception_handlers.append(exception_handler)


class Compilable:
    pass


Code = Union[Compilable, opcode.Opcode, opcode.Label]


@dataclass(frozen=True, kw_only=True)
class InnerCompilable(Compilable):
    compilable: Compilable
    before: Callable[[CompileMetadata], None] = lambda _: None
    after: Callable[[CompileMetadata], None] = lambda _: None


Context = TypeVar("Context")


def compile_context_manager(
    context: Context,
    out: MutableSequence[Code],
    metadata: CompileMetadata,
    before: Callable[[CompileMetadata], None],
    after: Callable[[CompileMetadata], None],
) -> ContextManager[Context]:
    @contextmanager
    def wrapper() -> Generator[Code, None, None]:
        start = len(out)
        before(metadata)
        try:
            yield context
        finally:
            after(metadata)
            for index in range(start, len(out)):
                compilable = out[index]
                match compilable:
                    case InnerCompilable(
                        compilable=inner, before=inner_before, after=inner_after
                    ):

                        def new_before(
                            _metadata: CompileMetadata, _inner_before=inner_before
                        ) -> Iterable[Code] | None:
                            _inner_before(_metadata)
                            before(_metadata)

                        def new_after(
                            _metadata: CompileMetadata, _inner_after=inner_after
                        ) -> Iterable[Code] | None:
                            after(_metadata)
                            _inner_after(_metadata)

                        out[index] = InnerCompilable(
                            compilable=inner, before=new_before, after=new_after
                        )
                    case Compilable():
                        out[index] = InnerCompilable(
                            compilable=compilable, before=before, after=after
                        )

    return wrapper()


def use_synthetics(
    count: int, out: MutableSequence[Code], metadata: CompileMetadata
) -> ContextManager[tuple[int, ...]]:
    new_variables = tuple(metadata._synthetic_count + n for n in range(count))  # noqa

    def add_synthetics(_metadata: CompileMetadata):
        _metadata._synthetic_count += count  # noqa
        _metadata._max_synthetics = max(
            _metadata._max_synthetics, _metadata._synthetic_count
        )  # noqa

    def remove_synthetics(_metadata: CompileMetadata):
        _metadata._synthetic_count -= count  # noqa

    return compile_context_manager(
        new_variables, out, metadata, add_synthetics, remove_synthetics
    )


@dataclass(frozen=True)
class AstCompilable(Compilable):
    node: ast.AST


@dataclass(frozen=True)
class CallableCompilable(Compilable):
    callable: Callable[[CompileMetadata], Iterable[Code]]


class AstPass(abc.ABC):
    @abc.abstractmethod
    def accept(
        self, compile_metadata: CompileMetadata, compilable: Compilable
    ) -> Iterable[Code] | None: ...


def unindent(code: str) -> str:
    lowest_indent = float("inf")
    for line in code.splitlines():
        indent = len(line) - len(line.lstrip())
        if indent == 0 and len(line) > 0:
            return code
        elif indent != 0:
            lowest_indent = min(lowest_indent, indent)

    out = ""
    for line in code.splitlines():
        if len(line) == 0:
            out += line
        else:
            out += line[lowest_indent:] + "\n"

    return out


@lru_cache
def default_passes() -> tuple[AstPass, ...]:
    from ._class import ClassPass
    from ._collection import CollectionPass
    from ._comprehension import ComprehensionPass
    from ._control_flow import ControlFlowPass
    from ._function import FunctionPass
    from ._exception import ExceptionPass
    from ._generator import GeneratorPass
    from ._meta import MetaPass
    from ._object import ObjectPass
    from ._operation import OperationPass
    from ._variable import VariablePass

    return (
        ClassPass(),
        CollectionPass(),
        ComprehensionPass(),
        ControlFlowPass(),
        FunctionPass(),
        ExceptionPass(),
        GeneratorPass(),
        MetaPass(),
        ObjectPass(),
        OperationPass(),
        VariablePass(),
    )


class AstCompiler:
    passes: tuple[AstPass, ...]

    def __init__(self, passes: tuple[AstPass, ...] = None):
        self.passes = default_passes()
        if passes is not None:
            self.passes = passes + self.passes

    def _inner_compile(
        self,
        compile_metadata: CompileMetadata,
        work_or_compiled_result: list[Code],
        labels: list[opcode.Label],
        work_index: int,
        end_index: int,
    ) -> None:
        while work_index < end_index:
            work = work_or_compiled_result[work_index]

            if isinstance(work, opcode.Label):
                work._bytecode_index = work_index
                work_or_compiled_result.pop(work_index)
                end_index -= 1
                labels.append(work)
                continue

            if not isinstance(work, Compilable):
                work_index += 1
                continue

            if isinstance(work, InnerCompilable):
                old_size = len(work_or_compiled_result)
                work.before(compile_metadata)
                work_or_compiled_result[work_index] = work.compilable
                self._inner_compile(
                    compile_metadata=compile_metadata,
                    labels=labels,
                    work_or_compiled_result=work_or_compiled_result,
                    work_index=work_index,
                    end_index=work_index + 1,
                )
                work.after(compile_metadata)
                new_size = len(work_or_compiled_result)
                size_increase = new_size - old_size
                end_index += size_increase
                work_index += size_increase
                continue

            if isinstance(work, CallableCompilable):
                old_size = len(work_or_compiled_result)
                work_or_compiled_result[work_index : work_index + 1] = work.callable(
                    compile_metadata
                )
                new_size = len(work_or_compiled_result)
                size_increase = new_size - old_size
                end_index += work_index + size_increase
                break

            for ast_pass in self.passes:
                compiled = ast_pass.accept(compile_metadata, work)
                if compiled is not None:
                    old_size = len(work_or_compiled_result)
                    work_or_compiled_result[work_index : work_index + 1] = compiled
                    new_size = len(work_or_compiled_result)
                    size_increase = new_size - old_size
                    end_index += size_increase
                    break
            else:
                raise RuntimeError(
                    f"Failed to compile {work}; no pass accepted node {work}."
                )

    def compile(
        self,
        function_name: str,
        signature: inspect.Signature,
        function_type: opcode.FunctionType,
        method_type: opcode.MethodType,
        local_names: frozenset[str],
        cell_names: frozenset[str],
        global_names: frozenset[str],
        free_names: frozenset[str],
        closure_dict: dict[str, CellType],
        globals_dict: dict[str, object],
        ast_node: ast.AST,
    ) -> Bytecode | opcode.ClassInfo:
        from ._class import class_body_prologue
        from ._generator import (
            generator_function_prologue,
            create_generator_class_from_bytecode,
        )

        compiled_code: list[Code] = []
        labels: list[opcode.Label] = []
        work_metadata = CompileMetadata(
            compiler=self,
            function_name=function_name,
            signature=signature,
            function_type=function_type,
            method_type=method_type,
            local_names=local_names,
            cell_names=cell_names,
            global_names=global_names,
            free_names=free_names,
            globals_dict=globals_dict,
            closure_dict=closure_dict,
        )

        is_generator_or_async = False
        is_class_body = False

        match function_type:
            case (
                opcode.FunctionType.GENERATOR
                | opcode.FunctionType.ASYNC_GENERATOR
                | opcode.FunctionType.COROUTINE_GENERATOR
                | opcode.FunctionType.ASYNC_FUNCTION
            ):
                is_generator_or_async = True
            case opcode.FunctionType.CLASS_BODY:
                is_class_body = True

        if is_generator_or_async:
            generator_function_prologue(compiled_code, work_metadata)
        elif is_class_body:
            class_body_prologue(compiled_code, work_metadata)

        parameter_cells = (cell_names - frozenset(free_names)) & {
            name for name in local_names if name in signature.parameters.keys()
        }

        for parameter_cell_var in parameter_cells:
            if not is_generator_or_async:
                compiled_code.append(opcode.LoadLocal(name=parameter_cell_var))
                compiled_code.append(
                    opcode.StoreCell(name=parameter_cell_var, is_free=False)
                )
            closure_dict[parameter_cell_var] = CellType()

        if isinstance(ast_node.body, ast.AST):
            compiled_code.append(AstCompilable(ast_node.body))
        else:
            compiled_code.extend(AstCompilable(node) for node in ast_node.body)

        self._inner_compile(
            compile_metadata=work_metadata,
            work_or_compiled_result=compiled_code,
            labels=labels,
            work_index=0,
            end_index=len(compiled_code),
        )

        if function_type is opcode.FunctionType.LAMBDA:
            compiled_code.append(opcode.ReturnValue())
        else:
            compiled_code.append(AstCompilable(ast.Return(ast.Constant(None))))
            self._inner_compile(
                compile_metadata=work_metadata,
                work_or_compiled_result=compiled_code,
                labels=labels,
                work_index=len(compiled_code) - 1,
                end_index=len(compiled_code),
            )
        opcodes = cast(Sequence[opcode.Opcode], compiled_code)
        instructions = [
            Instruction(opcode=op, bytecode_index=index, lineno=0)
            for index, op in enumerate(opcodes)
        ]

        deferred_processors = work_metadata.get_shared(
            DeferredProcessors
        ).deferred_processors
        # Sort exception handlers so narrower blocks get priority
        exception_handlers = sorted(
            work_metadata.get_shared(ExceptionHandlers).exception_handlers,
            key=lambda handler: (
                handler.to_label.index - handler.from_label.index,
                handler.from_label.index,
            ),
        )

        bytecode_descriptor = BytecodeDescriptor(
            instructions=tuple(instructions),
            signature=signature,
            function_type=function_type,
            labels=tuple(labels),
            exception_handlers=tuple(exception_handlers),
            closure=closure_dict,
            globals=globals_dict,
            synthetic_count=work_metadata._max_synthetics,
        )

        if len(deferred_processors) > 0:
            stack_metadata = resolve_stack_metadata(bytecode_descriptor)
            for deferred_processor in deferred_processors:
                deferred_processor(bytecode_descriptor, stack_metadata)
        else:
            stack_metadata = ()

        deferred_processors.clear()

        out = Bytecode(
            function_name=function_name,
            signature=bytecode_descriptor.signature,
            function_type=bytecode_descriptor.function_type,
            method_type=method_type,
            instructions=bytecode_descriptor.instructions,
            synthetic_count=work_metadata._max_synthetics,  # noqa
            exception_handlers=bytecode_descriptor.exception_handlers,
            stack_metadata=stack_metadata,
            closure=closure_dict,
            globals=globals_dict,
            free_names=work_metadata.free_names,
        )

        if is_generator_or_async:
            return create_generator_class_from_bytecode(
                out, work_metadata, parameter_cells
            )

        return out


def to_bytecode(function: FunctionType) -> Bytecode | opcode.ClassInfo:
    from ._function import find_used_variables

    source = inspect.getsource(function)
    source = unindent(source)
    function_ast: ast.FunctionDef = cast(ast.FunctionDef, ast.parse(source).body[0])
    used_vars = find_used_variables(function_ast, frozenset())

    closure_dict = {
        function.__code__.co_freevars[i]: function.__closure__[i]
        for i in range(len(function.__code__.co_freevars))
    }
    compiler = AstCompiler()
    return compiler.compile(
        ast_node=function_ast,
        function_name=function.__name__,
        function_type=opcode.FunctionType.for_function_ast(function_ast),
        method_type=opcode.MethodType.for_function_ast(function_ast),
        signature=inspect.signature(
            function,
            globals=function.__globals__,
            locals=function.__globals__,
            eval_str=True,
        ),
        # TODO: Remove __code__ usage
        local_names=frozenset(function.__code__.co_varnames) | used_vars.variable_names,
        free_names=frozenset(function.__code__.co_freevars),
        cell_names=frozenset(
            function.__code__.co_cellvars + function.__code__.co_freevars
        )
        | used_vars.cell_names,
        global_names=frozenset(),
        closure_dict=closure_dict,
        globals_dict=function.__globals__,
    )
