import types

from ._api import (
    AstPass,
    AstCompilable,
    CompileMetadata,
    Compilable,
    Code,
    Bytecode,
    AstPassSharedMetadata,
    BytecodeDescriptor,
    DeferredProcessors,
    ExceptionHandler,
    code_to_bytecode,
)
from .. import opcode as opcode

import ast
from dataclasses import dataclass, field, replace
import inspect
from typing import Iterable, MutableSequence, cast

from ..opcode import StackMetadata, Instruction, SaveGeneratorState


@dataclass
class GeneratorMetadata(AstPassSharedMetadata):
    yield_labels: list[opcode.Label] = field(default_factory=list)

    @property
    def yield_count(self):
        return len(self.yield_labels)

    def add_yield_label(self, label: opcode.Label) -> None:
        self.yield_labels.append(label)


def is_generator_or_async(function_ast: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if isinstance(function_ast, ast.AsyncFunctionDef):
        return True
    return is_generator(function_ast)


def is_generator(function_ast: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    generator_opcodes = ast.Yield, ast.YieldFrom
    for node in ast.walk(function_ast):
        if isinstance(node, generator_opcodes):
            return True
    return False


def generator_function_prologue(out: list[Code], metadata: CompileMetadata) -> None:
    # Make first synthetic store the generator object
    generator_metadata = metadata.get_shared(GeneratorMetadata)
    deferred_processors = metadata.get_shared(DeferredProcessors)

    metadata._synthetic_count += 1
    metadata._max_synthetics += 1
    out.extend(
        [
            opcode.LoadLocal(name="_generator_instance"),
            opcode.StoreSynthetic(index=0),
        ]
    )
    # First opcode is to jump to the corresponding yield
    yield_index = generator_metadata.yield_count
    yield_label = opcode.Label()
    out.append(opcode.JumpTo(target=yield_label))
    generator_metadata.add_yield_label(yield_label)
    yield_label._bytecode_index = 3
    out.extend(
        [
            opcode.LoadSynthetic(index=0),
            opcode.DelegateOrRestoreGeneratorState(
                state_id=yield_index, stack_metadata=cast(StackMetadata, None)
            ),
            opcode.Pop(),
        ]
    )

    def _deferred_processor(
        bytecode_descriptor: BytecodeDescriptor,
        stack_metadata: tuple[StackMetadata, ...],
    ) -> None:
        delegate_opcode = cast(
            opcode.DelegateOrRestoreGeneratorState,
            bytecode_descriptor.instructions[4].opcode,
        )
        delegate_opcode.stack_metadata = stack_metadata[2]

    deferred_processors.add_deferred_processor(_deferred_processor)


def create_generator_class_from_bytecode(
    bytecode: Bytecode,
    compile_metadata: CompileMetadata,
    parameter_cells: frozenset[str],
) -> opcode.ClassInfo:
    # Create a class object that jumps to each yield label
    generator_metadata = compile_metadata.get_shared(GeneratorMetadata)
    generator_class_attributes: dict[str, Bytecode] = dict()
    for yield_index, yield_label in enumerate(generator_metadata.yield_labels):
        new_instructions = list(bytecode.instructions)
        new_instructions[2] = Instruction(
            opcode=opcode.JumpTo(target=yield_label),
            bytecode_index=2,
            lineno=bytecode.instructions[0].lineno,
        )
        method_bytecode = replace(
            bytecode,
            function_type=opcode.FunctionType.FUNCTION,
            method_type=opcode.MethodType.VIRTUAL,
            # TODO: Generate a unique name incase '_generator_instance' was used already
            signature=inspect.Signature(
                [
                    inspect.Parameter(
                        name="_generator_instance",
                        kind=inspect.Parameter.POSITIONAL_ONLY,
                    )
                ]
            ),
            instructions=tuple(new_instructions),
        )
        generator_class_attributes[f"_next_{yield_index}"] = method_bytecode

    init_signature = inspect.Signature(
        parameters=[
            inspect.Parameter(
                name="_generator_instance", kind=inspect.Parameter.POSITIONAL_ONLY
            ),
            *bytecode.signature.parameters.values(),
        ]
    )
    init_instructions = []

    for parameter_cell_var in parameter_cells:
        init_instructions.append(opcode.LoadLocal(name=parameter_cell_var))
        init_instructions.append(
            opcode.StoreCell(name=parameter_cell_var, is_free=False)
        )

    init_instructions.append(opcode.LoadLocal(name="_generator_instance"))
    init_instructions.append(opcode.StoreSynthetic(index=0))
    # Need a value on the stack to match expected stack state for a yield
    init_instructions.append(opcode.LoadConstant(None))
    init_instructions.append(opcode.LoadSynthetic(index=0))
    init_instructions.append(
        opcode.SaveGeneratorState(
            state_id=0,
            stack_metadata=bytecode.stack_metadata[2],
        )
    )
    init_instructions.append(opcode.Pop())
    init_instructions.extend(
        [
            opcode.LoadConstant(None),
            opcode.LoadLocal(name="_generator_instance"),
            opcode.StoreAttr(name="_sent_value"),
            opcode.LoadConstant(None),
            opcode.LoadLocal(name="_generator_instance"),
            opcode.StoreAttr(name="_thrown_value"),
            opcode.LoadConstant(False),
            opcode.LoadLocal(name="_generator_instance"),
            opcode.StoreAttr(name="_generator_finished"),
            opcode.LoadConstant(opcode.GeneratorOperation.NEXT.value),
            opcode.LoadLocal(name="_generator_instance"),
            opcode.StoreAttr(name="_operation"),
            opcode.LoadConstant(None),
            opcode.LoadLocal(name="_generator_instance"),
            opcode.StoreAttr(name="_sub_generator"),
        ]
    )
    init_instructions.append(opcode.LoadConstant(None))
    init_instructions.append(opcode.ReturnValue())
    generator_class_attributes["__init__"] = code_to_bytecode(
        init_instructions, None, "__init__", init_signature, compile_metadata
    )

    iter_signature = inspect.Signature(
        parameters=[
            inspect.Parameter(
                name="_generator_instance", kind=inspect.Parameter.POSITIONAL_ONLY
            )
        ]
    )
    iter_instructions = []
    iter_instructions.extend(
        [opcode.LoadLocal(name="_generator_instance"), opcode.ReturnValue()]
    )
    iter_bytecode = code_to_bytecode(
        iter_instructions, None, "__iter__", iter_signature, compile_metadata
    )

    match bytecode.function_type:
        case opcode.FunctionType.GENERATOR:
            generator_class_attributes["__iter__"] = iter_bytecode
        case opcode.FunctionType.ASYNC_GENERATOR:
            generator_class_attributes["__aiter__"] = iter_bytecode
        case (
            opcode.FunctionType.COROUTINE_GENERATOR
            | opcode.FunctionType.ASYNC_FUNCTION
        ):
            #  Note: technically await should return a different object,
            #        but you cannot reuse an already run-coroutine
            generator_class_attributes["__iter__"] = iter_bytecode
            generator_class_attributes["__await__"] = iter_bytecode
        case _:
            raise RuntimeError(f"Unknown generator type {bytecode.function_type}")

    next_signature = inspect.Signature(
        parameters=[
            inspect.Parameter(
                name="_generator_instance", kind=inspect.Parameter.POSITIONAL_ONLY
            )
        ]
    )
    next_instructions = []
    next_exception_handlers = []
    generator_not_finished_label = opcode.Label()
    next_instructions.extend(
        [
            opcode.LoadLocal(name="_generator_instance"),
            opcode.LoadAttr(name="_generator_finished"),
            opcode.IfFalse(target=generator_not_finished_label),
            opcode.LoadConstant(StopIteration),
            opcode.CreateCallBuilder(),
            opcode.CallWithBuilder(),
            opcode.Raise(),
        ]
    )
    next_instructions.append(generator_not_finished_label)
    next_exception_handler_label = opcode.Label()
    next_exception_handlers.append(
        ExceptionHandler(
            exception_class=BaseException,
            from_label=generator_not_finished_label,
            to_label=next_exception_handler_label,
            handler_label=next_exception_handler_label,
        )
    )
    next_instructions.extend(
        [
            opcode.LoadLocal(name="_generator_instance"),
            opcode.LoadAttr(name="_state_id"),
            opcode.StoreLocal(name="state_id"),
        ]
    )
    # Note: a tuple lookup might be faster
    generator_finished_label = opcode.Label()
    for yield_index, yield_label in enumerate(generator_metadata.yield_labels):
        skip_label = opcode.Label()
        next_instructions.extend(
            [
                opcode.LoadConstant(yield_index),
                opcode.LoadLocal(name="state_id"),
                opcode.BinaryOp(operator=opcode.BinaryOperator.Eq),
                opcode.IfFalse(target=skip_label),
                opcode.LoadLocal(name="_generator_instance"),
                opcode.LoadAttr(name=f"_next_{yield_index}"),
                opcode.CreateCallBuilder(),
                opcode.CallWithBuilder(),
                opcode.StoreLocal(name="return_value"),
                opcode.LoadLocal(name="_generator_instance"),
                opcode.LoadAttr(name="_generator_finished"),
                opcode.IfTrue(target=generator_finished_label),
                opcode.LoadLocal(name="return_value"),
                opcode.ReturnValue(),
            ]
        )
        next_instructions.append(skip_label)
    next_instructions.extend(
        [
            opcode.LoadConstant(SystemError),
            opcode.CreateCallBuilder(),
            opcode.LoadConstant("Generated next method missing branch for state "),
            opcode.LoadLocal(name="state_id"),
            opcode.FormatValue(
                conversion=opcode.FormatConversion.TO_STRING, format_spec=""
            ),
            opcode.BinaryOp(operator=opcode.BinaryOperator.Add),
            opcode.WithPositionalArg(index=0),
            opcode.CallWithBuilder(),
            opcode.Raise(),
        ]
    )
    next_instructions.append(generator_finished_label)
    stop_iteration_has_value_label = opcode.Label()
    next_instructions.extend(
        [
            opcode.LoadConstant(True),
            opcode.LoadLocal(name="_generator_instance"),
            opcode.StoreAttr(name="_generator_finished"),
            opcode.LoadLocal(name="return_value"),
            opcode.LoadConstant(None),
            opcode.IsSameAs(negate=False),
            opcode.IfFalse(target=stop_iteration_has_value_label),
            opcode.LoadConstant(StopIteration),
            opcode.CreateCallBuilder(),
            opcode.CallWithBuilder(),
            opcode.Raise(),
        ]
    )
    next_instructions.append(stop_iteration_has_value_label)
    next_instructions.extend(
        [
            opcode.LoadConstant(StopIteration),
            opcode.CreateCallBuilder(),
            opcode.LoadLocal(name="return_value"),
            opcode.WithPositionalArg(index=0),
            opcode.CallWithBuilder(),
            opcode.Raise(),
        ]
    )
    next_instructions.append(next_exception_handler_label)
    next_instructions.extend(
        [
            opcode.LoadConstant(True),
            opcode.LoadLocal(name="_generator_instance"),
            opcode.StoreAttr(name="_generator_finished"),
            opcode.ReraiseLast(),
        ]
    )
    next_bytecode = code_to_bytecode(
        next_instructions,
        next_exception_handlers,
        "__next__",
        next_signature,
        compile_metadata,
    )

    match bytecode.function_type:
        case (
            opcode.FunctionType.GENERATOR
            | opcode.FunctionType.ASYNC_FUNCTION
            | opcode.FunctionType.COROUTINE_GENERATOR
        ):
            generator_class_attributes["__next__"] = next_bytecode
        case opcode.FunctionType.ASYNC_GENERATOR:
            generator_class_attributes["__anext__"] = next_bytecode
        case _:
            raise RuntimeError(f"Unknown generator type {bytecode.function_type}")

    send_signature = inspect.Signature(
        parameters=[
            inspect.Parameter(
                name="_generator_instance", kind=inspect.Parameter.POSITIONAL_ONLY
            ),
            inspect.Parameter(name="value", kind=inspect.Parameter.POSITIONAL_ONLY),
        ]
    )
    send_instructions = []
    send_instructions.extend(
        [
            opcode.LoadLocal(name="value"),
            opcode.LoadLocal(name="_generator_instance"),
            opcode.StoreAttr(name="_sent_value"),
            opcode.LoadConstant(opcode.GeneratorOperation.SEND.value),
            opcode.LoadLocal(name="_generator_instance"),
            opcode.StoreAttr(name="_operation"),
            opcode.LoadLocal(name="_generator_instance"),
            opcode.LoadAttr(name="__next__"),
            opcode.CreateCallBuilder(),
            opcode.CallWithBuilder(),
            opcode.ReturnValue(),
        ]
    )
    send_bytecode = code_to_bytecode(
        send_instructions, None, "send", send_signature, compile_metadata
    )

    match bytecode.function_type:
        case (
            opcode.FunctionType.GENERATOR
            | opcode.FunctionType.ASYNC_FUNCTION
            | opcode.FunctionType.COROUTINE_GENERATOR
        ):
            generator_class_attributes["send"] = send_bytecode
        case opcode.FunctionType.ASYNC_GENERATOR:
            generator_class_attributes["asend"] = send_bytecode
        case _:
            raise RuntimeError(f"Unknown generator type {bytecode.function_type}")

    throw_instructions = []
    throw_signature = inspect.Signature(
        parameters=[
            inspect.Parameter(
                name="_generator_instance", kind=inspect.Parameter.POSITIONAL_ONLY
            ),
            inspect.Parameter(name="error", kind=inspect.Parameter.POSITIONAL_ONLY),
        ]
    )

    throw_instructions.extend(
        [
            opcode.LoadLocal(name="error"),
            opcode.LoadLocal(name="_generator_instance"),
            opcode.StoreAttr(name="_thrown_value"),
            opcode.LoadConstant(opcode.GeneratorOperation.THROW.value),
            opcode.LoadLocal(name="_generator_instance"),
            opcode.StoreAttr(name="_operation"),
            opcode.LoadLocal(name="_generator_instance"),
            opcode.LoadAttr(name="__next__"),
            opcode.CreateCallBuilder(),
            opcode.CallWithBuilder(),
            opcode.ReturnValue(),
        ]
    )
    throw_bytecode = code_to_bytecode(
        throw_instructions, None, "throw", throw_signature, compile_metadata
    )

    match bytecode.function_type:
        case (
            opcode.FunctionType.GENERATOR
            | opcode.FunctionType.ASYNC_FUNCTION
            | opcode.FunctionType.COROUTINE_GENERATOR
        ):
            generator_class_attributes["throw"] = throw_bytecode
        case opcode.FunctionType.ASYNC_GENERATOR:
            generator_class_attributes["athrow"] = throw_bytecode
        case _:
            raise RuntimeError(f"Unknown generator type {bytecode.function_type}")

    return opcode.ClassInfo(
        name=f"<generator {bytecode.function_name}>",
        qualname=f"<generator {bytecode.function_name}>",
        class_attributes={
            name: types.FunctionType for name in generator_class_attributes.keys()
        },
        instance_attributes={
            "_state_id": int,
            "_saved_state": object,
            "_sent_value": object,
            "_thrown_value": BaseException | None,
            "_sub_generator": Iterable | None,
            "_operation": int,
            "_generator_finished": bool,
        },
        class_attribute_defaults=generator_class_attributes,
    )


def await_tos(out: MutableSequence[Code], compile_metadata: CompileMetadata):
    out.append(opcode.GetAwaitableIterator())
    return yield_from_tos(out, compile_metadata)


def yield_from_tos(out: MutableSequence[Code], compile_metadata: CompileMetadata):
    generator_metadata = compile_metadata.get_shared(GeneratorMetadata)
    deferred_processors = compile_metadata.get_shared(DeferredProcessors)

    yield_id = generator_metadata.yield_count
    out.append(opcode.LoadSynthetic(0))
    out.append(opcode.SetGeneratorDelegate())

    out.append(opcode.LoadConstant(None))
    out.append(opcode.LoadSynthetic(0))

    out.append(
        opcode.SaveGeneratorState(
            state_id=yield_id,
            stack_metadata=cast(StackMetadata, None),
        )
    )

    yield_label = opcode.Label()
    generator_metadata.add_yield_label(yield_label)
    out.append(yield_label)

    out.append(opcode.LoadSynthetic(0))

    out.append(
        opcode.DelegateOrRestoreGeneratorState(
            state_id=yield_id,
            stack_metadata=cast(StackMetadata, None),
        )
    )

    def deferred_processor(
        finalized_bytecode: BytecodeDescriptor,
        bytecode_stack_metadata: tuple[StackMetadata, ...],
    ):
        save_bytecode_index = next(
            ins.bytecode_index
            for ins in finalized_bytecode.instructions
            if isinstance(ins.opcode, SaveGeneratorState)
            and ins.opcode.state_id == yield_id
        )
        restore_bytecode_index = next(
            ins.bytecode_index
            for ins in finalized_bytecode.instructions
            if isinstance(ins.opcode, SaveGeneratorState)
            and ins.opcode.state_id == yield_id
        )
        stack_metadata = bytecode_stack_metadata[save_bytecode_index - 1]
        save_generator_state = cast(
            opcode.SaveGeneratorState,
            finalized_bytecode.instructions[save_bytecode_index].opcode,
        )
        restore_generator_state = cast(
            opcode.DelegateOrRestoreGeneratorState,
            finalized_bytecode.instructions[restore_bytecode_index].opcode,
        )
        save_generator_state.stack_metadata = stack_metadata
        restore_generator_state.stack_metadata = stack_metadata

    deferred_processors.add_deferred_processor(deferred_processor)
    return out


def yield_tos(out: MutableSequence[Code], compile_metadata: CompileMetadata):
    generator_metadata = compile_metadata.get_shared(GeneratorMetadata)
    deferred_processors = compile_metadata.get_shared(DeferredProcessors)

    yield_id = generator_metadata.yield_count
    out.append(opcode.LoadSynthetic(0))

    out.append(
        opcode.SaveGeneratorState(
            state_id=yield_id,
            stack_metadata=cast(StackMetadata, None),
        )
    )

    out.append(opcode.YieldValue())
    yield_label = opcode.Label()
    generator_metadata.add_yield_label(yield_label)

    out.append(yield_label)
    out.append(opcode.LoadSynthetic(0))

    out.append(
        opcode.DelegateOrRestoreGeneratorState(
            state_id=yield_id,
            stack_metadata=cast(StackMetadata, None),
        )
    )

    def deferred_processor(
        finalized_bytecode: BytecodeDescriptor,
        bytecode_stack_metadata: tuple[StackMetadata, ...],
    ):
        save_bytecode_index = next(
            ins.bytecode_index
            for ins in finalized_bytecode.instructions
            if isinstance(ins.opcode, SaveGeneratorState)
            and ins.opcode.state_id == yield_id
        )
        restore_bytecode_index = next(
            ins.bytecode_index
            for ins in finalized_bytecode.instructions
            if isinstance(ins.opcode, SaveGeneratorState)
            and ins.opcode.state_id == yield_id
        )
        stack_metadata = bytecode_stack_metadata[save_bytecode_index - 1]
        save_generator_state = cast(
            opcode.SaveGeneratorState,
            finalized_bytecode.instructions[save_bytecode_index].opcode,
        )
        restore_generator_state = cast(
            opcode.DelegateOrRestoreGeneratorState,
            finalized_bytecode.instructions[restore_bytecode_index].opcode,
        )
        save_generator_state.stack_metadata = stack_metadata
        restore_generator_state.stack_metadata = stack_metadata

    deferred_processors.add_deferred_processor(deferred_processor)
    return out


class GeneratorPass(AstPass):
    def accept(
        self, compile_metadata: CompileMetadata, compilable: Compilable
    ) -> Iterable[Code] | None:
        if not isinstance(compilable, AstCompilable):
            return None

        node = compilable.node
        out: list[Code] = []

        match node:
            case ast.Yield(value=value):
                out.append(AstCompilable(value))
                yield_tos(out, compile_metadata)
                return out

            case ast.YieldFrom(value=value):
                out.append(AstCompilable(value))
                out.append(opcode.GetIterator())
                yield_from_tos(out, compile_metadata)
                return out

            case ast.Await(value=value):
                out.append(AstCompilable(value))
                await_tos(out, compile_metadata)
                return out

            case _:
                return None
