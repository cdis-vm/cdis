import inspect

from . import code_to_bytecode
from ._api import (
    AstPass,
    AstCompilable,
    CompileMetadata,
    Compilable,
    Code,
    use_synthetics,
)
from ._generator import (
    generator_function_prologue,
    yield_tos,
    create_generator_class_from_bytecode,
)
from ._variable import RenamedVariables
from .. import opcode as opcode

import ast
from typing import Iterable, Callable, MutableSequence

from ..opcode import FunctionType


def get_comprehension_code(
    out: list[Code],
    elt_consumer: Callable[[MutableSequence[Code]], None],
    elts: tuple[ast.expr, ...],
    generators: list[ast.comprehension],
    metadata: CompileMetadata,
):
    end_label = opcode.Label()
    previous_generator_next_label = opcode.Label()
    cleanups = [
        with_comprehension_opcodes(
            out, metadata, generators[0], previous_generator_next_label, end_label
        )
    ]
    for generator in generators[1:]:
        new_generator_next_label = opcode.Label()
        cleanups.append(
            with_comprehension_opcodes(
                out,
                metadata,
                generator,
                new_generator_next_label,
                previous_generator_next_label,
            )
        )
        previous_generator_next_label = new_generator_next_label

    for elt in elts:
        out.append(AstCompilable(node=elt))

    elt_consumer(out)
    out.append(opcode.JumpTo(target=previous_generator_next_label))
    out.append(end_label)
    for cleanup in reversed(cleanups):
        cleanup()

    return out


def with_comprehension_opcodes(
    out: list[Code],
    metadata: CompileMetadata,
    generator: ast.comprehension,
    iterator_next_label: opcode.Label,
    iterator_exhausted_label: opcode.Label,
) -> Callable[[], None]:
    renames = metadata.get_shared(RenamedVariables())
    out.append(AstCompilable(generator.iter))
    new_synthetics = use_synthetics(1, out, metadata)
    [iterator_var] = new_synthetics.__enter__()

    out.extend(
        [
            opcode.GetIterator(),
            opcode.StoreSynthetic(iterator_var),
            iterator_next_label,
            opcode.LoadSynthetic(iterator_var),
            opcode.GetNextElseJumpTo(target=iterator_exhausted_label),
        ]
    )
    for condition in generator.ifs:
        out.append(AstCompilable(condition))
        out.append(opcode.IfFalse(target=iterator_next_label))

    rename_manager = renames.use_renames(generator.target, out, metadata)
    rename_manager.__enter__()
    out.append(AstCompilable(generator.target))

    def cleanup():
        rename_manager.__exit__(None, None, None)
        new_synthetics.__exit__(None, None, None)

    return cleanup


class ComprehensionPass(AstPass):
    def accept(
        self, metadata: CompileMetadata, compilable: Compilable
    ) -> Iterable[Code] | None:
        if not isinstance(compilable, AstCompilable):
            return None

        node = compilable.node
        out: list[Code] = []
        match node:
            case ast.GeneratorExp(elt=elt, generators=generators):
                generator_metadata = metadata.blank_copy()
                generator_metadata.local_names = metadata.local_names | {"generator"}
                generator_metadata.function_type = FunctionType.GENERATOR
                generator_instructions = []
                generator_function_prologue(generator_instructions, generator_metadata)

                def yield_value(_out: MutableSequence[Code]) -> None:
                    yield_tos(_out, generator_metadata)
                    _out.append(opcode.Pop())

                get_comprehension_code(
                    generator_instructions,
                    yield_value,
                    (elt,),
                    [
                        ast.comprehension(
                            target=generators[0].target,
                            iter=ast.Name(id="generator", ctx=ast.Load()),
                            ifs=generators[0].ifs,
                            is_async=generators[0].is_async,
                        )
                    ]
                    + generators[1:],
                    generator_metadata,
                )
                generator_instructions.append(
                    AstCompilable(ast.Return(ast.Constant(None)))
                )

                generator_bytecode = code_to_bytecode(
                    generator_instructions,
                    None,
                    "<gen_expr>",
                    inspect.Signature(
                        parameters=(
                            inspect.Parameter(
                                "generator", kind=inspect.Parameter.POSITIONAL_ONLY
                            ),
                        )
                    ),
                    generator_metadata,
                    function_type=FunctionType.GENERATOR,
                )

                generator_class = create_generator_class_from_bytecode(
                    generator_bytecode, generator_metadata, frozenset()
                )

                out.extend(
                    [
                        opcode.LoadAndBindInnerGenerator(generator_class),
                        opcode.CreateCallBuilder(),
                    ]
                )
                # First generator must be evaluated before generator is created
                out.append(AstCompilable(generators[0].iter))
                out.extend(
                    [
                        opcode.GetIterator(),
                        opcode.WithPositionalArg(index=0),
                        opcode.CallWithBuilder(),
                    ]
                )

                return out

            case ast.ListComp(elt, generators):
                out.append(opcode.NewList())
                with use_synthetics(1, out, metadata) as [list_variable_index]:
                    out.append(opcode.StoreSynthetic(list_variable_index))

                    def add_to_list(new_bytecode: MutableSequence[Code]) -> None:
                        new_bytecode.extend(
                            [
                                opcode.LoadSynthetic(list_variable_index),
                                opcode.Swap(),
                                opcode.ListAppend(),
                                opcode.Pop(),
                            ]
                        )

                    get_comprehension_code(
                        out, add_to_list, (elt,), generators, metadata
                    )
                    out.append(opcode.LoadSynthetic(list_variable_index))
                    return out

            case ast.SetComp(elt, generators):
                out.append(opcode.NewSet())
                with use_synthetics(1, out, metadata) as [set_variable_index]:
                    out.append(opcode.StoreSynthetic(set_variable_index))

                    def add_to_set(new_bytecode: MutableSequence[Code]) -> None:
                        new_bytecode.extend(
                            [
                                opcode.LoadSynthetic(set_variable_index),
                                opcode.Swap(),
                                opcode.SetAdd(),
                                opcode.Pop(),
                            ]
                        )

                    get_comprehension_code(
                        out, add_to_set, (elt,), generators, metadata
                    )
                    out.append(opcode.LoadSynthetic(set_variable_index))
                    return out

            case ast.DictComp(key, value, generators):
                out.append(opcode.NewDict())
                with use_synthetics(1, out, metadata) as [dict_variable_index]:
                    out.append(opcode.StoreSynthetic(dict_variable_index))

                    def add_to_dict(new_bytecode: MutableSequence[Code]) -> None:
                        with use_synthetics(1, out, metadata) as [value_variable_index]:
                            new_bytecode.extend(
                                [
                                    opcode.StoreSynthetic(value_variable_index),
                                    opcode.LoadSynthetic(dict_variable_index),
                                    opcode.Swap(),
                                    opcode.LoadSynthetic(value_variable_index),
                                    opcode.DictPut(),
                                    opcode.Pop(),
                                ]
                            )

                    get_comprehension_code(
                        out, add_to_dict, (key, value), generators, metadata
                    )
                    out.append(opcode.LoadSynthetic(dict_variable_index))
                    return out

            case _:
                return None
