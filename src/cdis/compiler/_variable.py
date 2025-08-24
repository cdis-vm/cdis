from ._api import (
    AstPass,
    AstCompilable,
    AstPassSharedMetadata,
    CompileMetadata,
    Compilable,
    Code,
    compile_context_manager,
    use_synthetics,
)
from .. import opcode as opcode

import ast
from dataclasses import dataclass
from typing import Iterable, MutableSequence, ContextManager


def _impossible_state(msg):
    raise RuntimeError(f"Impossible state: {msg}")


@dataclass(frozen=True)
class AssignmentCode:
    stack_items: int
    load_target: Iterable[Code]
    load_current_value: Iterable[Code]
    store_value: Iterable[Code]
    delete_value: Iterable[Code]


@dataclass(frozen=True)
class RenamedVariable:
    read_opcode: opcode.Opcode
    write_opcode: opcode.Opcode
    delete_opcode: opcode.Opcode


class RenamedVariables(AstPassSharedMetadata):
    renamed_variables: dict[str, RenamedVariable]

    def __init__(self):
        self.renamed_variables = {}

    def use_renames(
        self, target: ast.AST, out: MutableSequence[Code], metadata: CompileMetadata
    ) -> ContextManager["RenamedVariables"]:
        new_renamed_variables, new_count = self._renames_for_target(target, metadata)
        old = self.renamed_variables

        def before(_metadata: CompileMetadata):
            nonlocal self
            self.renamed_variables = new_renamed_variables
            _metadata._synthetic_count += new_count
            _metadata._max_synthetics = max(
                _metadata._max_synthetics, _metadata._synthetic_count
            )

        def after(_metadata: CompileMetadata):
            nonlocal self
            self.renamed_variables = old
            _metadata._synthetic_count -= new_count

        return compile_context_manager(self, out, metadata, before, after)

    def _renames_for_target(
        self, target: ast.AST, metadata: CompileMetadata
    ) -> tuple[dict[str, RenamedVariable], int]:
        match target:
            case ast.Name(id=name):
                renamed_variable = RenamedVariable(
                    read_opcode=opcode.LoadSynthetic(metadata.next_synthetic),
                    write_opcode=opcode.StoreSynthetic(metadata.next_synthetic),
                    # TODO: is DeleteSynthetic needed?
                    delete_opcode=opcode.Nop(),
                )
                return {**self.renamed_variables, name: renamed_variable}, 1

            case ast.Attribute():
                return self.renamed_variables, 0

            case ast.Subscript():
                return self.renamed_variables, 0

            case ast.Starred(value=inner):
                return self._renames_for_target(inner, metadata)

            case ast.List(elts=elts, ctx=ctx):
                return self._renames_for_target(ast.Tuple(elts=elts, ctx=ctx), metadata)

            case ast.Tuple(elts=elts):
                total_count = 0
                new_renames = {**self.renamed_variables}
                for elt in elts:
                    new_renames, added_count = self._renames_for_target(elt, metadata)
                    total_count += added_count
                return new_renames, total_count

            case _:
                raise NotImplementedError(f"Not implemented assignment: {type(target)}")

        raise RuntimeError(f"Missing return in case {type(target)} in _get_renames")


def with_assignment_code(
    out: MutableSequence[Code], target: ast.expr, metadata: CompileMetadata
) -> None:
    code = get_assignment_code(target, metadata)
    out.extend(code.load_target)
    out.extend(code.store_value)


def get_assignment_code(
    expression: ast.expr, metadata: CompileMetadata
) -> AssignmentCode:
    renames = metadata.get_shared(RenamedVariables()).renamed_variables
    match expression:
        case ast.Name(id=name):
            if name in renames:
                return AssignmentCode(
                    stack_items=0,
                    load_target=(),
                    load_current_value=(renames[name].read_opcode,),
                    store_value=(renames[name].write_opcode,),
                    delete_value=(renames[name].delete_opcode,),
                )
            elif name in metadata.cell_names:
                return AssignmentCode(
                    stack_items=0,
                    load_target=(),
                    load_current_value=(
                        opcode.LoadCell(name, name in metadata.free_names),
                    ),
                    store_value=(opcode.StoreCell(name, name in metadata.free_names),),
                    delete_value=(
                        opcode.DeleteCell(name, name in metadata.free_names),
                    ),
                )
            elif name in metadata.local_names:
                if metadata.function_type is not opcode.FunctionType.CLASS_BODY:
                    return AssignmentCode(
                        stack_items=0,
                        load_target=(),
                        load_current_value=(opcode.LoadLocal(name),),
                        store_value=(opcode.StoreLocal(name),),
                        delete_value=(opcode.DeleteLocal(name),),
                    )
                else:
                    # class body uses a mapping stored in the first synthetic
                    return AssignmentCode(
                        stack_items=0,
                        load_target=(),
                        load_current_value=(
                            opcode.LoadSynthetic(0),
                            opcode.LoadTypeAttrOrGlobal(name),
                        ),
                        store_value=(
                            opcode.LoadSynthetic(0),
                            opcode.StoreTypeAttr(name),
                        ),
                        delete_value=(
                            opcode.LoadSynthetic(0),
                            opcode.DeleteTypeAttr(name),
                        ),
                    )
            elif (
                name in metadata.global_names
                or metadata.function_type is not opcode.FunctionType.CLASS_BODY
            ):
                return AssignmentCode(
                    stack_items=0,
                    load_target=(),
                    load_current_value=(opcode.LoadGlobal(name),),
                    store_value=(opcode.StoreGlobal(name),),
                    delete_value=(opcode.DeleteGlobal(name),),
                )
            else:
                return AssignmentCode(
                    stack_items=0,
                    load_target=(),
                    load_current_value=(
                        opcode.LoadSynthetic(0),
                        opcode.LoadTypeAttrOrGlobal(name),
                    ),
                    store_value=(opcode.LoadSynthetic(0), opcode.StoreTypeAttr(name)),
                    delete_value=(opcode.LoadSynthetic(0), opcode.DeleteTypeAttr(name)),
                )

        case ast.Attribute(attr=attr_name) as a:
            return AssignmentCode(
                stack_items=1,
                load_target=(AstCompilable(a.value),),
                load_current_value=(opcode.LoadAttr(attr_name),),
                store_value=(opcode.StoreAttr(attr_name),),
                delete_value=(opcode.DeleteAttr(attr_name),),
            )

        case ast.Subscript(value, slice=slice_):
            return AssignmentCode(
                stack_items=2,
                load_target=(AstCompilable(value), AstCompilable(slice_)),
                load_current_value=(opcode.GetItem(),),
                store_value=(opcode.SetItem(),),
                delete_value=(opcode.DeleteItem(),),
            )

        case ast.Starred(value=inner):
            return get_assignment_code(inner, metadata)

        case ast.List(elts, ctx):
            return get_assignment_code(ast.Tuple(elts=elts, ctx=ctx), metadata)

        case ast.Tuple(elts):

            def store_elts() -> tuple[opcode.Opcode | Compilable, ...]:
                out = []
                star_index = None
                for index, elet in enumerate(elts):
                    if isinstance(elet, ast.Starred):
                        star_index = index
                        break
                if star_index is None:
                    before_count = len(elts)
                    after_count = 0
                else:
                    before_count = star_index
                    after_count = len(elts) - star_index - 1

                out.append(
                    opcode.UnpackElements(
                        before_count=before_count,
                        has_extras=star_index is not None,
                        after_count=after_count,
                    )
                )
                for elt in elts:
                    assignment_code = get_assignment_code(elt, metadata)
                    out.extend(assignment_code.load_target)
                    out.extend(assignment_code.store_value)

                return tuple(out)

            def delete_elts() -> tuple[opcode.Opcode | Compilable, ...]:
                out: list[opcode.Opcode | Compilable, ...] = []  # noqa
                for elt in elts:
                    assignment_code = get_assignment_code(elt, metadata)
                    out.extend(assignment_code.load_target)
                    out.extend(assignment_code.delete_value)
                return tuple(out)

            return AssignmentCode(
                stack_items=0,
                load_target=(),
                load_current_value=(),
                store_value=store_elts(),
                delete_value=delete_elts(),
            )

        case _:
            raise NotImplementedError(f"Not implemented assignment: {type(expression)}")

    raise RuntimeError(
        f"Missing return in case {type(expression)} in get_assignment_code"
    )


class VariablePass(AstPass):
    def accept(
        self, metadata: CompileMetadata, compilable: Compilable
    ) -> Iterable[Code] | None:
        if not isinstance(compilable, AstCompilable):
            return None

        out: list[Code] = []
        match compilable.node:
            case ast.Global():
                return []

            case ast.Nonlocal():
                return []

            case ast.Assign(targets=targets, value=value):
                out = [AstCompilable(value)]
                for target in targets[:-1]:
                    out.append(opcode.Dup())
                    code = get_assignment_code(target, metadata)
                    out.extend(code.load_target)
                    out.extend(code.store_value)
                code = get_assignment_code(targets[-1], metadata)
                out.extend(code.load_target)
                out.extend(code.store_value)
                return out

            case ast.AugAssign(target, op, value):
                assignment_code = get_assignment_code(target, metadata)
                out.extend(assignment_code.load_target)

                with use_synthetics(
                    assignment_code.stack_items, out, metadata
                ) as stack_items:
                    for stack_variable in stack_items:
                        out.append(opcode.StoreSynthetic(stack_variable))

                for stack_variable in reversed(stack_items):
                    out.append(opcode.LoadSynthetic(stack_variable))

                out.extend(assignment_code.load_current_value)
                out.append(AstCompilable(value))
                out.append(
                    opcode.BinaryOp(opcode.BinaryOperator["I" + type(op).__name__])
                )

                for stack_variable in reversed(stack_items):
                    out.append(opcode.LoadSynthetic(stack_variable))

                out.extend(assignment_code.store_value)
                return out

            case ast.Name(id=name, ctx=ast.Store()):
                out = []
                code = get_assignment_code(compilable.node, metadata)
                out.extend(code.load_target)
                out.extend(code.store_value)
                return out

            case ast.Name(id=name, ctx=ast.Load()):
                renames = metadata.get_shared(RenamedVariables())
                if name in renames.renamed_variables:
                    return [renames.renamed_variables[name].read_opcode]
                if name in metadata.cell_names:
                    return [opcode.LoadCell(name, name in metadata.free_names)]
                elif (
                    name in metadata.local_names
                    and metadata.function_type is not opcode.FunctionType.CLASS_BODY
                ):
                    return [opcode.LoadLocal(name)]
                elif (
                    name in metadata.global_names
                    or metadata.function_type is not opcode.FunctionType.CLASS_BODY
                ):
                    return [opcode.LoadGlobal(name)]
                else:
                    return [
                        opcode.LoadSynthetic(0),
                        opcode.LoadTypeAttrOrGlobal(name=name),
                    ]

            case ast.NamedExpr(target, value):
                out.extend([AstCompilable(value), opcode.Dup()])
                with_assignment_code(out, target, metadata)
                return out

            case ast.Delete(targets):
                for target in targets:
                    assignment_code = get_assignment_code(target, metadata)
                    out.extend(assignment_code.load_target)
                    out.extend(assignment_code.delete_value)
                return out

        return None


__all__ = ("VariablePass", "RenamedVariables")
