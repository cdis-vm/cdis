from . import code_to_bytecode
from ._api import AstPass, AstCompilable, CompileMetadata, Compilable, Code
from dataclasses import dataclass

from ._variable import with_assignment_code
from .. import opcode as opcode

import ast
import inspect
from typing import Iterable, Callable


@dataclass(frozen=True)
class UsedVariables:
    variable_names: frozenset[str]
    cell_names: frozenset[str]


@dataclass(frozen=True)
class SignatureAndParameters:
    signature: inspect.Signature
    parameters_with_defaults: tuple[str, ...]


def get_signature_from_arguments(args: ast.arguments) -> SignatureAndParameters:
    parameters = []
    parameters_with_defaults = []

    # TODO: Evaluate type from annotation

    for param in args.args:
        parameters.append(
            inspect.Parameter(
                name=param.arg,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=param.annotation,
            )
        )
    for param in args.posonlyargs:
        parameters.append(
            inspect.Parameter(
                name=param.arg,
                kind=inspect.Parameter.POSITIONAL_ONLY,
                annotation=param.annotation,
            )
        )
    if args.vararg:
        parameters.append(
            inspect.Parameter(
                name=args.vararg.arg,
                kind=inspect.Parameter.VAR_POSITIONAL,
                annotation=args.vararg.annotation,
            )
        )
    for param in args.kwonlyargs:
        parameters.append(
            inspect.Parameter(
                name=param.arg,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=param.annotation,
            )
        )
    if args.kwarg:
        parameters.append(
            inspect.Parameter(
                name=args.vararg.arg,
                kind=inspect.Parameter.VAR_KEYWORD,
                annotation=args.vararg.annotation,
            )
        )
    signature = inspect.Signature(parameters)
    remaining_pos_only = len(args.posonlyargs)
    remaining_pos_or_kw = len(args.args)
    for i in range(len(args.defaults)):
        if remaining_pos_only > 0:
            parameters_with_defaults.append(
                args.posonlyargs[remaining_pos_only - 1].arg
            )
            remaining_pos_only -= 1
        else:
            parameters_with_defaults.append(args.args[remaining_pos_or_kw - 1].arg)
            remaining_pos_or_kw -= 1

    for i in range(len(args.kw_defaults)):
        if args.kw_defaults[i] is not None:
            parameters_with_defaults.append(args.kwonlyargs[i].arg)

    return SignatureAndParameters(
        signature=signature, parameters_with_defaults=tuple(parameters_with_defaults)
    )


def find_used_variables(
    func_def: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda,
    outer_variables: frozenset[str],
) -> UsedVariables:
    variable_names = set()
    cell_names = set()

    def add_names_to_variable_names(assignment_target):
        match assignment_target:
            case ast.Name(id=name):
                variable_names.add(name)
                if name in outer_variables:
                    cell_names.add(name)

            case ast.Attribute():
                pass

            case ast.Subscript():
                pass

            case ast.Starred(value=inner):
                add_names_to_variable_names(inner)

            case ast.List(elts=elts):
                for list_item in elts:
                    add_names_to_variable_names(list_item)

            case ast.Tuple(elts=elts):
                for list_item in elts:
                    add_names_to_variable_names(list_item)

            case _:
                raise NotImplementedError(
                    f"Not implemented assignment: {type(assignment_target)}"
                )

    def iterate_node(node, all_is_outer: bool):
        if isinstance(node, ast.ClassDef):
            return

        for child in ast.iter_child_nodes(node):
            match child:
                case ast.Assign(targets):
                    for target in targets:
                        add_names_to_variable_names(target)

                case ast.Name(id=name):
                    if all_is_outer or name in outer_variables:
                        variable_names.add(name)
                        cell_names.add(name)

                case _:
                    iterate_node(child, all_is_outer)

        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                iterate_node(node.args, True)
                if node.returns is not None:
                    iterate_node(node.returns, True)

                variable_names.add(node.name)
                if node.name in outer_variables:
                    cell_names.add(node.name)
            case ast.ClassDef():
                variable_names.add(node.name)
                if node.name in outer_variables:
                    cell_names.add(node.name)
            case ast.Lambda():
                iterate_node(node.args, True)

    match func_def:
        # Do not capture body; that make it a str instead of an ast object!
        case ast.FunctionDef() | ast.AsyncFunctionDef():
            for statement in func_def.body:
                iterate_node(statement, False)
            for value in _get_arg_dict(func_def).values():
                iterate_node(value, False)

        case ast.Lambda():
            iterate_node(func_def.body, False)
            for value in _get_arg_dict(func_def).values():
                iterate_node(value, False)

        case ast.ClassDef():
            for statement in func_def.body:
                iterate_node(statement, False)

        case _:
            raise NotImplementedError(
                f"Not implemented function ast type: {type(func_def)}"
            )

    match func_def:
        case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.Lambda():
            for arg in func_def.args.args:
                variable_names.add(arg.arg)
            for arg in func_def.args.posonlyargs:
                variable_names.add(arg.arg)
            for arg in func_def.args.kwonlyargs:
                variable_names.add(arg.arg)
            if func_def.args.vararg is not None:
                variable_names.add(func_def.args.vararg.arg)
            if func_def.args.kwarg is not None:
                variable_names.add(func_def.args.kwarg.arg)
        case ast.ClassDef():
            pass
        case _:
            raise RuntimeError(f"Not implemented function ast type: {type(func_def)}")

    return UsedVariables(
        variable_names=frozenset(variable_names), cell_names=frozenset(cell_names)
    )


def _get_arg_dict(
    func_def: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef,
) -> dict[str, ast.expr]:
    if isinstance(func_def, ast.ClassDef):
        return {}  # TODO: get attribute type hints

    arg_info = func_def.args
    all_args = [*arg_info.posonlyargs, *arg_info.args, *arg_info.kwonlyargs]

    if arg_info.vararg is not None:
        all_args.append(arg_info.vararg)

    if arg_info.kwarg is not None:
        all_args.append(arg_info.kwarg)

    out = {arg.arg: arg.annotation for arg in all_args if arg.annotation is not None}

    if hasattr(func_def, "returns"):
        if func_def.returns is not None:
            out["return"] = func_def.returns

    return out


def _create_annotation_dict(
    out: list[Code],
    all_args: dict[str, ast.expr],
    value_converter: Callable[[list[Code], ast.expr], None],
) -> None:
    out.append(opcode.NewDict())
    for arg, annotation in all_args.items():
        out.append(opcode.LoadConstant(arg))
        value_converter(out, annotation)
        out.append(opcode.DictPut())

    out.append(opcode.ReturnValue())


def create_annotation_function(
    func_def: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef,
    compile_metadata: CompileMetadata,
):
    instructions: list[Code] = []
    annotate_signature = inspect.Signature(
        parameters=(
            inspect.Parameter("format", kind=inspect.Parameter.POSITIONAL_ONLY),
        )
    )
    arg_info = _get_arg_dict(func_def)
    check_format_forward_ref = opcode.Label()
    check_format_source = opcode.Label()
    fail_label = opcode.Label()
    instructions.extend(
        [
            opcode.LoadLocal(name="format"),
            opcode.LoadConstant(1),  # value
            opcode.BinaryOp(operator=opcode.BinaryOperator.Eq),
            opcode.IfFalse(target=check_format_forward_ref),
        ]
    )

    _create_annotation_dict(
        instructions, arg_info, lambda b, e: b.append(AstCompilable(e))
    )

    instructions.append(check_format_forward_ref)
    instructions.extend(
        [
            opcode.LoadLocal(name="format"),
            opcode.LoadConstant(2),  # forward_ref
            opcode.BinaryOp(operator=opcode.BinaryOperator.Eq),
            opcode.IfFalse(target=check_format_source),
        ]
    )
    instructions.extend([opcode.LoadConstant(NotImplementedError), opcode.Raise()])

    instructions.append(check_format_source)
    instructions.extend(
        [
            opcode.LoadLocal(name="format"),
            opcode.LoadConstant(3),  # source
            opcode.BinaryOp(operator=opcode.BinaryOperator.Eq),
            opcode.IfFalse(target=fail_label),
        ]
    )
    instructions.extend([opcode.LoadConstant(NotImplementedError), opcode.Raise()])

    instructions.append(fail_label)
    instructions.extend([opcode.LoadConstant(NotImplementedError), opcode.Raise()])
    return code_to_bytecode(
        instructions,
        None,
        "__annotate__",
        annotate_signature,
        compile_metadata.blank_copy(),
    )


class FunctionPass(AstPass):
    def accept(
        self, compile_metadata: CompileMetadata, compilable: Compilable
    ) -> Iterable[Code] | None:
        if not isinstance(compilable, AstCompilable):
            return None

        node = compilable.node
        out: list[Code] = []

        match node:
            case ast.Call(func, args, keywords):
                out.append(AstCompilable(func))
                out.append(opcode.CreateCallBuilder())
                extended = False
                arg_index = 0
                for arg in args:
                    match arg:
                        case ast.Starred(value):
                            out.append(AstCompilable(value))
                            out.append(opcode.ExtendPositionalArgs())
                            extended = True
                        case _ as expression:
                            out.append(AstCompilable(expression))
                            if extended:
                                out.append(opcode.AppendPositionalArg())
                            else:
                                out.append(opcode.WithPositionalArg(arg_index))
                                arg_index += 1

                for kwarg in keywords:
                    out.append(AstCompilable(kwarg.value))
                    if kwarg.arg is None:
                        out.append(opcode.ExtendKeywordArgs())
                    else:
                        out.append(opcode.WithKeywordArg(kwarg.arg))

                out.append(opcode.CallWithBuilder())
                return out

            case ast.Lambda(args):
                inner_function = compile_metadata.compile_inner_function(node)
                for default_arg in args.defaults:
                    out.append(AstCompilable(default_arg))
                for default_arg in args.kw_defaults:
                    out.append(AstCompilable(default_arg))
                out.append(opcode.LoadAndBindInnerFunction(inner_function))
                return out

            case ast.FunctionDef() as func_def:
                inner_function = compile_metadata.compile_inner_function(func_def)
                for default_arg in func_def.args.defaults:
                    out.append(AstCompilable(default_arg))
                for default_arg in func_def.args.kw_defaults:
                    out.append(AstCompilable(default_arg))
                out.append(opcode.LoadAndBindInnerFunction(inner_function))
                for decorator in func_def.decorator_list:
                    out.append(AstCompilable(decorator))
                    out.extend(
                        [
                            opcode.CreateCallBuilder(),
                            opcode.Swap(),
                            opcode.WithPositionalArg(index=0),
                            opcode.CallWithBuilder(),
                        ]
                    )
                with_assignment_code(
                    out, ast.Name(id=func_def.name, ctx=ast.Store), compile_metadata
                )
                return out

            case _:
                return None
