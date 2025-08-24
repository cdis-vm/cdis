from ._api import AstPass, AstCompilable, CompileMetadata, Compilable, Code
from .. import opcode as opcode

import ast
from typing import Iterable


def get_comparison_op(cmpop: ast.cmpop) -> opcode.Opcode:
    match cmpop:
        case ast.Is():
            return opcode.IsSameAs(negate=False)
        case ast.IsNot():
            return opcode.IsSameAs(negate=True)
        case ast.In():
            return opcode.IsContainedIn(negate=False)
        case ast.NotIn():
            return opcode.IsContainedIn(negate=True)
        case _:
            operator = opcode.BinaryOperator[type(cmpop).__name__]
            return opcode.BinaryOp(operator)


class OperationPass(AstPass):
    def accept(
        self, compile_metadata: CompileMetadata, compilable: Compilable
    ) -> Iterable[Code] | None:
        if not isinstance(compilable, AstCompilable):
            return None

        node = compilable.node
        match node:
            case ast.Not():
                if_true_label = opcode.Label()
                done_label = opcode.Label()
                return [
                    opcode.IfTrue(if_true_label),
                    opcode.LoadConstant(True),
                    opcode.JumpTo(done_label),
                    if_true_label,
                    opcode.LoadConstant(False),
                    done_label,
                ]

            case ast.UnaryOp(op=op, operand=operand):
                return [
                    AstCompilable(operand),
                    opcode.UnaryOp(opcode.UnaryOperator[type(op).__name__]),
                ]

            case ast.BinOp(left, op, right):
                return [
                    AstCompilable(left),
                    AstCompilable(right),
                    opcode.BinaryOp(opcode.BinaryOperator[type(op).__name__]),
                ]

            case ast.Compare(left, ops, comparators):
                out: list[Code] = [AstCompilable(left)]
                false_label = opcode.Label()
                for i in range(len(ops) - 1):
                    out.extend(
                        [
                            AstCompilable(comparators[i]),
                            opcode.DupX1(),
                            get_comparison_op(ops[i]),
                            opcode.Dup(),
                            opcode.IfFalse(false_label),
                            opcode.Pop(),
                        ]
                    )

                out.extend(
                    [
                        AstCompilable(comparators[-1]),
                        get_comparison_op(ops[-1]),
                    ]
                )
                done_label = opcode.Label()
                if len(ops) > 1:
                    out.extend(
                        [
                            opcode.JumpTo(done_label),
                            false_label,
                            opcode.Swap(),
                            opcode.Pop(),
                            done_label,
                        ]
                    )

                return out

            case ast.BoolOp(op, values):
                match op:
                    case ast.And():
                        jump_opcode = opcode.IfFalse
                    case ast.Or():
                        jump_opcode = opcode.IfTrue
                    case _:
                        raise ValueError(f"Unhandled BoolOp {op}")

                out: list[Code] = []
                end_label = opcode.Label()
                for value in values[:-1]:
                    out.extend(
                        [
                            AstCompilable(value),
                            opcode.Dup(),
                            jump_opcode(end_label),
                            opcode.Pop(),
                        ]
                    )

                out.extend([AstCompilable(values[-1]), end_label])
                return out

            case ast.FormattedValue(value, conversion, format_spec):
                out: list[Code] = [
                    AstCompilable(value),
                    opcode.FormatValue(
                        conversion=opcode.FormatConversion.from_int(conversion),
                        # format_spec should always be a constant string or None
                        format_spec=format_spec.values[0].value
                        if format_spec is not None
                        else "",
                    ),
                ]
                return out

            case ast.JoinedStr(values):
                out: list[Code] = []
                for value in values:
                    out.append(AstCompilable(value))
                out.append(opcode.JoinStringValues(count=len(values)))
                return out
            case _:
                return None
