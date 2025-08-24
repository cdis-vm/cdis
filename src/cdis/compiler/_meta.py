from ._api import AstPass, AstCompilable, CompileMetadata, Compilable, Code
from ._variable import with_assignment_code
from .. import opcode as opcode

import ast
from typing import Iterable


class MetaPass(AstPass):
    def accept(
        self, compile_metadata: CompileMetadata, compilable: Compilable
    ) -> Iterable[Code] | None:
        if not isinstance(compilable, AstCompilable):
            return None

        node = compilable.node
        match node:
            case ast.Constant(value=value):
                return [opcode.LoadConstant(value)]
            case ast.Pass():
                return [opcode.Nop()]
            case ast.Expr(value=value):
                return [AstCompilable(value), opcode.Pop()]
            case ast.Import(names=aliases):
                out: list[Code] = []
                for alias in aliases:
                    out.append(
                        opcode.ImportModule(name=alias.name, level=0, from_list=()),
                    )
                    asname = alias.name.split(".")[0]
                    if alias.asname is not None:
                        asname = alias.asname

                    with_assignment_code(
                        out, ast.Name(id=asname, ctx=ast.Store), compile_metadata
                    )
                return out

            case ast.ImportFrom(module=module, names=aliases, level=level):
                out: list[Code] = [
                    opcode.ImportModule(
                        name=module,
                        level=level,
                        from_list=tuple(alias.name for alias in aliases),
                    )
                ]
                for alias in aliases:
                    asname = alias.asname if alias.asname is not None else alias.name
                    out.extend(
                        [
                            opcode.Dup(),
                            opcode.LoadAttr(
                                name=alias.name,
                            ),
                        ]
                    )
                    with_assignment_code(
                        out, ast.Name(id=asname, ctx=ast.Store), compile_metadata
                    )
                out.append(opcode.Pop())
                return out
            case _:
                return None
