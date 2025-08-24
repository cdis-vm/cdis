from ._api import AstPass, AstCompilable, CompileMetadata, Compilable, Code
from .. import opcode as opcode

import ast
from typing import Iterable


class ObjectPass(AstPass):
    def accept(
        self, compile_metadata: CompileMetadata, compilable: Compilable
    ) -> Iterable[Code] | None:
        if not isinstance(compilable, AstCompilable):
            return None

        node = compilable.node
        out: list[Code] = []

        match node:
            case ast.Attribute(attr=attr_name) as a:
                out.append(AstCompilable(a.value))
                out.append(opcode.LoadAttr(attr_name))
                return out
            case _:
                return None
