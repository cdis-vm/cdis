from ._api import AstPass, AstCompilable, CompileMetadata, Compilable, Code
from .. import opcode as opcode

import ast
from typing import Iterable


class CollectionPass(AstPass):
    def accept(
        self, compile_metadata: CompileMetadata, compilable: Compilable
    ) -> Iterable[Code] | None:
        if not isinstance(compilable, AstCompilable):
            return None

        node = compilable.node
        out: list[Code] = []
        match node:
            case ast.Slice(lower=start, upper=stop, step=step):
                if start is not None:
                    out.append(AstCompilable(start))
                else:
                    out.append(opcode.LoadConstant(None))

                if stop is not None:
                    out.append(AstCompilable(stop))
                else:
                    out.append(opcode.LoadConstant(None))

                if step is not None:
                    out.append(AstCompilable(step))
                else:
                    out.append(opcode.LoadConstant(None))

                out.append(opcode.BuildSlice())
                return out

            case ast.List(elts):
                out.append(opcode.NewList())
                for elt in elts:
                    match elt:
                        case ast.Starred(value):
                            out.append(AstCompilable(value))
                            out.append(opcode.ListExtend())
                        case _:
                            out.append(AstCompilable(elt))
                            out.append(opcode.ListAppend())
                return out

            case ast.Tuple(elts):
                out.append(opcode.NewList())
                for elt in elts:
                    match elt:
                        case ast.Starred(value):
                            out.append(AstCompilable(value))
                            out.append(opcode.ListExtend())
                        case _:
                            out.append(AstCompilable(elt))
                            out.append(opcode.ListAppend())
                out.append(opcode.ListToTuple())
                return out

            case ast.Set(elts):
                out.append(opcode.NewSet())
                for elt in elts:
                    match elt:
                        case ast.Starred(value):
                            out.append(AstCompilable(value))
                            out.append(opcode.SetUpdate())
                        case _:
                            out.append(AstCompilable(elt))
                            out.append(opcode.SetAdd())
                return out

            case ast.Dict(keys, values):
                out.append(opcode.NewDict())
                for index, value in enumerate(values):
                    key = keys[index]
                    if key is None:
                        out.append(AstCompilable(value))
                        out.append(opcode.DictUpdate())
                    else:
                        out.append(AstCompilable(key))
                        out.append(AstCompilable(value))
                        out.append(opcode.DictPut())
                return out

            case ast.Subscript(value=value, slice=slice_, ctx=ctx):
                out.append(AstCompilable(value))
                out.append(AstCompilable(slice_))
                match ctx:
                    case ast.Load():
                        out.append(opcode.GetItem())
                    case ast.Store():
                        out.append(opcode.SetItem())
                    case ast.Delete():
                        out.append(opcode.DeleteItem())
                    case _:
                        raise NotImplementedError(f"Unhandled Subscript ctx: {ctx}")
                return out
            case _:
                return None
