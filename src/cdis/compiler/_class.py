from ._api import AstPass, AstCompilable, CompileMetadata, Compilable, Code
from ._variable import with_assignment_code
from .. import opcode as opcode

import ast
from typing import Iterable, MutableSequence


def class_body_prologue(
    out: MutableSequence[Code], compile_metadata: CompileMetadata
) -> None:
    # Make first synthetic store the class attribute mapping
    compile_metadata._synthetic_count += 1
    compile_metadata._max_synthetics += 1
    out.extend(
        [
            opcode.LoadLocal(name="_attribute_mapping"),
            opcode.StoreSynthetic(index=0),
        ]
    )


class ClassPass(AstPass):
    def accept(
        self, compile_metadata: CompileMetadata, compilable: Compilable
    ) -> Iterable[Code] | None:
        if not isinstance(compilable, AstCompilable):
            return None

        node = compilable.node
        out: list[Code] = []
        match node:
            case ast.ClassDef(
                name=name,
                bases=bases,
                keywords=keywords,
                decorator_list=decorators,
            ):
                body_bytecode = compile_metadata.compile_inner_function(node).bytecode
                out.append(opcode.NewList())

                for base in bases:
                    out.append(AstCompilable(base))
                    out.append(opcode.ListAppend())
                else:
                    out.append(opcode.ListToTuple())

                out.append(opcode.NewDict())
                for keyword in keywords:
                    out.append(opcode.LoadConstant(keyword.arg))
                    out.append(AstCompilable(keyword.value))
                    out.append(opcode.DictPut())

                out.append(
                    opcode.LoadAndBindInnerClass(
                        class_name=name, class_body=body_bytecode
                    )
                )

                for decorator in decorators:
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
                    out, ast.Name(id=name, ctx=ast.Store), compile_metadata
                )
                return out

            case _:
                return None
