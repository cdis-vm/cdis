from ._api import (
    AstPass,
    AstCompilable,
    CompileMetadata,
    Compilable,
    Code,
    use_synthetics,
    ExceptionHandlers,
    ExceptionHandler,
    compile_context_manager,
)
from ._variable import with_assignment_code
from ._generator import await_tos
from .. import opcode as opcode

import ast
from typing import Iterable, MutableSequence, Sequence, ContextManager


def use_finally_block(
    out: MutableSequence[Code],
    finally_block: Sequence[Code],
    compile_metadata: CompileMetadata,
) -> ContextManager[Sequence[Code]]:
    def push_block(metadata: CompileMetadata):
        exception_handlers = metadata.get_shared(ExceptionHandlers)
        exception_handlers.push_finally_block(finally_block)

    def pop_block(metadata: CompileMetadata):
        exception_handlers = metadata.get_shared(ExceptionHandlers)
        exception_handlers.push_finally_block(finally_block)

    return compile_context_manager(
        finally_block, out, compile_metadata, push_block, pop_block
    )


def unuse_finally_block(
    out: MutableSequence[Code], first_unused: int, compile_metadata: CompileMetadata
) -> ContextManager[Sequence[Code]]:
    original: list[Sequence[Code]] = []

    def push_block(metadata: CompileMetadata):
        nonlocal original
        exception_handlers = metadata.get_shared(ExceptionHandlers)
        original = exception_handlers.finally_blocks
        exception_handlers.finally_blocks = exception_handlers.finally_blocks[
            :first_unused
        ]

    def pop_block(metadata: CompileMetadata):
        nonlocal original
        exception_handlers = metadata.get_shared(ExceptionHandlers)
        exception_handlers.finally_blocks = original

    return compile_context_manager(
        compile_metadata.get_shared(ExceptionHandlers).finally_blocks[first_unused],
        out,
        compile_metadata,
        push_block,
        pop_block,
    )


def _with(
    out: MutableSequence[Code],
    with_items,
    body,
    metadata: CompileMetadata,
    *,
    is_async: bool,
):
    enter_method = "__aenter__" if is_async else "__enter__"
    exit_method = "__aexit__" if is_async else "__exit__"
    exception_handlers = metadata.get_shared(ExceptionHandlers)

    def call_exit(exception_index, exit_manager, exit_index):
        return (
            opcode.LoadSynthetic(index=exit_index),
            opcode.CreateCallBuilder(),
            opcode.LoadSynthetic(index=exit_manager),
            opcode.WithPositionalArg(index=0),
            opcode.LoadSynthetic(index=exception_index),
            opcode.GetType(),
            opcode.WithPositionalArg(index=1),
            opcode.LoadSynthetic(index=exception_index),
            opcode.WithPositionalArg(index=2),
            opcode.LoadSynthetic(index=exception_index),
            opcode.LoadAttr(name="__traceback__"),
            opcode.WithPositionalArg(index=3),
            opcode.CallWithBuilder(),
        )

    def call_normal_exit(exit_manager, exit_index):
        return (
            opcode.LoadSynthetic(index=exit_index),
            opcode.CreateCallBuilder(),
            opcode.LoadSynthetic(index=exit_manager),
            opcode.WithPositionalArg(index=0),
            opcode.LoadConstant(None),
            opcode.WithPositionalArg(index=1),
            opcode.LoadConstant(None),
            opcode.WithPositionalArg(index=2),
            opcode.LoadConstant(None),
            opcode.WithPositionalArg(index=3),
            opcode.CallWithBuilder(),
            *([] if is_async else [opcode.Pop()]),
        )

    with use_synthetics(2 * len(with_items), out, metadata) as synthetics:
        exit_callables = synthetics[0::2]
        exit_managers = synthetics[1::2]
        manager_index = 0
        for with_item in with_items:
            item_start_label = opcode.Label()
            item_done_label = opcode.Label()
            next_label = opcode.Label()
            handler_start = opcode.Label()

            out.extend(
                [
                    item_start_label,
                    AstCompilable(with_item.context_expr),
                    opcode.Dup(),
                    opcode.Dup(),
                    opcode.StoreSynthetic(index=exit_managers[manager_index]),
                    opcode.LoadObjectTypeAttr(name=enter_method),
                    opcode.Swap(),
                    opcode.LoadObjectTypeAttr(name=exit_method),
                    opcode.StoreSynthetic(index=exit_callables[manager_index]),
                    opcode.CreateCallBuilder(),
                    opcode.LoadSynthetic(index=exit_managers[manager_index]),
                    opcode.WithPositionalArg(index=0),
                    opcode.CallWithBuilder(),
                ]
            )

            if is_async:
                await_tos(out, metadata)

            if with_item.optional_vars is not None:
                with_assignment_code(out, with_item.optional_vars, metadata)
            else:
                out.append(opcode.Pop())

            out.append(item_done_label)
            # If an exception is raised when calling __enter__,
            # do not shallow the exception, even if __exit__ return True
            if manager_index != 0:
                # Call exits of all context managers initialized up to this point
                out.append(opcode.JumpTo(next_label))
                exception_handlers.add_exception_handler(
                    ExceptionHandler(
                        exception_class=BaseException,
                        from_label=item_start_label,
                        to_label=item_done_label,
                        handler_label=handler_start,
                    )
                )
                out.append(handler_start)
                with use_synthetics(1, out, metadata) as [exception]:
                    out.append(opcode.StoreSynthetic(index=exception))
                    for _index in range(manager_index):
                        exit_manager = exit_managers[_index]
                        exit_callable = exit_callables[_index]
                        out.extend(call_exit(exception, exit_manager, exit_callable))
                        if is_async:
                            await_tos(out, metadata)
                        out.append(opcode.Pop())
                out.append(opcode.ReraiseLast())
                out.append(next_label)

            manager_index += 1

        with_start = opcode.Label()
        with_end = opcode.Label()
        handler_start = opcode.Label()
        handler_end = opcode.Label()

        out.append(with_start)
        for inner_stmt in body:
            out.append(AstCompilable(inner_stmt))

        out.extend([opcode.JumpTo(handler_end), with_end, handler_start])
        exception_handlers.add_exception_handler(
            ExceptionHandler(
                exception_class=BaseException,
                from_label=with_start,
                to_label=with_end,
                handler_label=handler_start,
            )
        )

        with use_synthetics(1, out, metadata) as [exception]:
            out.append(opcode.StoreSynthetic(index=exception))
            out.append(opcode.LoadConstant(constant=False))
            for _index in range(len(exit_managers)):
                exit_manager = exit_managers[_index]
                exit_callable = exit_callables[_index]
                out.extend(call_exit(exception, exit_manager, exit_callable))
                if is_async:
                    await_tos(out, metadata)
                # if there are multiple context managers,
                # the exception is shallow if ANY of their exits
                # return a truthy value
                out.append(opcode.AsBool())
                out.append(opcode.BinaryOp(operator=opcode.BinaryOperator.BitOr))
            out.append(opcode.IfTrue(target=handler_end))
            out.append(opcode.ReraiseLast())

        out.append(handler_end)
        for _index in range(len(exit_managers)):
            exit_manager = exit_managers[_index]
            exit_callable = exit_callables[_index]
            out.extend(call_normal_exit(exit_manager, exit_callable))
            if is_async:
                await_tos(out, metadata)
                out.append(opcode.Pop())
    return out


class ExceptionPass(AstPass):
    def accept(
        self, compile_metadata: CompileMetadata, compilable: Compilable
    ) -> Iterable[Code] | None:
        if not isinstance(compilable, AstCompilable):
            return None

        node = compilable.node
        out: list[Code] = []
        exception_handlers = compile_metadata.get_shared(ExceptionHandlers)

        match node:
            case ast.Raise(exc=exc, cause=cause):
                if exc is None:
                    return [opcode.ReraiseLast()]
                out.append(AstCompilable(exc))
                if cause is None:
                    out.append(opcode.Raise())
                    return out
                out.append(AstCompilable(cause))
                out.append(opcode.RaiseWithCause())
                return out

            case ast.Try(
                body=body, handlers=handlers, orelse=orelse, finalbody=finalbody
            ):
                try_start = opcode.Label()
                try_end = opcode.Label()
                except_start = opcode.Label()
                except_finally = opcode.Label()
                try_finally = opcode.Label()

                out.append(try_start)
                final_block = [AstCompilable(stmt) for stmt in finalbody]
                with use_finally_block(out, final_block, compile_metadata):
                    for statement in body:
                        out.append(AstCompilable(statement))
                    out.append(try_end)

                    for statement in orelse:
                        out.append(AstCompilable(statement))

                    out.append(opcode.JumpTo(try_finally))

                    exception_handlers.add_exception_handler(
                        ExceptionHandler(
                            exception_class=BaseException,
                            from_label=try_start,
                            to_label=try_end,
                            handler_label=except_start,
                        )
                    )

                    out.append(except_start)
                    with use_synthetics(1, out, compile_metadata) as [caught_exception]:
                        out.append(opcode.StoreSynthetic(index=caught_exception))
                        next_handler = opcode.Label()
                        for except_handler in handlers:
                            if except_handler.type is not None:
                                out.append(opcode.LoadSynthetic(index=caught_exception))
                                out.append(AstCompilable(except_handler.type))
                                out.append(
                                    opcode.JumpIfNotMatchExceptType(target=next_handler)
                                )
                            if except_handler.name is not None:
                                out.append(opcode.LoadSynthetic(index=caught_exception))
                                out.append(opcode.StoreLocal(except_handler.name))

                            for statement in except_handler.body:
                                out.append(AstCompilable(statement))

                            out.append(opcode.JumpTo(try_finally))
                            out.append(next_handler)
                            next_handler = opcode.Label()

                        out.append(opcode.LoadSynthetic(index=caught_exception))
                        out.append(except_finally)

                out.append(opcode.Pop())

                exception_handlers.add_exception_handler(
                    ExceptionHandler(
                        exception_class=BaseException,
                        from_label=try_start,
                        to_label=except_finally,
                        handler_label=except_finally,
                    )
                )

                out.extend(final_block)
                out.append(opcode.ReraiseLast())
                out.append(try_finally)
                out.extend(final_block)

                return out

            case ast.AsyncWith(items=with_items, body=body):
                _with(out, with_items, body, compile_metadata, is_async=True)
                return out

            case ast.With(items=with_items, body=body):
                _with(out, with_items, body, compile_metadata, is_async=False)
                return out
