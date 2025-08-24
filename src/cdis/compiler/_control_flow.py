from dataclasses import dataclass, field

from ._api import (
    AstPass,
    AstCompilable,
    CompileMetadata,
    Compilable,
    Code,
    use_synthetics,
    compile_context_manager,
    ExceptionHandlers,
    ExceptionHandler,
)
from ._exception import unuse_finally_block
from ._generator import await_tos
from ._variable import with_assignment_code
from .. import opcode as opcode

import ast
from typing import Iterable, ContextManager, MutableSequence


def with_match_opcodes(
    out: list[Code],
    match_case: ast.match_case,
    next_case_label: opcode.Label,
    match_end_label: opcode.Label,
    metadata: CompileMetadata,
) -> None:
    with_match_pattern_opcodes(out, match_case.pattern, True, next_case_label, metadata)
    if match_case.guard is not None:
        out.append(AstCompilable(match_case.guard))
        out.append(opcode.IfFalse(target=next_case_label))
    for statement in match_case.body:
        out.append(AstCompilable(statement))
    out.append(opcode.JumpTo(match_end_label))


def with_match_pattern_opcodes(
    out: list[Code],
    pattern: ast.pattern,
    should_pop: bool,
    next_case_label: opcode.Label,
    metadata: CompileMetadata,
) -> None:
    match pattern:
        case ast.MatchValue(value=value):
            out.extend(
                [
                    opcode.Dup(),
                    AstCompilable(value),
                    opcode.BinaryOp(operator=opcode.BinaryOperator.Eq),
                    opcode.IfFalse(target=next_case_label),
                ]
            )
            if should_pop:
                out.append(opcode.Pop())
            return

        case ast.MatchSingleton(value=value):
            out.extend(
                [
                    opcode.Dup(),
                    opcode.LoadConstant(constant=value),
                    opcode.IsSameAs(negate=False),
                    opcode.IfFalse(target=next_case_label),
                ]
            )
            if should_pop:
                out.append(opcode.Pop())
            return

        case ast.MatchSequence(patterns=subpatterns):
            has_star = any(
                isinstance(subpattern, ast.MatchStar) for subpattern in subpatterns
            )
            expected_length = len(subpatterns) - 1 if has_star else len(subpatterns)
            star_index = 0
            if has_star:
                for index, subpattern in enumerate(subpatterns):
                    if isinstance(subpattern, ast.MatchStar):
                        star_index = index
                        break
                before_count = star_index
                after_count = expected_length - before_count
            else:
                before_count = expected_length
                after_count = 0
            is_exact = not has_star
            out.extend(
                [
                    opcode.MatchSequence(
                        length=expected_length,
                        is_exact=is_exact,
                        target=next_case_label,
                    ),
                    opcode.Dup(),
                    opcode.UnpackElements(
                        before_count=before_count,
                        after_count=after_count,
                        has_extras=has_star,
                    ),
                ]
            )
            with use_synthetics(len(subpatterns), out, metadata) as element_values:
                for variable in element_values:
                    out.append(opcode.StoreSynthetic(index=variable))

                match_failed_label = opcode.Label()
                match_success_label = opcode.Label()
                for i in range(len(subpatterns)):
                    out.append(opcode.LoadSynthetic(index=element_values[i]))
                    with_match_pattern_opcodes(
                        out, subpatterns[i], True, match_failed_label, metadata
                    )

                if should_pop:
                    out.append(opcode.Pop())

                out.extend(
                    [
                        opcode.JumpTo(target=match_success_label),
                        match_failed_label,
                        opcode.Pop(),
                        opcode.JumpTo(target=next_case_label),
                        match_success_label,
                    ]
                )
                return

        case ast.MatchMapping(keys=keys, patterns=subpatterns, rest=extras_name):
            has_extras = extras_name is not None
            # keys must be literals, so we can evaluate them
            key_constants: list[ast.Constant] = keys  # noqa
            const_keys = tuple(key.value for key in key_constants)
            out.extend(
                [
                    opcode.MatchMapping(
                        keys=const_keys,
                        target=next_case_label,
                    ),
                    opcode.Dup(),
                    opcode.UnpackMapping(
                        keys=const_keys,
                        has_extras=has_extras,
                    ),
                ]
            )

            if has_extras:
                out.append(AstCompilable(ast.Name(id=extras_name, ctx=ast.Store())))

            match_failed_label = opcode.Label()
            match_success_label = opcode.Label()

            with use_synthetics(len(subpatterns), out, metadata) as key_values:
                for variable in key_values:
                    out.append(opcode.StoreSynthetic(index=variable))

            for index, subpattern in enumerate(subpatterns):
                out.append(opcode.LoadSynthetic(index=key_values[index]))
                with_match_pattern_opcodes(
                    out, subpattern, True, match_failed_label, metadata
                )

            if should_pop:
                out.append(opcode.Pop())
            out.extend(
                [
                    opcode.JumpTo(target=match_success_label),
                    match_failed_label,
                    opcode.Pop(),
                    opcode.JumpTo(target=next_case_label),
                    match_success_label,
                ]
            )
            return

        case ast.MatchClass(
            cls=matched_class,
            patterns=subpatterns,
            kwd_patterns=kwd_subpatterns,
            kwd_attrs=kwd_attrs,
        ):
            out.append(AstCompilable(matched_class))
            out.append(
                opcode.MatchClass(
                    target=next_case_label,
                    attributes=tuple(kwd_attrs),
                    positional_count=len(subpatterns),
                )
            )
            attribute_vars = {}
            positional_vars = {}

            # Store the values on the stack in reverse order
            with use_synthetics(
                len(subpatterns) + len(kwd_attrs), out, metadata
            ) as synthetics:
                synthetic_index = 0
                for kwd_attr in reversed(kwd_attrs):
                    attribute_vars[kwd_attr] = synthetics[synthetic_index]
                    out.append(opcode.StoreSynthetic(index=synthetics[synthetic_index]))
                    synthetic_index += 1
                for index, _ in enumerate(subpatterns):
                    positional_vars[len(subpatterns) - index - 1] = synthetics[
                        synthetic_index
                    ]
                    out.append(opcode.StoreSynthetic(index=synthetics[synthetic_index]))
                    synthetic_index += 1

                match_success_label = opcode.Label()
                match_failed_label = opcode.Label()

                for index, subpattern in enumerate(subpatterns):
                    out.append(opcode.LoadSynthetic(index=positional_vars[index]))
                    with_match_pattern_opcodes(
                        out, subpattern, True, match_failed_label, metadata
                    )

                for index, subpattern in enumerate(kwd_subpatterns):
                    out.append(
                        opcode.LoadSynthetic(index=attribute_vars[kwd_attrs[index]])
                    )
                    with_match_pattern_opcodes(
                        out, subpattern, True, match_failed_label, metadata
                    )

                out.extend(
                    [
                        opcode.JumpTo(target=match_success_label),
                        match_failed_label,
                        opcode.Pop(),
                        opcode.JumpTo(target=next_case_label),
                        match_success_label,
                    ]
                )
                return

        case ast.MatchStar(name=name):
            if name is not None:
                out.append(AstCompilable(ast.Name(id=name, ctx=ast.Store())))
            else:
                out.append(opcode.Pop())
            return

        case ast.MatchAs(pattern=subpattern, name=name):
            if subpattern is not None:
                with_match_pattern_opcodes(
                    out, subpattern, False, next_case_label, metadata
                )
            if name is not None:
                # TODO: Delay assignment until guard evaluation
                out.append(AstCompilable(ast.Name(id=name, ctx=ast.Store())))
            else:
                out.append(opcode.Pop())
            return

        case ast.MatchOr(patterns=subpatterns):
            next_pattern_label = opcode.Label()
            pattern_end_label = opcode.Label()
            for subpattern in subpatterns:
                with_match_pattern_opcodes(
                    out, subpattern, should_pop, next_pattern_label, metadata
                )
                out.append(opcode.JumpTo(pattern_end_label))
                out.append(next_pattern_label)
                next_pattern_label = opcode.Label()
            out.append(next_pattern_label)
            out.append(opcode.JumpTo(next_case_label))
            out.append(pattern_end_label)
            return

        case _:
            raise NotImplementedError(f"Not implemented pattern: {type(pattern)}")

    raise RuntimeError(
        f"Missing return in case {type(pattern)} in with_match_pattern_opcodes"
    )


@dataclass
class LoopState:
    break_labels: list[opcode.Label] = field(default_factory=list)
    continue_labels: list[opcode.Label] = field(default_factory=list)

    def use_labels(
        self,
        out: MutableSequence[Code],
        metadata: CompileMetadata,
        break_label: opcode.Label,
        continue_label: opcode.Label,
    ) -> ContextManager["LoopState"]:
        insert_index = len(self.break_labels)

        def before(_metadata: CompileMetadata):
            self.break_labels.append(break_label)
            self.continue_labels.append(continue_label)

        def after(_metadata: CompileMetadata):
            self.break_labels.pop(insert_index)
            self.continue_labels.pop(insert_index)

        return compile_context_manager(self, out, metadata, before, after)


class ControlFlowPass(AstPass):
    def accept(
        self, compile_metadata: CompileMetadata, compilable: Compilable
    ) -> Iterable[Code] | None:
        if not isinstance(compilable, AstCompilable):
            return None

        node = compilable.node
        out: list[Code] = []
        match node:
            case ast.If(test, body, orelse):
                false_label = opcode.Label()
                done_label = opcode.Label()
                out.append(AstCompilable(test))
                out.append(opcode.IfFalse(false_label))

                for body_statement in body:
                    out.append(AstCompilable(body_statement))

                out.append(opcode.JumpTo(done_label))
                out.append(false_label)

                for else_statement in orelse:
                    out.append(AstCompilable(else_statement))

                out.append(done_label)
                return out

            case ast.IfExp(test, body, orelse):
                end_label = opcode.Label()
                orelse_label = opcode.Label()
                out.append(AstCompilable(test))
                out.append(opcode.IfFalse(target=orelse_label))
                out.append(AstCompilable(body))
                out.append(opcode.JumpTo(target=end_label))
                out.append(orelse_label)
                out.append(AstCompilable(orelse))
                out.append(end_label)
                return out

            case ast.Match(subject=subject, cases=cases):
                out.append(AstCompilable(subject))
                next_case_label = opcode.Label()
                match_end_label = opcode.Label()
                for case in cases:
                    with_match_opcodes(
                        out, case, next_case_label, match_end_label, compile_metadata
                    )
                    out.append(next_case_label)
                    next_case_label = opcode.Label()
                out.append(opcode.Pop())
                out.append(match_end_label)
                return out

            case ast.Return(value=value):
                exception_handlers = compile_metadata.get_shared(ExceptionHandlers)
                if value is not None:
                    out.append(AstCompilable(value))
                else:
                    out.append(opcode.LoadConstant(None))

                match compile_metadata.function_type:
                    case (
                        opcode.FunctionType.GENERATOR
                        | opcode.FunctionType.ASYNC_FUNCTION
                        | opcode.FunctionType.COROUTINE_GENERATOR
                        | opcode.FunctionType.ASYNC_GENERATOR
                    ):
                        is_generator_or_async_function = True
                    case opcode.FunctionType.FUNCTION | opcode.FunctionType.CLASS_BODY:
                        is_generator_or_async_function = False
                    case _:
                        raise RuntimeError(
                            f"Unknown function type {compile_metadata.function_type}"
                        )

                # TODO
                if len(exception_handlers.finally_blocks) == 0:
                    if is_generator_or_async_function:
                        out.extend(
                            [
                                opcode.LoadConstant(True),
                                opcode.LoadSynthetic(index=0),
                                opcode.StoreAttr(name="_generator_finished"),
                            ]
                        )
                    out.append(opcode.ReturnValue())
                    return out

                with use_synthetics(1, out, compile_metadata) as [return_value_holder]:
                    out.append(opcode.StoreSynthetic(return_value_holder))
                    finally_blocks = exception_handlers.finally_blocks

                    for i in range(len(finally_blocks) - 1, -1, -1):
                        with unuse_finally_block(out, i, compile_metadata) as block:
                            out.extend(block)

                    out.append(opcode.LoadSynthetic(return_value_holder))

                    if is_generator_or_async_function:
                        out.extend(
                            [
                                opcode.LoadConstant(True),
                                opcode.LoadSynthetic(index=0),
                                opcode.StoreAttr(name="_generator_finished"),
                            ]
                        )
                    out.append(opcode.ReturnValue())
                    return out

            case ast.Break():
                loop_state = compile_metadata.get(self, lambda: LoopState())
                return [opcode.JumpTo(loop_state.break_labels[-1])]

            case ast.Continue():
                loop_state = compile_metadata.get(self, lambda: LoopState())
                return [opcode.JumpTo(loop_state.continue_labels[-1])]

            case ast.While(test, body, orelse):
                test_label = opcode.Label()
                false_label = opcode.Label()
                break_label = opcode.Label()
                out.append(test_label)
                out.append(AstCompilable(test))
                out.append(opcode.IfFalse(false_label))

                loop_state = compile_metadata.get(self, lambda: LoopState())
                with loop_state.use_labels(
                    out, compile_metadata, break_label, test_label
                ):
                    for body_statement in body:
                        out.append(AstCompilable(body_statement))

                out.append(opcode.JumpTo(test_label))
                out.append(false_label)

                for else_statement in orelse:
                    out.append(AstCompilable(else_statement))

                out.append(break_label)
                return out

            case ast.For(target=target, iter=iter, body=body, orelse=orelse):
                with use_synthetics(1, out, compile_metadata) as [iterator_synthetic]:
                    next_label = opcode.Label()
                    got_next_label = opcode.Label()
                    iterator_exhausted_label = opcode.Label()
                    break_label = opcode.Label()
                    out.extend(
                        [
                            AstCompilable(iter),
                            opcode.GetIterator(),
                            opcode.StoreSynthetic(iterator_synthetic),
                            next_label,
                            opcode.LoadSynthetic(iterator_synthetic),
                            opcode.GetNextElseJumpTo(target=iterator_exhausted_label),
                        ]
                    )
                    with_assignment_code(out, target, compile_metadata)
                    out.append(got_next_label)

                    loop_state = compile_metadata.get(self, lambda: LoopState())
                    with loop_state.use_labels(
                        out, compile_metadata, break_label, next_label
                    ):
                        for body_statement in body:
                            out.append(AstCompilable(body_statement))

                    out.append(opcode.JumpTo(next_label))
                    out.append(iterator_exhausted_label)

                    for else_statement in orelse:
                        out.append(AstCompilable(else_statement))

                    out.append(break_label)
                    return out

            case ast.AsyncFor(target=target, iter=iter, body=body, orelse=orelse):
                exception_handlers = compile_metadata.get_shared(ExceptionHandlers)
                with use_synthetics(1, out, compile_metadata) as [iterator_synthetic]:
                    next_label = opcode.Label()
                    got_next_label = opcode.Label()
                    iterator_exhausted_label = opcode.Label()
                    break_label = opcode.Label()
                    out.extend(
                        [
                            AstCompilable(iter),
                            opcode.GetAsyncIterator(),
                            opcode.StoreSynthetic(iterator_synthetic),
                            next_label,
                            opcode.LoadSynthetic(iterator_synthetic),
                            opcode.GetAsyncNext(),
                        ]
                    )

                    await_start_label = opcode.Label()
                    out.append(await_start_label)
                    await_tos(out, compile_metadata)
                    await_end_label = opcode.Label()

                    out.append(await_end_label)
                    with_assignment_code(out, target, compile_metadata)
                    out.append(got_next_label)

                    loop_state = compile_metadata.get(self, lambda: LoopState())
                    with loop_state.use_labels(
                        out, compile_metadata, break_label, next_label
                    ):
                        for body_statement in body:
                            out.append(AstCompilable(body_statement))

                    out.append(opcode.JumpTo(next_label))
                    out.append(iterator_exhausted_label)

                    for else_statement in orelse:
                        out.append(AstCompilable(else_statement))

                    iteration_done_label = opcode.Label()
                    out.append(iteration_done_label)
                    exception_handlers.add_exception_handler(
                        ExceptionHandler(
                            exception_class=StopAsyncIteration,
                            from_label=await_start_label,
                            to_label=await_end_label,
                            handler_label=iteration_done_label,
                        )
                    )
                    out.append(opcode.Pop())
                    out.append(break_label)
                    return out

            case _:
                return None
