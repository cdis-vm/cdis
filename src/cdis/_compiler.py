from ._bytecode import Instruction, Label
from cdis import bytecode as opcode
from dataclasses import dataclass, replace, field
from types import FunctionType, CellType
from typing import cast, Callable
import inspect
import ast


def _impossible_state(msg):
    raise RuntimeError(f"Impossible state: {msg}")


@dataclass(frozen=True)
class ExceptionHandler:
    exception_class: type
    from_label: Label
    to_label: Label
    handler_label: Label


@dataclass(frozen=True)
class AssignmentCode:
    stack_items: int
    load_target: Callable[["Bytecode"], "Bytecode"]
    load_current_value: Callable[["Bytecode"], "Bytecode"]
    store_value: Callable[["Bytecode"], "Bytecode"]
    delete_value: Callable[["Bytecode"], "Bytecode"]


@dataclass(frozen=True)
class Bytecode:
    function_name: str
    signature: inspect.Signature
    instructions: tuple[Instruction, ...] = ()
    closure: dict[str, CellType] = field(default_factory=dict)
    globals: dict[str, object] = field(default_factory=dict)
    local_names: frozenset[str] = frozenset()
    cell_names: frozenset[str] = frozenset()
    free_names: frozenset[str] = frozenset()
    global_names: frozenset[str] = frozenset()
    current_synthetic: int = -1
    synthetic_count: int = 0
    labels: tuple[Label, ...] = ()
    continue_labels: tuple[Label, ...] = ()
    break_labels: tuple[Label, ...] = ()
    exception_handlers: tuple[ExceptionHandler, ...] = ()
    finally_blocks: tuple[tuple[ast.stmt, ...], ...] = ()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        out = ""
        out += f"function_name: {self.function_name}\n"
        out += f"locals: {self.local_names}\n"
        out += f"cells: {self.cell_names}\n"
        out += f"free: {self.free_names}\n"
        out += f"globals: {self.global_names}\n"
        out += f"exception handlers: {self.exception_handlers}\n"
        out += f"instructions:\n"
        for instruction in self.instructions:
            out += f"{instruction.bytecode_index:<16}{str(instruction.opcode)}\n"
        return out

    def _evaluate_constant_expr(self, expression: ast.expr):
        from cdis._vm import CDisVM

        new_bytecode = replace(
            self, instructions=(), finally_blocks=(), signature=inspect.Signature()
        )
        new_bytecode = new_bytecode.with_statement_opcodes(
            ast.Return(value=expression, lineno=0, col_offset=0)
        )
        vm = CDisVM()
        return vm.run(new_bytecode)

    def add_local(self, name: str) -> "Bytecode":
        return replace(self, local_names=self.local_names | {name})

    def add_global(self, name: str) -> "Bytecode":
        return replace(self, global_names=self.global_names | {name})

    def add_cell(self, name: str) -> "Bytecode":
        return replace(self, cell_names=self.cell_names | {name})

    def new_synthetic(self) -> "Bytecode":
        return replace(
            self,
            current_synthetic=self.current_synthetic + 1,
            synthetic_count=max(self.synthetic_count, self.current_synthetic + 2),
        )

    def free_synthetic(self) -> "Bytecode":
        return replace(self, current_synthetic=self.current_synthetic - 1)

    def add_label(self, label: Label) -> "Bytecode":
        label._bytecode_index = len(self.instructions)
        return replace(self, labels=self.labels + (label,))

    def push_continue_label(self, label: Label) -> "Bytecode":
        return replace(self, continue_labels=self.continue_labels + (label,))

    def pop_continue_label(self) -> "Bytecode":
        return replace(self, continue_labels=self.continue_labels[:-1])

    @property
    def continue_label(self) -> "Label":
        return self.continue_labels[-1]

    def push_break_label(self, label: Label) -> "Bytecode":
        return replace(self, break_labels=self.break_labels + (label,))

    def pop_break_label(self) -> "Bytecode":
        return replace(self, break_labels=self.break_labels[:-1])

    @property
    def break_label(self) -> "Label":
        return self.break_labels[-1]

    def add_exception_handler(self, handler: ExceptionHandler) -> "Bytecode":
        return replace(self, exception_handlers=self.exception_handlers + (handler,))

    def push_finally_block(self, block: tuple[ast.stmt, ...]):
        return replace(self, finally_blocks=self.finally_blocks + (block,))

    def pop_finally_block(self):
        return replace(self, finally_blocks=self.finally_blocks[:-1])

    def add_op(self, op: opcode.Opcode) -> "Bytecode":
        lineno = self.instructions[-1].lineno if len(self.instructions) > 0 else 0
        return replace(
            self,
            instructions=self.instructions
            + (
                Instruction(
                    opcode=op,
                    bytecode_index=len(self.instructions),
                    lineno=lineno,
                ),
            ),
        )

    def add_ops(self, *ops: opcode.Opcode | Label) -> "Bytecode":
        out = self
        for op in ops:
            if isinstance(op, Label):
                out = out.add_label(op)
            else:
                out = out.add_op(op)
        return out

    def add_statement(self, statement: ast.stmt, op: opcode.Opcode) -> "Bytecode":
        return replace(
            self,
            instructions=self.instructions
            + (
                Instruction(
                    opcode=op,
                    bytecode_index=len(self.instructions),
                    lineno=statement.lineno
                    if hasattr(statement, "lineno")
                    else self.instructions[-1].lineno,
                ),
            ),
        )

    def add_expression(self, expression: ast.expr, op: opcode.Opcode) -> "Bytecode":
        return replace(
            self,
            instructions=self.instructions
            + (
                Instruction(
                    opcode=op,
                    bytecode_index=len(self.instructions),
                    lineno=expression.lineno
                    if hasattr(expression, "lineno")
                    else self.instructions[-1].lineno,
                ),
            ),
        )

    def with_negated_tos(self) -> "Bytecode":
        if_true_label = Label()
        done_label = Label()
        return self.add_ops(
            opcode.IfTrue(if_true_label),
            opcode.LoadConstant(True),
            opcode.JumpTo(done_label),
            if_true_label,
            opcode.LoadConstant(False),
            done_label,
        )

    def with_comparison_op(self, cmpop: ast.cmpop) -> "Bytecode":
        match cmpop:
            case ast.Is():
                return self.add_op(opcode.IsSameAs(negate=False))
            case ast.IsNot():
                return self.add_op(opcode.IsSameAs(negate=True))
            case ast.In():
                return self.add_op(opcode.IsContainedIn(negate=False))
            case ast.NotIn():
                return self.add_op(opcode.IsContainedIn(negate=True))
            case _:
                operator = opcode.BinaryOperator[type(cmpop).__name__]
                return self.add_op(opcode.BinaryOp(operator))

    def with_generator_opcodes(
        self,
        generator: ast.comprehension,
        generator_start: Label,
        generator_end: Label,
        accept_elt: Callable[["Bytecode"], "Bytecode"],
    ) -> "Bytecode":
        # TODO
        raise NotImplementedError

    def with_expression_opcodes(self, expression: ast.expr) -> "Bytecode":
        match expression:
            case ast.Constant(value):
                return self.add_expression(expression, opcode.LoadConstant(value))

            case ast.Attribute(attr) as a:
                out = self.with_expression_opcodes(a.value)
                out = out.add_expression(expression, opcode.LoadAttr(attr.id))
                return out

            case ast.Name(id=name, ctx=ast.Load()):
                if name in self.cell_names:
                    return self.add_expression(
                        expression, opcode.LoadCell(name, name in self.free_names)
                    )
                elif name in self.local_names:
                    return self.add_expression(expression, opcode.LoadLocal(name))
                else:
                    return self.add_expression(expression, opcode.LoadGlobal(name))

            case ast.UnaryOp(op, operand):
                out = self.with_expression_opcodes(operand)
                match operand:
                    case ast.Not():
                        return out.with_negated_tos()
                    case _:
                        return out.add_expression(
                            expression,
                            opcode.UnaryOp(opcode.UnaryOperator[type(op).__name__]),
                        )

            case ast.BinOp(left, op, right):
                out = self.with_expression_opcodes(left)
                out = out.with_expression_opcodes(right)
                return out.add_expression(
                    expression,
                    opcode.BinaryOp(opcode.BinaryOperator[type(op).__name__]),
                )

            case ast.Compare(left, ops, comparators):
                out = self.with_expression_opcodes(left)
                false_label = Label()
                for i in range(len(ops) - 1):
                    out = out.with_expression_opcodes(comparators[i])
                    out = out.add_op(opcode.DupX1())
                    out = out.with_comparison_op(ops[i])

                    out = out.add_ops(
                        opcode.Dup(), opcode.IfFalse(false_label), opcode.Pop()
                    )

                out = out.with_expression_opcodes(comparators[-1])
                out = out.with_comparison_op(ops[-1])
                done_label = Label()
                if len(ops) > 1:
                    out = out.add_op(opcode.JumpTo(done_label))
                    out = out.add_label(false_label)
                    out = out.add_ops(opcode.Swap(), opcode.Pop())
                    out = out.add_label(done_label)

                return out

            case ast.BoolOp(op, values):
                match op:
                    case ast.And():
                        jump_opcode = opcode.IfFalse
                    case ast.Or():
                        jump_opcode = opcode.IfTrue
                    case _:
                        raise ValueError(f"Unhandled BoolOp {op}")

                out = self
                end_label = Label()
                for value in values[:-1]:
                    out = out.with_expression_opcodes(value)
                    out = out.add_op(opcode.Dup())
                    out = out.add_op(jump_opcode(end_label))
                    out = out.add_op(opcode.Pop())
                out = out.with_expression_opcodes(values[-1])
                return out.add_label(end_label)

            case ast.Call(func, args, keywords):
                out = self.with_expression_opcodes(func)
                out = out.add_op(opcode.CreateCallBuilder())
                extended = False
                arg_index = 0
                for arg in args:
                    match arg:
                        case ast.Starred(value):
                            out = out.with_expression_opcodes(value)
                            out = out.add_op(opcode.ExtendPositionalArgs())
                            extended = True
                        case _ as expression:
                            out = out.with_expression_opcodes(expression)
                            if extended:
                                out = out.add_op(opcode.AppendPositionalArg())
                            else:
                                out = out.add_op(opcode.WithPositionalArg(arg_index))
                                arg_index += 1

                for kwarg in keywords:
                    out = out.with_expression_opcodes(kwarg.value)
                    if kwarg.arg is None:
                        out = out.add_op(opcode.ExtendKeywordArgs())
                    else:
                        out = out.add_op(opcode.WithKeywordArg(kwarg.arg))

                return out.add_op(opcode.CallWithBuilder())

            case ast.List(elts):
                out = self.add_op(opcode.NewList())
                for elt in elts:
                    match elt:
                        case ast.Starred(value):
                            out = out.with_expression_opcodes(value)
                            out = out.add_op(opcode.ListExtend())
                        case _:
                            out = out.with_expression_opcodes(elt)
                            out = out.add_op(opcode.ListAppend())
                return out

            case ast.Tuple(elts):
                out = self.add_op(opcode.NewList())
                for elt in elts:
                    match elt:
                        case ast.Starred(value):
                            out = out.with_expression_opcodes(value)
                            out = out.add_op(opcode.ListExtend())
                        case _:
                            out = out.with_expression_opcodes(elt)
                            out = out.add_op(opcode.ListAppend())
                out = out.add_op(opcode.ListToTuple())
                return out

            case ast.Set(elts):
                out = self.add_op(opcode.NewSet())
                for elt in elts:
                    match elt:
                        case ast.Starred(value):
                            out = out.with_expression_opcodes(value)
                            out = out.add_op(opcode.SetUpdate())
                        case _:
                            out = out.with_expression_opcodes(elt)
                            out = out.add_op(opcode.SetAdd())
                return out

            case ast.Dict(keys, values):
                out = self.add_op(opcode.NewDict())
                for index, value in enumerate(values):
                    key = keys[index]
                    if key is None:
                        out = out.with_expression_opcodes(value)
                        out = out.add_op(opcode.DictUpdate())
                    else:
                        out = out.with_expression_opcodes(key)
                        out = out.with_expression_opcodes(value)
                        out = out.add_op(opcode.DictPut())
                return out

            case ast.Subscript(value, slice=slice_, ctx=ctx):
                out = self.with_expression_opcodes(value)
                out = out.with_expression_opcodes(slice_)
                match ctx:
                    case ast.Load():
                        out = out.add_op(opcode.GetItem())
                    case ast.Store():
                        out = out.add_op(opcode.SetItem())
                    case ast.Delete():
                        out = out.add_op(opcode.DeleteItem())
                    case _:
                        raise NotImplementedError(f"Unhandled Subscript ctx: {ctx}")
                return out

            case ast.FormattedValue(value, conversion, format_spec):
                out = self.with_expression_opcodes(value)
                out = out.add_op(
                    opcode.FormatValue(
                        conversion=opcode.FormatConversion.from_int(conversion),
                        # format_spec should always be a constant string or None
                        format_spec=format_spec.values[0].value
                        if format_spec is not None
                        else "",
                    )
                )
                return out

            case ast.JoinedStr(values):
                out = self
                for value in values:
                    out = out.with_expression_opcodes(value)
                out = out.add_op(opcode.JoinStringValues(count=len(values)))
                return out

            case _:
                raise NotImplementedError(
                    f"Not implemented statement: {type(expression)}"
                )

        raise RuntimeError(
            f"Missing return in case {type(expression)} in with_expression_opcodes"
        )

    def get_assignment_code(self, expression: ast.expr) -> AssignmentCode:
        match expression:
            case ast.Name(id=name):
                if name in self.cell_names:
                    return AssignmentCode(
                        stack_items=0,
                        load_target=lambda b: b,
                        load_current_value=lambda b: b.add_expression(
                            expression, opcode.LoadCell(name, name in self.free_names)
                        ),
                        store_value=lambda b: b.add_expression(
                            expression, opcode.StoreCell(name, name in self.free_names)
                        ),
                        delete_value=lambda b: b.add_expression(
                            expression, opcode.DeleteCell(name, name in self.free_names)
                        ),
                    )
                elif name in self.local_names:
                    return AssignmentCode(
                        stack_items=0,
                        load_target=lambda b: b,
                        load_current_value=lambda b: b.add_expression(
                            expression, opcode.LoadLocal(name)
                        ),
                        store_value=lambda b: b.add_expression(
                            expression, opcode.StoreLocal(name)
                        ),
                        delete_value=lambda b: b.add_expression(
                            expression, opcode.DeleteLocal(name)
                        ),
                    )
                else:
                    return AssignmentCode(
                        stack_items=0,
                        load_target=lambda b: b,
                        load_current_value=lambda b: b.add_expression(
                            expression, opcode.LoadGlobal(name)
                        ),
                        store_value=lambda b: b.add_expression(
                            expression, opcode.StoreGlobal(name)
                        ),
                        delete_value=lambda b: b.add_expression(
                            expression, opcode.DeleteGlobal(name)
                        ),
                    )

            case ast.Attribute(attr) as a:
                return AssignmentCode(
                    stack_items=1,
                    load_target=lambda b: b.with_expression_opcodes(a.value),
                    load_current_value=lambda b: b.add_expression(
                        expression, opcode.LoadAttr(attr.id)
                    ),
                    store_value=lambda b: b.add_expression(
                        expression, opcode.StoreAttr(attr.id)
                    ),
                    delete_value=lambda b: b.add_expression(
                        expression, opcode.DeleteAttr(attr.id)
                    ),
                )

            case ast.Subscript(value, slice=slice_, ctx=ctx):
                return AssignmentCode(
                    stack_items=2,
                    load_target=lambda b: b.with_expression_opcodes(
                        value
                    ).with_expression_opcodes(slice_),
                    load_current_value=lambda b: b.add_op(opcode.GetItem()),
                    store_value=lambda b: b.add_op(opcode.SetItem()),
                    delete_value=lambda b: b.add_op(opcode.DeleteItem()),
                )

            case ast.Starred(value=inner):
                return self.get_assignment_code(inner)

            case ast.List(elts, ctx):
                return self.get_assignment_code(ast.Tuple(elts=elts, ctx=ctx))

            case ast.Tuple(elts):

                def store_elts(bytecode: "Bytecode") -> "Bytecode":
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

                    bytecode = bytecode.add_op(
                        opcode.UnpackElements(
                            before_count=before_count,
                            has_extras=star_index is not None,
                            after_count=after_count,
                        )
                    )
                    for elt in elts:
                        assignment_code = bytecode.get_assignment_code(elt)
                        bytecode = assignment_code.load_target(bytecode)
                        bytecode = assignment_code.store_value(bytecode)

                    return bytecode

                def delete_elts(bytecode: "Bytecode") -> "Bytecode":
                    for elt in elts:
                        assignment_code = bytecode.get_assignment_code(elt)
                        bytecode = assignment_code.load_target(bytecode)
                        bytecode = assignment_code.delete_value(bytecode)
                    return bytecode

                return AssignmentCode(
                    stack_items=0,
                    load_target=lambda b: b,
                    load_current_value=lambda b: _impossible_state(
                        "Attempted augmented assignment to a tuple"
                    ),
                    store_value=store_elts,
                    delete_value=delete_elts,
                )

            case _:
                raise NotImplementedError(
                    f"Not implemented assignment: {type(expression)}"
                )

        raise RuntimeError(
            f"Missing return in case {type(expression)} in get_assignment_code"
        )

    def with_assignment_opcodes(self, expression: ast.expr) -> "Bytecode":
        assignment_code = self.get_assignment_code(expression)
        return assignment_code.store_value(assignment_code.load_target(self))

    def with_statement_opcodes(self, statement: ast.stmt) -> "Bytecode":
        match statement:
            case ast.Pass():
                return self.add_statement(statement, opcode.Nop())

            case ast.Global(names):
                out = self
                for name in names:
                    out = out.add_global(name)
                return out

            case ast.Nonlocal(names):
                out = self
                for name in names:
                    out = out.add_cell(name)
                return out

            case ast.Expr(value):
                out = self.with_expression_opcodes(value)
                out = out.add_op(opcode.Pop())
                return out

            case ast.Return(value):
                if value is not None:
                    out = self.with_expression_opcodes(value)
                else:
                    out = self.with_expression_opcodes(
                        ast.Constant(None, lineno=statement.lineno)
                    )

                if len(out.finally_blocks) == 0:
                    return out.add_statement(statement, opcode.ReturnValue())

                out = out.new_synthetic()
                return_value_holder = out.current_synthetic

                out = out.add_op(opcode.StoreSynthetic(return_value_holder))

                finally_blocks = out.finally_blocks
                for i in range(len(finally_blocks) - 1, -1, -1):
                    block = finally_blocks[i]
                    out = replace(out, finally_blocks=finally_blocks[:i])
                    for statement in block:
                        out = out.with_statement_opcodes(statement)

                out = replace(out, finally_blocks=finally_blocks)
                out = out.add_op(opcode.LoadSynthetic(return_value_holder))
                out = out.free_synthetic()
                return out.add_statement(statement, opcode.ReturnValue())

            case ast.Raise(exc, cause):
                if exc is None:
                    return self.add_op(opcode.ReraiseLast())
                out = self.with_expression_opcodes(exc)
                if cause is None:
                    return out.add_op(opcode.Raise())
                out = out.with_expression_opcodes(cause)
                return out.add_op(opcode.RaiseWithCause())

            case ast.Try(body, handlers, orelse, finalbody):
                try_start = Label()
                try_end = Label()
                except_finally = Label()
                try_finally = Label()

                out = self.add_label(try_start)
                out = out.push_finally_block(finalbody)
                for statement in body:
                    out = out.with_statement_opcodes(statement)
                out = out.add_label(try_end)

                for statement in orelse:
                    out = out.with_statement_opcodes(statement)

                out = out.add_op(opcode.JumpTo(try_finally))

                for except_handler in handlers:
                    handler_start = Label()
                    handler_end = Label()
                    out = out.add_label(handler_start)
                    except_type = (
                        out._evaluate_constant_expr(except_handler.type)
                        if except_handler.type is not None
                        else BaseException
                    )
                    if except_handler.name is not None:
                        out = out.add_op(opcode.StoreLocal(except_handler.name))
                    else:
                        out = out.add_op(opcode.Pop())

                    for statement in except_handler.body:
                        out = out.with_statement_opcodes(statement)

                    out = out.add_label(handler_end)
                    out = out.add_op(opcode.JumpTo(try_finally))

                    out = out.add_exception_handler(
                        ExceptionHandler(
                            exception_class=except_type,
                            from_label=try_start,
                            to_label=try_end,
                            handler_label=handler_start,
                        )
                    )

                out = out.add_label(except_finally)
                out = out.pop_finally_block()

                out = out.add_exception_handler(
                    ExceptionHandler(
                        exception_class=BaseException,
                        from_label=try_start,
                        to_label=except_finally,
                        handler_label=except_finally,
                    )
                )

                for statement in finalbody:
                    out = out.with_statement_opcodes(statement)

                out = out.add_op(opcode.ReraiseLast())

                out = out.add_label(try_finally)

                for statement in finalbody:
                    out = out.with_statement_opcodes(statement)

                return out

            case ast.Assign(targets, value):
                out = self.with_expression_opcodes(value)
                for target in targets[:-1]:
                    out = out.add_op(opcode.Dup())
                    out = out.with_assignment_opcodes(target)
                out = out.with_assignment_opcodes(targets[-1])
                return out

            case ast.AugAssign(target, op, value):
                assignment_code = self.get_assignment_code(target)
                out = assignment_code.load_target(self)
                stack_items = []

                for _ in range(assignment_code.stack_items):
                    out = out.new_synthetic()
                    stack_items.append(out.current_synthetic)
                    out = out.add_ops(opcode.StoreSynthetic(out.current_synthetic))

                for i in reversed(range(assignment_code.stack_items)):
                    out = out.add_ops(opcode.LoadSynthetic(stack_items[i]))

                out = assignment_code.load_current_value(out)
                out = out.with_expression_opcodes(value)
                out = out.add_op(
                    opcode.BinaryOp(opcode.BinaryOperator["I" + type(op).__name__])
                )

                for i in reversed(range(assignment_code.stack_items)):
                    out = out.add_ops(opcode.LoadSynthetic(stack_items[i]))
                    out = out.free_synthetic()

                out = assignment_code.store_value(out)
                return out

            case ast.Delete(targets):
                out = self
                for target in targets:
                    assignment_code = out.get_assignment_code(target)
                    out = assignment_code.load_target(out)
                    out = assignment_code.delete_value(out)
                return out

            case ast.If(test, body, orelse):
                false_label = Label()
                done_label = Label()
                out = self.with_expression_opcodes(test)
                out = out.add_op(opcode.IfFalse(false_label))

                for body_statement in body:
                    out = out.with_statement_opcodes(body_statement)

                out = out.add_op(opcode.JumpTo(done_label))
                out = out.add_label(false_label)

                for else_statement in orelse:
                    out = out.with_statement_opcodes(else_statement)

                out = out.add_label(done_label)
                return out

            case ast.Break():
                return self.add_op(opcode.JumpTo(self.break_label))

            case ast.Continue():
                return self.add_op(opcode.JumpTo(self.continue_label))

            case ast.While(test, body, orelse):
                test_label = Label()
                false_label = Label()
                break_label = Label()
                out = self.add_label(test_label)
                out = out.with_expression_opcodes(test)

                out = out.add_op(opcode.IfFalse(false_label))

                out = out.push_continue_label(test_label)
                out = out.push_break_label(break_label)

                for body_statement in body:
                    out = out.with_statement_opcodes(body_statement)

                out = out.pop_continue_label()
                out = out.pop_break_label()

                out = out.add_op(opcode.JumpTo(test_label))
                out = out.add_label(false_label)

                for else_statement in orelse:
                    out = out.with_statement_opcodes(else_statement)

                out = out.add_label(break_label)
                return out

            case ast.For(target, iter, body, orelse, type_comment):
                out = self.new_synthetic()
                iterator_synthetic = out.current_synthetic
                next_label = Label()
                got_next_label = Label()
                iterator_exhausted_label = Label()
                break_label = Label()
                out = out.with_expression_opcodes(iter)
                out = out.add_op(opcode.GetIterator())
                out = out.add_op(opcode.StoreSynthetic(iterator_synthetic))
                out = out.add_label(next_label)
                out = out.add_op(opcode.LoadSynthetic(iterator_synthetic))
                out = out.add_op(opcode.TryNext())
                out = out.with_assignment_opcodes(target)
                out = out.add_label(got_next_label)
                out = out.add_exception_handler(
                    ExceptionHandler(
                        exception_class=StopIteration,
                        from_label=next_label,
                        to_label=got_next_label,
                        handler_label=iterator_exhausted_label,
                    )
                )

                out = out.push_continue_label(next_label)
                out = out.push_break_label(break_label)

                for body_statement in body:
                    out = out.with_statement_opcodes(body_statement)

                out = out.pop_continue_label()
                out = out.pop_break_label()

                out = out.add_op(opcode.JumpTo(next_label))
                out = out.add_label(iterator_exhausted_label)

                for else_statement in orelse:
                    out = out.with_statement_opcodes(else_statement)

                out = out.add_label(break_label)
                return out

            case _:
                raise NotImplementedError(
                    f"Not implemented statement: {type(statement)}"
                )

        raise RuntimeError(
            f"Missing return in case {type(statement)} in with_statement_opcodes"
        )


def unindent(code: str) -> str:
    lowest_indent = float("inf")
    for line in code.splitlines():
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            return code
        lowest_indent = min(lowest_indent, indent)

    out = ""
    for line in code.splitlines():
        out += line[lowest_indent:] + "\n"

    return out


def to_bytecode(function: FunctionType) -> Bytecode:
    source = inspect.getsource(function)
    source = unindent(source)
    function_ast: ast.FunctionDef = cast(ast.FunctionDef, ast.parse(source).body[0])

    free_vars = function.__code__.co_freevars
    cell_names = frozenset(function.__code__.co_cellvars + free_vars)
    local_names = frozenset(set(function.__code__.co_varnames) - set(cell_names))

    closure_dict = {}
    for i in range(len(free_vars)):
        closure_dict[free_vars[i]] = function.__closure__[i]

    bytecode = Bytecode(
        function_name=function.__name__,
        signature=inspect.signature(function),
        local_names=local_names,
        cell_names=cell_names,
        free_names=frozenset(function.__code__.co_freevars),
        globals=function.__globals__,
        closure=closure_dict,
    )

    for statement in function_ast.body:
        bytecode = bytecode.with_statement_opcodes(statement)

    bytecode = bytecode.with_statement_opcodes(
        ast.Return(
            ast.Constant(
                None,
                lineno=function_ast.body[-1].lineno,
            ),
            lineno=function_ast.body[-1].lineno,
        )
    )
    return bytecode
