from ._bytecode import Instruction, Label, InnerFunction
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
class RenamedVariable:
    read_opcode: opcode.Opcode
    write_opcode: opcode.Opcode
    delete_opcode: opcode.Opcode


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
            ast.Return(value=expression, lineno=0, col_offset=0), {}
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

    def with_comprehension_opcodes(
        self,
        generator: ast.comprehension,
        iterator_next_label: Label,
        iterator_exhausted_label: Label,
        renames: dict[str, RenamedVariable],
    ) -> tuple["Bytecode", dict[str, RenamedVariable],int]:
        out = self.with_expression_opcodes(generator.iter, renames)
        out = out.new_synthetic()
        out = out.add_op(opcode.GetIterator())
        out = out.add_op(opcode.StoreSynthetic(out.current_synthetic))
        out = out.add_label(iterator_next_label)
        out = out.add_op(opcode.LoadSynthetic(out.current_synthetic))
        out = out.add_op(opcode.GetNextElseJumpTo(target=iterator_exhausted_label))
        for condition in generator.ifs:
            out = out.with_expression_opcodes(condition, renames)
            out = out.add_op(opcode.IfFalse(target=iterator_next_label))
        out, new_renames, used_variables_count = out._get_renames(generator.target, renames)
        out = out.with_assignment_opcodes(generator.target, new_renames)
        return out, new_renames, used_variables_count + 1

    def _get_renames(self, target: ast.expr, renames: dict[str, RenamedVariable]) -> tuple['Bytecode', dict[str, RenamedVariable], int]:
        match target:
            case ast.Name(id=name):
                out = self.new_synthetic()
                renamed_variable = RenamedVariable(
                    read_opcode=opcode.LoadSynthetic(out.current_synthetic),
                    write_opcode=opcode.StoreSynthetic(out.current_synthetic),
                    # TODO: is DeleteSynthetic needed?
                    delete_opcode=opcode.Nop()
                )
                return out, { **renames, name: renamed_variable }, 1

            case ast.Attribute(attr) as a:
                return self, renames, 0

            case ast.Subscript(value, slice=slice_, ctx=ctx):
                return self, renames, 0

            case ast.Starred(value=inner):
                return self._get_renames(inner, renames)

            case ast.List(elts, ctx):
                return self._get_renames(ast.Tuple(elts=elts, ctx=ctx), renames)

            case ast.Tuple(elts):
                total_count = 0
                new_renames = renames
                new_bytecode = self
                for elt in elts:
                    new_bytecode, new_renames, added_count = new_bytecode._get_renames(elt, new_renames)
                    total_count += added_count
                return new_bytecode, new_renames, total_count

            case _:
                raise NotImplementedError(
                    f"Not implemented assignment: {type(target)}"
                )

        raise RuntimeError(
            f"Missing return in case {type(expression)} in get_assignment_code"
        )

    def with_expression_opcodes(self, expression: ast.expr, renames: dict[str, RenamedVariable]) -> "Bytecode":
        match expression:
            case ast.Constant(value):
                return self.add_expression(expression, opcode.LoadConstant(value))

            case ast.Attribute(attr) as a:
                out = self.with_expression_opcodes(a.value, renames)
                out = out.add_expression(expression, opcode.LoadAttr(attr.id))
                return out

            case ast.Name(id=name, ctx=ast.Load()):
                if name in renames:
                    return self.add_expression(expression, renames[name].read_opcode)
                if name in self.cell_names:
                    return self.add_expression(
                        expression, opcode.LoadCell(name, name in self.free_names)
                    )
                elif name in self.local_names:
                    return self.add_expression(expression, opcode.LoadLocal(name))
                else:
                    return self.add_expression(expression, opcode.LoadGlobal(name))

            case ast.UnaryOp(op, operand):
                out = self.with_expression_opcodes(operand, renames)
                match operand:
                    case ast.Not():
                        return out.with_negated_tos()
                    case _:
                        return out.add_expression(
                            expression,
                            opcode.UnaryOp(opcode.UnaryOperator[type(op).__name__]),
                        )

            case ast.BinOp(left, op, right):
                out = self.with_expression_opcodes(left, renames)
                out = out.with_expression_opcodes(right, renames)
                return out.add_expression(
                    expression,
                    opcode.BinaryOp(opcode.BinaryOperator[type(op).__name__]),
                )

            case ast.Compare(left, ops, comparators):
                out = self.with_expression_opcodes(left, renames)
                false_label = Label()
                for i in range(len(ops) - 1):
                    out = out.with_expression_opcodes(comparators[i], renames)
                    out = out.add_op(opcode.DupX1())
                    out = out.with_comparison_op(ops[i])

                    out = out.add_ops(
                        opcode.Dup(), opcode.IfFalse(false_label), opcode.Pop()
                    )

                out = out.with_expression_opcodes(comparators[-1], renames)
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
                    out = out.with_expression_opcodes(value, renames)
                    out = out.add_op(opcode.Dup())
                    out = out.add_op(jump_opcode(end_label))
                    out = out.add_op(opcode.Pop())
                out = out.with_expression_opcodes(values[-1], renames)
                return out.add_label(end_label)

            case ast.Call(func, args, keywords):
                out = self.with_expression_opcodes(func, renames)
                out = out.add_op(opcode.CreateCallBuilder())
                extended = False
                arg_index = 0
                for arg in args:
                    match arg:
                        case ast.Starred(value):
                            out = out.with_expression_opcodes(value, renames)
                            out = out.add_op(opcode.ExtendPositionalArgs())
                            extended = True
                        case _ as expression:
                            out = out.with_expression_opcodes(expression, renames)
                            if extended:
                                out = out.add_op(opcode.AppendPositionalArg())
                            else:
                                out = out.add_op(opcode.WithPositionalArg(arg_index))
                                arg_index += 1

                for kwarg in keywords:
                    out = out.with_expression_opcodes(kwarg.value, renames)
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
                            out = out.with_expression_opcodes(value, renames)
                            out = out.add_op(opcode.ListExtend())
                        case _:
                            out = out.with_expression_opcodes(elt, renames)
                            out = out.add_op(opcode.ListAppend())
                return out

            case ast.Tuple(elts):
                out = self.add_op(opcode.NewList())
                for elt in elts:
                    match elt:
                        case ast.Starred(value):
                            out = out.with_expression_opcodes(value, renames)
                            out = out.add_op(opcode.ListExtend())
                        case _:
                            out = out.with_expression_opcodes(elt, renames)
                            out = out.add_op(opcode.ListAppend())
                out = out.add_op(opcode.ListToTuple())
                return out

            case ast.Set(elts):
                out = self.add_op(opcode.NewSet())
                for elt in elts:
                    match elt:
                        case ast.Starred(value):
                            out = out.with_expression_opcodes(value, renames)
                            out = out.add_op(opcode.SetUpdate())
                        case _:
                            out = out.with_expression_opcodes(elt, renames)
                            out = out.add_op(opcode.SetAdd())
                return out

            case ast.Dict(keys, values):
                out = self.add_op(opcode.NewDict())
                for index, value in enumerate(values):
                    key = keys[index]
                    if key is None:
                        out = out.with_expression_opcodes(value, renames)
                        out = out.add_op(opcode.DictUpdate())
                    else:
                        out = out.with_expression_opcodes(key, renames)
                        out = out.with_expression_opcodes(value, renames)
                        out = out.add_op(opcode.DictPut())
                return out

            case ast.Subscript(value, slice=slice_, ctx=ctx):
                out = self.with_expression_opcodes(value, renames)
                out = out.with_expression_opcodes(slice_, renames)
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
                out = self.with_expression_opcodes(value, renames)
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
                    out = out.with_expression_opcodes(value, renames)
                out = out.add_op(opcode.JoinStringValues(count=len(values)))
                return out

            # TODO: ast.GeneratorExp

            case ast.ListComp(elt, generators):
                out = self.add_op(opcode.NewList())
                out = out.new_synthetic()
                list_variable_index = out.current_synthetic
                out = out.add_op(opcode.StoreSynthetic(list_variable_index))

                def add_to_list(new_bytecode: "Bytecode") -> "Bytecode":
                    new_bytecode = new_bytecode.add_op(opcode.LoadSynthetic(list_variable_index))
                    new_bytecode = new_bytecode.add_op(opcode.Swap())
                    new_bytecode = new_bytecode.add_op(opcode.ListAppend())
                    new_bytecode = new_bytecode.add_op(opcode.Pop())
                    return new_bytecode

                out = out.get_comprehension_code(add_to_list, (elt,), generators, renames)
                out = out.add_op(opcode.LoadSynthetic(list_variable_index))
                out = out.free_synthetic()
                return out

            case ast.SetComp(elt, generators):
                out = self.add_op(opcode.NewSet())
                out = out.new_synthetic()
                set_variable_index = out.current_synthetic
                out = out.add_op(opcode.StoreSynthetic(set_variable_index))

                def add_to_set(new_bytecode: "Bytecode") -> "Bytecode":
                    new_bytecode = new_bytecode.add_op(opcode.LoadSynthetic(set_variable_index))
                    new_bytecode = new_bytecode.add_op(opcode.Swap())
                    new_bytecode = new_bytecode.add_op(opcode.SetAdd())
                    new_bytecode = new_bytecode.add_op(opcode.Pop())
                    return new_bytecode

                out = out.get_comprehension_code(add_to_set, (elt,), generators, renames)
                out = out.add_op(opcode.LoadSynthetic(set_variable_index))
                out = out.free_synthetic()
                return out

            case ast.DictComp(key, value, generators):
                out = self.add_op(opcode.NewDict())
                out = out.new_synthetic()
                dict_variable_index = out.current_synthetic
                out = out.add_op(opcode.StoreSynthetic(dict_variable_index))

                def add_to_set(new_bytecode: "Bytecode") -> "Bytecode":
                    new_bytecode = new_bytecode.new_synthetic()
                    new_bytecode = new_bytecode.add_op(opcode.StoreSynthetic(new_bytecode.current_synthetic))
                    new_bytecode = new_bytecode.add_op(opcode.LoadSynthetic(dict_variable_index))
                    new_bytecode = new_bytecode.add_op(opcode.Swap())
                    new_bytecode = new_bytecode.add_op(opcode.LoadSynthetic(new_bytecode.current_synthetic))
                    new_bytecode = new_bytecode.add_op(opcode.DictPut())
                    new_bytecode = new_bytecode.add_op(opcode.Pop())
                    new_bytecode = new_bytecode.free_synthetic()
                    return new_bytecode

                out = out.get_comprehension_code(add_to_set, (key, value), generators, renames)
                out = out.add_op(opcode.LoadSynthetic(dict_variable_index))
                out = out.free_synthetic()
                return out

            case ast.Lambda(args):
                inner_function = self._create_lambda(expression)
                out = self
                for default_arg in args.defaults:
                    out = out.with_expression_opcodes(default_arg, renames)
                for default_arg in args.kw_defaults:
                    out = out.with_expression_opcodes(default_arg, renames)
                out = out.add_op(opcode.LoadAndBindInnerFunction(inner_function))
                return out

            case ast.IfExp(test, body, orelse):
                end_label = opcode.Label()
                orelse_label = opcode.Label()
                out = self
                out = out.with_expression_opcodes(test, renames)
                out = out.add_op(opcode.IfFalse(target=orelse_label))
                out = out.with_expression_opcodes(body, renames)
                out = out.add_op(opcode.JumpTo(target=end_label))
                out = out.add_label(orelse_label)
                out = out.with_expression_opcodes(orelse, renames)
                out = out.add_label(end_label)
                return out

            case _:
                raise NotImplementedError(
                    f"Not implemented statement: {type(expression)}"
                )

        raise RuntimeError(
            f"Missing return in case {type(expression)} in with_expression_opcodes"
        )

    def get_comprehension_code(self,
                               elt_consumer: Callable[["Bytecode"], "Bytecode"],
                               elts: tuple[ast.expr, ...],
                               generators: list[ast.comprehension],
                               renames: dict[str, RenamedVariable]):
        end_label = Label()
        used_variables_count = 0
        previous_generator_next_label = Label()
        out, new_renames, new_variables = self.with_comprehension_opcodes(generators[0],
                                                                          previous_generator_next_label,
                                                                          end_label,
                                                                          renames)
        used_variables_count += new_variables
        for generator in generators[1:]:
            new_generator_next_label = Label()
            out, new_renames, new_variables = out.with_comprehension_opcodes(generator,
                                                                             new_generator_next_label,
                                                                             previous_generator_next_label,
                                                                             new_renames)
            used_variables_count += new_variables
            previous_generator_next_label = new_generator_next_label

        for elt in elts:
            out = out.with_expression_opcodes(elt, new_renames)

        out = elt_consumer(out)
        out = out.add_op(opcode.JumpTo(target=previous_generator_next_label))
        out = out.add_label(end_label)
        for used_variable in range(used_variables_count):
            out = out.free_synthetic()
        return out

    def get_assignment_code(self, expression: ast.expr, renames: dict[str, RenamedVariable]) -> AssignmentCode:
        match expression:
            case ast.Name(id=name):
                if name in renames:
                    return AssignmentCode(
                        stack_items=0,
                        load_target=lambda b: b,
                        load_current_value=lambda b: b.add_expression(
                            expression, renames[name].read_opcode
                        ),
                        store_value=lambda b: b.add_expression(
                            expression, renames[name].write_opcode
                        ),
                        delete_value=lambda b: b.add_expression(
                            expression, renames[name].delete_opcode
                        ),
                    )
                elif name in self.cell_names:
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
                    load_target=lambda b: b.with_expression_opcodes(a.value, renames),
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
                        value, renames
                    ).with_expression_opcodes(slice_, renames),
                    load_current_value=lambda b: b.add_op(opcode.GetItem()),
                    store_value=lambda b: b.add_op(opcode.SetItem()),
                    delete_value=lambda b: b.add_op(opcode.DeleteItem()),
                )

            case ast.Starred(value=inner):
                return self.get_assignment_code(inner, renames)

            case ast.List(elts, ctx):
                return self.get_assignment_code(ast.Tuple(elts=elts, ctx=ctx), renames)

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
                        assignment_code = bytecode.get_assignment_code(elt, renames)
                        bytecode = assignment_code.load_target(bytecode)
                        bytecode = assignment_code.store_value(bytecode)

                    return bytecode

                def delete_elts(bytecode: "Bytecode") -> "Bytecode":
                    for elt in elts:
                        assignment_code = bytecode.get_assignment_code(elt, renames)
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

    def with_assignment_opcodes(self, expression: ast.expr, renames: dict[str, RenamedVariable]) -> "Bytecode":
        assignment_code = self.get_assignment_code(expression, renames)
        return assignment_code.store_value(assignment_code.load_target(self))

    def with_statement_opcodes(self, statement: ast.stmt, renames: dict[str, RenamedVariable]) -> "Bytecode":
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
                out = self.with_expression_opcodes(value, renames)
                out = out.add_op(opcode.Pop())
                return out

            case ast.Return(value):
                if value is not None:
                    out = self.with_expression_opcodes(value, renames)
                else:
                    out = self.with_expression_opcodes(
                        ast.Constant(None, lineno=statement.lineno), renames
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
                        out = out.with_statement_opcodes(statement, renames)

                out = replace(out, finally_blocks=finally_blocks)
                out = out.add_op(opcode.LoadSynthetic(return_value_holder))
                out = out.free_synthetic()
                return out.add_statement(statement, opcode.ReturnValue())

            case ast.Raise(exc, cause):
                if exc is None:
                    return self.add_op(opcode.ReraiseLast())
                out = self.with_expression_opcodes(exc, renames)
                if cause is None:
                    return out.add_op(opcode.Raise())
                out = out.with_expression_opcodes(cause, renames)
                return out.add_op(opcode.RaiseWithCause())

            case ast.Try(body, handlers, orelse, finalbody):
                try_start = Label()
                try_end = Label()
                except_finally = Label()
                try_finally = Label()

                out = self.add_label(try_start)
                out = out.push_finally_block(finalbody)
                for statement in body:
                    out = out.with_statement_opcodes(statement, renames)
                out = out.add_label(try_end)

                for statement in orelse:
                    out = out.with_statement_opcodes(statement, renames)

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
                        out = out.with_statement_opcodes(statement, renames)

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
                    out = out.with_statement_opcodes(statement, renames)

                out = out.add_op(opcode.ReraiseLast())

                out = out.add_label(try_finally)

                for statement in finalbody:
                    out = out.with_statement_opcodes(statement, renames)

                return out

            case ast.Assign(targets, value):
                out = self.with_expression_opcodes(value, renames)
                for target in targets[:-1]:
                    out = out.add_op(opcode.Dup())
                    out = out.with_assignment_opcodes(target, renames)
                out = out.with_assignment_opcodes(targets[-1], renames)
                return out

            case ast.AugAssign(target, op, value):
                assignment_code = self.get_assignment_code(target, renames)
                out = assignment_code.load_target(self)
                stack_items = []

                for _ in range(assignment_code.stack_items):
                    out = out.new_synthetic()
                    stack_items.append(out.current_synthetic)
                    out = out.add_ops(opcode.StoreSynthetic(out.current_synthetic))

                for i in reversed(range(assignment_code.stack_items)):
                    out = out.add_ops(opcode.LoadSynthetic(stack_items[i]))

                out = assignment_code.load_current_value(out)
                out = out.with_expression_opcodes(value, renames)
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
                    assignment_code = out.get_assignment_code(target, renames)
                    out = assignment_code.load_target(out)
                    out = assignment_code.delete_value(out)
                return out

            case ast.If(test, body, orelse):
                false_label = Label()
                done_label = Label()
                out = self.with_expression_opcodes(test, renames)
                out = out.add_op(opcode.IfFalse(false_label))

                for body_statement in body:
                    out = out.with_statement_opcodes(body_statement, renames)

                out = out.add_op(opcode.JumpTo(done_label))
                out = out.add_label(false_label)

                for else_statement in orelse:
                    out = out.with_statement_opcodes(else_statement, renames)

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
                out = out.with_expression_opcodes(test, renames)

                out = out.add_op(opcode.IfFalse(false_label))

                out = out.push_continue_label(test_label)
                out = out.push_break_label(break_label)

                for body_statement in body:
                    out = out.with_statement_opcodes(body_statement, renames)

                out = out.pop_continue_label()
                out = out.pop_break_label()

                out = out.add_op(opcode.JumpTo(test_label))
                out = out.add_label(false_label)

                for else_statement in orelse:
                    out = out.with_statement_opcodes(else_statement, renames)

                out = out.add_label(break_label)
                return out

            case ast.For(target, iter, body, orelse, type_comment):
                out = self.new_synthetic()
                iterator_synthetic = out.current_synthetic
                next_label = Label()
                got_next_label = Label()
                iterator_exhausted_label = Label()
                break_label = Label()
                out = out.with_expression_opcodes(iter, renames)
                out = out.add_op(opcode.GetIterator())
                out = out.add_op(opcode.StoreSynthetic(iterator_synthetic))
                out = out.add_label(next_label)
                out = out.add_op(opcode.LoadSynthetic(iterator_synthetic))
                out = out.add_op(opcode.GetNextElseJumpTo(target=iterator_exhausted_label))
                out = out.with_assignment_opcodes(target, renames)
                out = out.add_label(got_next_label)

                out = out.push_continue_label(next_label)
                out = out.push_break_label(break_label)

                for body_statement in body:
                    out = out.with_statement_opcodes(body_statement, renames)

                out = out.pop_continue_label()
                out = out.pop_break_label()

                out = out.add_op(opcode.JumpTo(next_label))
                out = out.add_label(iterator_exhausted_label)

                for else_statement in orelse:
                    out = out.with_statement_opcodes(else_statement, renames)

                out = out.add_label(break_label)
                return out

            case ast.FunctionDef() as func_def:
                inner_function = self._create_inner_function(func_def)
                out = self
                for default_arg in func_def.args.defaults:
                    out = out.with_expression_opcodes(default_arg, renames)
                for default_arg in func_def.args.kw_defaults:
                    out = out.with_expression_opcodes(default_arg, renames)
                out = out.add_op(opcode.LoadAndBindInnerFunction(inner_function))
                for decorator in func_def.decorator_list:
                    pass  # TODO
                out = out.with_assignment_opcodes(ast.Name(id=func_def.name, ctx=ast.Store), renames)
                return out

            case _:
                raise NotImplementedError(
                    f"Not implemented statement: {type(statement)}"
                )

        raise RuntimeError(
            f"Missing return in case {type(statement)} in with_statement_opcodes"
        )

    def _create_inner_function(self, func_def: ast.FunctionDef) -> InnerFunction:
        signature_and_parameters = get_signature_from_arguments(func_def.args)
        used_variables = find_used_variables(func_def, self.local_names | self.cell_names | self.free_names)
        free_name_set = self.cell_names & used_variables.variable_names
        closure = (CellType(),) * len(free_name_set)

        inner_bytecode = ast_to_bytecode(func_def, signature_and_parameters.signature, self.globals,
                                         closure,
                                         tuple(free_name_set),
                                         tuple(used_variables.cell_names),
                                         tuple(used_variables.variable_names)
                                         )
        return InnerFunction(bytecode=inner_bytecode,
                             parameters_with_defaults=tuple(signature_and_parameters.parameters_with_defaults))

    def _create_lambda(self, lambda_def: ast.Lambda) -> InnerFunction:
        signature_and_parameters = get_signature_from_arguments(lambda_def.args)
        used_variables = find_used_variables(lambda_def, self.local_names | self.cell_names | self.free_names)
        free_name_set = self.cell_names & used_variables.variable_names
        closure = (CellType(),) * len(free_name_set)
        inner_bytecode = lambda_ast_to_bytecode(lambda_def, signature_and_parameters.signature, self.globals,
                                         closure,
                                         tuple(free_name_set),
                                         tuple(used_variables.cell_names),
                                         tuple(used_variables.variable_names)
                                         )
        return InnerFunction(bytecode=inner_bytecode,
                             parameters_with_defaults=signature_and_parameters.parameters_with_defaults)


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
    for param in args.args:
        parameters.append(inspect.Parameter(
            name=param.arg,
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=param.annotation,
        ))
    for param in args.posonlyargs:
        parameters.append(inspect.Parameter(
            name=param.arg,
            kind=inspect.Parameter.POSITIONAL_ONLY,
            annotation=param.annotation,
        ))
    if args.vararg:
        parameters.append(inspect.Parameter(
            name=args.vararg.arg,
            kind=inspect.Parameter.VAR_POSITIONAL,
            annotation=args.vararg.annotation,
        ))
    for param in args.kwonlyargs:
        parameters.append(inspect.Parameter(
            name=param.arg,
            kind=inspect.Parameter.KEYWORD_ONLY,
            annotation=param.annotation,
        ))
    if args.kwarg:
        parameters.append(inspect.Parameter(
            name=args.vararg.arg,
            kind=inspect.Parameter.VAR_KEYWORD,
            annotation=args.vararg.annotation,
        ))
    signature = inspect.Signature(parameters)
    remaining_pos_only = len(args.posonlyargs)
    remaining_pos_or_kw = len(args.args)
    for i in range(len(args.defaults)):
        if remaining_pos_only > 0:
            parameters_with_defaults.append(args.posonlyargs[remaining_pos_only - 1].arg)
            remaining_pos_only -= 1
        else:
            parameters_with_defaults.append(args.args[remaining_pos_or_kw - 1].arg)
            remaining_pos_or_kw -= 1

    for i in range(len(args.kw_defaults)):
        if args.kw_defaults[i] is not None:
            parameters_with_defaults.append(args.kwonlyargs[i].arg)

    return SignatureAndParameters(signature=signature,
                                  parameters_with_defaults=tuple(parameters_with_defaults))


def find_used_variables(func_def: ast.FunctionDef | ast.Lambda, outer_variables: frozenset[str]) -> UsedVariables:
    variable_names = set()
    cell_names = set()
    def add_names_to_variable_names(assignment_target):
        match assignment_target:
            case ast.Name(id=name):
                variable_names.add(name)
                if name in outer_variables:
                    cell_names.add(name)

            case ast.Attribute() as a:
                pass

            case ast.Subscript():
                pass

            case ast.Starred(value=inner):
                add_names_to_variable_names(inner)

            case ast.List(elts):
                for list_item in elts:
                    add_names_to_variable_names(list_item)

            case ast.Tuple(elts):
                for list_item in elts:
                    add_names_to_variable_names(list_item)

            case _:
                raise NotImplementedError(
                    f"Not implemented assignment: {type(assignment_target)}"
                )

    def iterate_node(node):
        for child in ast.iter_child_nodes(node):
            match child:
                case ast.Assign(targets):
                    for target in targets:
                        add_names_to_variable_names(target)

                case ast.Name(id=name):
                    if name in outer_variables:
                        variable_names.add(name)
                        cell_names.add(name)

                case _:
                    iterate_node(child)

    match func_def:
        # Do not capture body; that make it a str instead of an ast object!
        case ast.FunctionDef():
            for statement in func_def.body:
                iterate_node(statement)

        case ast.Lambda():
            iterate_node(func_def.body)

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

    return UsedVariables(variable_names=frozenset(variable_names),
                         cell_names=frozenset(cell_names))


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

    return ast_to_bytecode(function_ast, inspect.signature(function),
                           function.__globals__, function.__closure__,
                           function.__code__.co_freevars, function.__code__.co_cellvars,
                           function.__code__.co_varnames)


def ast_to_bytecode(function_ast: ast.FunctionDef,
                    signature: inspect.Signature,
                    function_globals: dict[str, object],
                    function_closure: tuple[CellType, ...],
                    free_names: tuple[str, ...],
                    cell_names: tuple[str, ...],
                    variable_names: tuple[str, ...]) -> Bytecode:
    free_vars = free_names
    cell_names = frozenset(cell_names + free_vars)
    local_names = frozenset(set(variable_names) - set(cell_names))

    closure_dict = {}
    for i in range(len(free_vars)):
        closure_dict[free_vars[i]] = function_closure[i]

    bytecode = Bytecode(
        function_name=function_ast.name,
        signature=signature,
        local_names=local_names,
        cell_names=cell_names,
        free_names=frozenset(free_vars),
        globals=function_globals,
        closure=closure_dict,
    )

    parameter_cells = (cell_names - frozenset(free_names)) & {name for name in variable_names[:len(signature.parameters)]}
    for parameter_cell_var in parameter_cells:
        bytecode = bytecode.add_op(opcode.LoadLocal(name=parameter_cell_var))
        bytecode = bytecode.add_op(opcode.StoreCell(name=parameter_cell_var, is_free=False))
        closure_dict[parameter_cell_var] = CellType()

    for statement in function_ast.body:
        bytecode = bytecode.with_statement_opcodes(statement, {})

    bytecode = bytecode.with_statement_opcodes(
        ast.Return(
            ast.Constant(
                None,
                lineno=function_ast.body[-1].lineno,
            ),
            lineno=function_ast.body[-1].lineno,
        )
    , {})
    return bytecode


def lambda_ast_to_bytecode(lambda_ast: ast.Lambda,
                    signature: inspect.Signature,
                    function_globals: dict[str, object],
                    function_closure: tuple[CellType, ...],
                    free_names: tuple[str, ...],
                    cell_names: tuple[str, ...],
                    variable_names: tuple[str, ...]) -> Bytecode:
    free_vars = free_names
    cell_names = frozenset(cell_names + free_vars)
    local_names = frozenset(set(variable_names) - set(cell_names))

    closure_dict = {}
    for i in range(len(free_vars)):
        closure_dict[free_vars[i]] = function_closure[i]

    bytecode = Bytecode(
        function_name='lambda',
        signature=signature,
        local_names=local_names,
        cell_names=cell_names,
        free_names=frozenset(free_vars),
        globals=function_globals,
        closure=closure_dict,
    )

    parameter_cells = (cell_names - frozenset(free_names)) & {name for name in variable_names[:len(signature.parameters)]}
    for parameter_cell_var in parameter_cells:
        bytecode = bytecode.add_op(opcode.LoadLocal(name=parameter_cell_var))
        bytecode = bytecode.add_op(opcode.StoreCell(name=parameter_cell_var, is_free=False))
        closure_dict[parameter_cell_var] = CellType()

    bytecode = bytecode.with_statement_opcodes(
        ast.Return(
            lambda_ast.body,
            lineno=lambda_ast.body.lineno,
        )
    , {})
    return bytecode