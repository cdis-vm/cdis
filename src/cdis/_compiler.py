import types
from dis import Bytecode

from ._bytecode import Instruction, Label, InnerFunction, StackMetadata
from ._type_analysis import resolve_stack_metadata
from cdis import bytecode as opcode
from dataclasses import dataclass, replace, field
from functools import cached_property
from types import FunctionType, CellType
from typing import cast, Callable, Iterable, Union
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
    function_type: opcode.FunctionType
    method_type: opcode.MethodType
    generator_instance_parameter: str | None
    signature: inspect.Signature
    annotate_function: Union["Bytecode", None] = None
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
    yield_labels: tuple[Label, ...] = ()
    exception_handlers: tuple[ExceptionHandler, ...] = ()
    finally_blocks: tuple[tuple[ast.stmt, ...], ...] = ()
    deferred_processors: tuple[Callable[[Bytecode], Bytecode], ...] = ()

    @cached_property
    def stack_metadata(self) -> tuple[StackMetadata, ...]:
        return resolve_stack_metadata(self)

    @property
    def current_yield(self) -> int:
        return len(self.yield_labels)

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
        out += "instructions:\n"
        for instruction in self.instructions:
            out += f"{instruction.bytecode_index:<16}{str(instruction.opcode)}\n"
        return out

    def _evaluate_constant_expr(self, expression: ast.expr):
        from cdis._vm import CDisVM

        # Make FunctionType FUNCTION, since other FunctionTypes might augment
        # return and expect the stack to be in a certain state
        new_bytecode = replace(
            self,
            function_type=opcode.FunctionType.FUNCTION,
            method_type=opcode.MethodType.VIRTUAL,
            instructions=(),
            finally_blocks=(),
            signature=inspect.Signature(),
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

    def add_yield_label(self, label: Label) -> "Bytecode":
        out = self.add_label(label)
        return replace(out, yield_labels=out.yield_labels + (label,))

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

    def add_deferred_processor(
        self, processor: Callable[[Bytecode], Bytecode]
    ) -> "Bytecode":
        return replace(
            self, deferred_processors=self.deferred_processors + (processor,)
        )

    def process_deferred(self) -> "Bytecode":
        out = self
        for processor in self.deferred_processors:
            out = processor(out)
        replace(out, deferred_processors=())
        return out

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
    ) -> tuple["Bytecode", dict[str, RenamedVariable], int]:
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
        out, new_renames, used_variables_count = out._get_renames(
            generator.target, renames
        )
        out = out.with_assignment_opcodes(generator.target, new_renames)
        return out, new_renames, used_variables_count + 1

    def _get_renames(
        self, target: ast.expr, renames: dict[str, RenamedVariable]
    ) -> tuple["Bytecode", dict[str, RenamedVariable], int]:
        match target:
            case ast.Name(id=name):
                out = self.new_synthetic()
                renamed_variable = RenamedVariable(
                    read_opcode=opcode.LoadSynthetic(out.current_synthetic),
                    write_opcode=opcode.StoreSynthetic(out.current_synthetic),
                    # TODO: is DeleteSynthetic needed?
                    delete_opcode=opcode.Nop(),
                )
                return out, {**renames, name: renamed_variable}, 1

            case ast.Attribute():
                return self, renames, 0

            case ast.Subscript():
                return self, renames, 0

            case ast.Starred(value=inner):
                return self._get_renames(inner, renames)

            case ast.List(elts=elts, ctx=ctx):
                return self._get_renames(ast.Tuple(elts=elts, ctx=ctx), renames)

            case ast.Tuple(elts=elts):
                total_count = 0
                new_renames = renames
                new_bytecode = self
                for elt in elts:
                    new_bytecode, new_renames, added_count = new_bytecode._get_renames(
                        elt, new_renames
                    )
                    total_count += added_count
                return new_bytecode, new_renames, total_count

            case _:
                raise NotImplementedError(f"Not implemented assignment: {type(target)}")

        raise RuntimeError(f"Missing return in case {type(target)} in _get_renames")

    def with_expression_opcodes(
        self, expression: ast.expr, renames: dict[str, RenamedVariable]
    ) -> "Bytecode":
        match expression:
            case ast.Constant(value=value):
                return self.add_expression(expression, opcode.LoadConstant(value))

            case ast.Attribute(attr=attr_name) as a:
                out = self.with_expression_opcodes(a.value, renames)
                out = out.add_expression(expression, opcode.LoadAttr(attr_name))
                return out

            case ast.Name(id=name, ctx=ast.Load()):
                if name in renames:
                    return self.add_expression(expression, renames[name].read_opcode)
                if name in self.cell_names:
                    return self.add_expression(
                        expression, opcode.LoadCell(name, name in self.free_names)
                    )
                elif (
                    name in self.local_names
                    and self.function_type is not opcode.FunctionType.CLASS_BODY
                ):
                    return self.add_expression(expression, opcode.LoadLocal(name))
                elif (
                    name in self.global_names
                    or self.function_type is not opcode.FunctionType.CLASS_BODY
                ):
                    return self.add_expression(expression, opcode.LoadGlobal(name))
                else:
                    return self.add_ops(
                        opcode.LoadSynthetic(0), opcode.LoadTypeAttrOrGlobal(name=name)
                    )

            case ast.NamedExpr(target, value):
                out = self.with_expression_opcodes(value, renames)
                out = out.add_op(opcode.Dup())
                return out.with_assignment_opcodes(target, renames)

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

            case ast.Slice(lower=start, upper=stop, step=step):
                out = self
                if start is not None:
                    out = out.with_expression_opcodes(start, renames)
                else:
                    out = out.add_op(opcode.LoadConstant(None))

                if stop is not None:
                    out = out.with_expression_opcodes(stop, renames)
                else:
                    out = out.add_op(opcode.LoadConstant(None))

                if step is not None:
                    out = out.with_expression_opcodes(step, renames)
                else:
                    out = out.add_op(opcode.LoadConstant(None))

                return out.add_expression(expression, opcode.BuildSlice())

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

            case ast.Subscript(value=value, slice=slice_, ctx=ctx):
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

            case ast.GeneratorExp(elt=elt, generators=generators):
                generator_bytecode = Bytecode(
                    function_name=f"{self.function_name}_generator",
                    function_type=opcode.FunctionType.GENERATOR,
                    method_type=opcode.MethodType.VIRTUAL,
                    generator_instance_parameter="_generator",
                    signature=inspect.Signature(
                        parameters=(
                            inspect.Parameter(
                                name="generator",
                                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            ),
                        ),
                    ),
                    global_names=self.global_names,
                    cell_names=self.cell_names,
                    local_names=self.local_names | {"generator"},
                )
                generator_bytecode = _generator_function_prologue(generator_bytecode)

                def yield_value(new_bytecode: "Bytecode") -> "Bytecode":
                    new_bytecode = new_bytecode._yield_tos()
                    new_bytecode = new_bytecode.add_op(opcode.Pop())
                    return new_bytecode

                generator_bytecode = generator_bytecode.get_comprehension_code(
                    yield_value,
                    (elt,),
                    [
                        ast.comprehension(
                            target=generators[0].target,
                            iter=ast.Name(id="generator", ctx=ast.Load()),
                            ifs=generators[0].ifs,
                            is_async=generators[0].is_async,
                        )
                    ]
                    + generators[1:],
                    renames,
                )
                generator_bytecode = generator_bytecode.with_statement_opcodes(
                    ast.Return(ast.Constant(None)), {}
                )
                generator_bytecode.process_deferred()
                generator_class = _create_generator_class_from_bytecode(
                    generator_bytecode, frozenset()
                )
                out = self.add_ops(
                    opcode.LoadAndBindInnerGenerator(generator_class),
                    opcode.CreateCallBuilder(),
                )
                # First generator must be evaluated before generator is created
                out = out.with_expression_opcodes(generators[0].iter, renames)
                out = out.add_ops(
                    opcode.GetIterator(),
                    opcode.WithPositionalArg(index=0),
                    opcode.CallWithBuilder(),
                )

                return out

            case ast.ListComp(elt, generators):
                out = self.add_op(opcode.NewList())
                out = out.new_synthetic()
                list_variable_index = out.current_synthetic
                out = out.add_op(opcode.StoreSynthetic(list_variable_index))

                def add_to_list(new_bytecode: "Bytecode") -> "Bytecode":
                    new_bytecode = new_bytecode.add_op(
                        opcode.LoadSynthetic(list_variable_index)
                    )
                    new_bytecode = new_bytecode.add_op(opcode.Swap())
                    new_bytecode = new_bytecode.add_op(opcode.ListAppend())
                    new_bytecode = new_bytecode.add_op(opcode.Pop())
                    return new_bytecode

                out = out.get_comprehension_code(
                    add_to_list, (elt,), generators, renames
                )
                out = out.add_op(opcode.LoadSynthetic(list_variable_index))
                out = out.free_synthetic()
                return out

            case ast.SetComp(elt, generators):
                out = self.add_op(opcode.NewSet())
                out = out.new_synthetic()
                set_variable_index = out.current_synthetic
                out = out.add_op(opcode.StoreSynthetic(set_variable_index))

                def add_to_set(new_bytecode: "Bytecode") -> "Bytecode":
                    new_bytecode = new_bytecode.add_op(
                        opcode.LoadSynthetic(set_variable_index)
                    )
                    new_bytecode = new_bytecode.add_op(opcode.Swap())
                    new_bytecode = new_bytecode.add_op(opcode.SetAdd())
                    new_bytecode = new_bytecode.add_op(opcode.Pop())
                    return new_bytecode

                out = out.get_comprehension_code(
                    add_to_set, (elt,), generators, renames
                )
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
                    new_bytecode = new_bytecode.add_op(
                        opcode.StoreSynthetic(new_bytecode.current_synthetic)
                    )
                    new_bytecode = new_bytecode.add_op(
                        opcode.LoadSynthetic(dict_variable_index)
                    )
                    new_bytecode = new_bytecode.add_op(opcode.Swap())
                    new_bytecode = new_bytecode.add_op(
                        opcode.LoadSynthetic(new_bytecode.current_synthetic)
                    )
                    new_bytecode = new_bytecode.add_op(opcode.DictPut())
                    new_bytecode = new_bytecode.add_op(opcode.Pop())
                    new_bytecode = new_bytecode.free_synthetic()
                    return new_bytecode

                out = out.get_comprehension_code(
                    add_to_set, (key, value), generators, renames
                )
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

            case ast.Yield(value=value):
                out = self.with_expression_opcodes(value, renames)
                return out._yield_tos()

            case ast.YieldFrom(value=value):
                out = self.with_expression_opcodes(value, renames)
                out = out.add_op(opcode.GetIterator())
                return out._yield_from_tos()

            case ast.Await(value=value):
                out = self.with_expression_opcodes(value, renames)
                return out._await_tos()

            case _:
                raise NotImplementedError(
                    f"Not implemented statement: {type(expression)}"
                )

        raise RuntimeError(
            f"Missing return in case {type(expression)} in with_expression_opcodes"
        )

    def _yield_tos(self):
        out = self
        yield_id = out.current_yield
        out = out.add_op(opcode.LoadSynthetic(0))

        save_bytecode_index = len(out.instructions)
        out = out.add_op(
            opcode.SaveGeneratorState(
                state_id=yield_id,
                stack_metadata=cast(StackMetadata, None),
            )
        )

        out = out.add_op(opcode.YieldValue())
        yield_label = Label()
        out = out.add_yield_label(yield_label)
        out = out.add_label(yield_label)
        out = out.add_op(opcode.LoadSynthetic(0))

        restore_bytecode_index = len(out.instructions)
        out = out.add_op(
            opcode.DelegateOrRestoreGeneratorState(
                state_id=yield_id,
                stack_metadata=cast(StackMetadata, None),
            )
        )

        def deferred_processor(finalized_bytecode: "Bytecode"):
            stack_metadata = finalized_bytecode.stack_metadata[save_bytecode_index - 1]
            old_save_instructions = finalized_bytecode.instructions[save_bytecode_index]
            old_restore_instruction = finalized_bytecode.instructions[
                restore_bytecode_index
            ]

            new_instructions = (
                *finalized_bytecode.instructions[:save_bytecode_index],
                Instruction(
                    bytecode_index=save_bytecode_index,
                    lineno=old_save_instructions.lineno,
                    opcode=opcode.SaveGeneratorState(
                        state_id=yield_id, stack_metadata=stack_metadata
                    ),
                ),
                *finalized_bytecode.instructions[
                    save_bytecode_index + 1 : restore_bytecode_index
                ],
                Instruction(
                    bytecode_index=restore_bytecode_index,
                    lineno=old_restore_instruction.lineno,
                    opcode=opcode.DelegateOrRestoreGeneratorState(
                        state_id=yield_id, stack_metadata=stack_metadata
                    ),
                ),
                *finalized_bytecode.instructions[restore_bytecode_index + 1 :],
            )
            return replace(finalized_bytecode, instructions=new_instructions)

        out = out.add_deferred_processor(deferred_processor)
        return out

    def _await_tos(self):
        out = self.add_op(opcode.GetAwaitableIterator())
        return out._yield_from_tos()

    def _yield_from_tos(self):
        out = self
        yield_id = out.current_yield
        out = out.add_op(opcode.LoadSynthetic(0))
        out = out.add_op(opcode.SetGeneratorDelegate())

        out = out.add_op(opcode.LoadConstant(None))
        out = out.add_op(opcode.LoadSynthetic(0))
        save_bytecode_index = len(out.instructions)
        out = out.add_op(
            opcode.SaveGeneratorState(
                state_id=yield_id,
                stack_metadata=cast(StackMetadata, None),
            )
        )

        yield_label = Label()
        out = out.add_yield_label(yield_label)
        out = out.add_label(yield_label)

        out = out.add_op(opcode.LoadSynthetic(0))

        restore_bytecode_index = len(out.instructions)
        out = out.add_op(
            opcode.DelegateOrRestoreGeneratorState(
                state_id=yield_id,
                stack_metadata=cast(StackMetadata, None),
            )
        )

        def deferred_processor(finalized_bytecode: "Bytecode"):
            stack_metadata = finalized_bytecode.stack_metadata[save_bytecode_index - 1]
            old_save_instructions = finalized_bytecode.instructions[save_bytecode_index]
            old_restore_instruction = finalized_bytecode.instructions[
                restore_bytecode_index
            ]

            new_instructions = (
                *finalized_bytecode.instructions[:save_bytecode_index],
                Instruction(
                    bytecode_index=save_bytecode_index,
                    lineno=old_save_instructions.lineno,
                    opcode=opcode.SaveGeneratorState(
                        state_id=yield_id, stack_metadata=stack_metadata
                    ),
                ),
                *finalized_bytecode.instructions[
                    save_bytecode_index + 1 : restore_bytecode_index
                ],
                Instruction(
                    bytecode_index=restore_bytecode_index,
                    lineno=old_restore_instruction.lineno,
                    opcode=opcode.DelegateOrRestoreGeneratorState(
                        state_id=yield_id, stack_metadata=stack_metadata
                    ),
                ),
                *finalized_bytecode.instructions[restore_bytecode_index + 1 :],
            )
            return replace(finalized_bytecode, instructions=new_instructions)

        out = out.add_deferred_processor(deferred_processor)
        return out

    def _with(
        self, with_items, body, *, is_async: bool, renames: dict[str, RenamedVariable]
    ):
        exit_callables = []
        exit_managers = []
        out = self

        enter_method = "__aenter__" if is_async else "__enter__"
        exit_method = "__aexit__" if is_async else "__exit__"

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

        for with_item in with_items:
            item_start_label = Label()
            item_done_label = Label()
            next_label = Label()
            handler_start = Label()

            out = out.add_label(item_start_label)
            out = out.with_expression_opcodes(with_item.context_expr, renames)

            out = out.new_synthetic()
            exit_managers.append(out.current_synthetic)
            out = out.new_synthetic()
            exit_callables.append(out.current_synthetic)

            out = out.add_ops(
                opcode.Dup(),
                opcode.Dup(),
                opcode.StoreSynthetic(index=exit_managers[-1]),
                opcode.LoadObjectTypeAttr(name=enter_method),
                opcode.Swap(),
                opcode.LoadObjectTypeAttr(name=exit_method),
                opcode.StoreSynthetic(index=exit_callables[-1]),
                opcode.CreateCallBuilder(),
                opcode.LoadSynthetic(index=exit_managers[-1]),
                opcode.WithPositionalArg(index=0),
                opcode.CallWithBuilder(),
            )
            if is_async:
                out = out._await_tos()

            if with_item.optional_vars is not None:
                out = out.with_assignment_opcodes(with_item.optional_vars, renames)
            else:
                out = out.add_op(opcode.Pop())

            out.add_label(item_done_label)
            # If an exception is raised when calling __enter__,
            # do not shallow the exception, even if __exit__ return True
            if len(exit_callables) != 1:
                # Call exits of all context managers initialized up to this point
                out = out.add_op(opcode.JumpTo(next_label))
                out = out.add_exception_handler(
                    ExceptionHandler(
                        exception_class=BaseException,
                        from_label=item_start_label,
                        to_label=item_done_label,
                        handler_label=handler_start,
                    )
                )
                out = out.add_label(handler_start)
                out = out.new_synthetic()
                exception = out.current_synthetic
                out = out.add_op(opcode.StoreSynthetic(index=exception))
                for _index in range(len(exit_managers) - 1):
                    exit_manager = exit_managers[_index]
                    exit_callable = exit_callables[_index]
                    out = out.add_ops(
                        *call_exit(exception, exit_manager, exit_callable)
                    )
                    if is_async:
                        out = out._await_tos()
                    out.add_op(opcode.Pop())
                out = out.free_synthetic()
                out = out.add_op(opcode.ReraiseLast())
                out = out.add_label(next_label)
        with_start = Label()
        with_end = Label()
        handler_start = Label()
        handler_end = Label()
        out = out.add_label(with_start)
        for inner_stmt in body:
            out = out.with_statement_opcodes(inner_stmt, renames)
        out = out.add_op(opcode.JumpTo(handler_end))
        out = out.add_label(with_end)
        out = out.add_label(handler_start)
        out = out.add_exception_handler(
            ExceptionHandler(
                exception_class=BaseException,
                from_label=with_start,
                to_label=with_end,
                handler_label=handler_start,
            )
        )
        out = out.new_synthetic()
        exception = out.current_synthetic
        out = out.add_op(opcode.StoreSynthetic(index=exception))
        out = out.add_op(opcode.LoadConstant(constant=False))
        for _index in range(len(exit_managers)):
            exit_manager = exit_managers[_index]
            exit_callable = exit_callables[_index]
            out = out.add_ops(*call_exit(exception, exit_manager, exit_callable))
            if is_async:
                out = out._await_tos()
            # if there are multiple context managers,
            # the exception is shallow if ANY of their exits
            # return a truthy value
            out = out.add_op(opcode.AsBool())
            out = out.add_op(opcode.BinaryOp(operator=opcode.BinaryOperator.BitOr))
        out = out.add_op(opcode.IfTrue(target=handler_end))
        out = out.add_op(opcode.ReraiseLast())
        out = out.free_synthetic()
        out = out.add_label(handler_end)
        for _index in range(len(exit_managers)):
            exit_manager = exit_managers[_index]
            exit_callable = exit_callables[_index]
            out = out.add_ops(*call_normal_exit(exit_manager, exit_callable))
            if is_async:
                out = out._await_tos()
                out = out.add_op(opcode.Pop())
        for _ in exit_callables:
            out = out.free_synthetic()
            out = out.free_synthetic()
        return out

    def get_comprehension_code(
        self,
        elt_consumer: Callable[["Bytecode"], "Bytecode"],
        elts: tuple[ast.expr, ...],
        generators: list[ast.comprehension],
        renames: dict[str, RenamedVariable],
    ):
        end_label = Label()
        used_variables_count = 0
        previous_generator_next_label = Label()
        out, new_renames, new_variables = self.with_comprehension_opcodes(
            generators[0], previous_generator_next_label, end_label, renames
        )
        used_variables_count += new_variables
        for generator in generators[1:]:
            new_generator_next_label = Label()
            out, new_renames, new_variables = out.with_comprehension_opcodes(
                generator,
                new_generator_next_label,
                previous_generator_next_label,
                new_renames,
            )
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

    def get_assignment_code(
        self, expression: ast.expr, renames: dict[str, RenamedVariable]
    ) -> AssignmentCode:
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
                    if self.function_type is not opcode.FunctionType.CLASS_BODY:
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
                        # class body uses a mapping stored in the first synthetic
                        return AssignmentCode(
                            stack_items=0,
                            load_target=lambda b: b,
                            load_current_value=lambda b: b.add_ops(
                                opcode.LoadSynthetic(0),
                                opcode.LoadTypeAttrOrGlobal(name),
                            ),
                            store_value=lambda b: b.add_ops(
                                opcode.LoadSynthetic(0), opcode.StoreTypeAttr(name)
                            ),
                            delete_value=lambda b: b.add_ops(
                                opcode.LoadSynthetic(0), opcode.DeleteTypeAttr(name)
                            ),
                        )
                elif (
                    name in self.global_names
                    or self.function_type is not opcode.FunctionType.CLASS_BODY
                ):
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
                else:
                    return AssignmentCode(
                        stack_items=0,
                        load_target=lambda b: b,
                        load_current_value=lambda b: b.add_ops(
                            opcode.LoadSynthetic(0), opcode.LoadTypeAttrOrGlobal(name)
                        ),
                        store_value=lambda b: b.add_ops(
                            opcode.LoadSynthetic(0), opcode.StoreTypeAttr(name)
                        ),
                        delete_value=lambda b: b.add_ops(
                            opcode.LoadSynthetic(0), opcode.DeleteTypeAttr(name)
                        ),
                    )

            case ast.Attribute(attr=attr_name) as a:
                return AssignmentCode(
                    stack_items=1,
                    load_target=lambda b: b.with_expression_opcodes(a.value, renames),
                    load_current_value=lambda b: b.add_expression(
                        expression, opcode.LoadAttr(attr_name)
                    ),
                    store_value=lambda b: b.add_expression(
                        expression, opcode.StoreAttr(attr_name)
                    ),
                    delete_value=lambda b: b.add_expression(
                        expression, opcode.DeleteAttr(attr_name)
                    ),
                )

            case ast.Subscript(value, slice=slice_):
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

    def with_assignment_opcodes(
        self, expression: ast.expr, renames: dict[str, RenamedVariable]
    ) -> "Bytecode":
        assignment_code = self.get_assignment_code(expression, renames)
        return assignment_code.store_value(assignment_code.load_target(self))

    def with_statement_opcodes(
        self, statement: ast.stmt, renames: dict[str, RenamedVariable]
    ) -> "Bytecode":
        match statement:
            case ast.Pass():
                return self.add_statement(statement, opcode.Nop())

            case ast.Global(names=names):
                out = self
                for name in names:
                    out = out.add_global(name)
                return out

            case ast.Nonlocal(names=names):
                out = self
                for name in names:
                    out = out.add_cell(name)
                return out

            case ast.Expr(value=value):
                out = self.with_expression_opcodes(value, renames)
                out = out.add_op(opcode.Pop())
                return out

            case ast.Return(value=value):
                if value is not None:
                    out = self.with_expression_opcodes(value, renames)
                else:
                    out = self.with_expression_opcodes(
                        ast.Constant(None, lineno=statement.lineno), renames
                    )

                match out.function_type:
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
                        raise RuntimeError(f"Unknown function type {out.function_type}")
                if len(out.finally_blocks) == 0:
                    if is_generator_or_async_function:
                        out = out.add_ops(
                            opcode.LoadConstant(True),
                            opcode.LoadSynthetic(index=0),
                            opcode.StoreAttr(name="_generator_finished"),
                        )
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

                if is_generator_or_async_function:
                    out = out.add_ops(
                        opcode.LoadConstant(True),
                        opcode.LoadSynthetic(index=0),
                        opcode.StoreAttr(name="_generator_finished"),
                    )
                return out.add_statement(statement, opcode.ReturnValue())

            case ast.Raise(exc=exc, cause=cause):
                if exc is None:
                    return self.add_op(opcode.ReraiseLast())
                out = self.with_expression_opcodes(exc, renames)
                if cause is None:
                    return out.add_op(opcode.Raise())
                out = out.with_expression_opcodes(cause, renames)
                return out.add_op(opcode.RaiseWithCause())

            case ast.Try(
                body=body, handlers=handlers, orelse=orelse, finalbody=finalbody
            ):
                try_start = Label()
                try_end = Label()
                except_start = Label()
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

                out = out.add_exception_handler(
                    ExceptionHandler(
                        exception_class=BaseException,
                        from_label=try_start,
                        to_label=try_end,
                        handler_label=except_start,
                    )
                )

                out = out.add_label(except_start)
                out = out.new_synthetic()
                caught_exception = out.current_synthetic
                out = out.add_op(opcode.StoreSynthetic(index=caught_exception))
                next_handler = Label()
                for except_handler in handlers:
                    if except_handler.type is not None:
                        out = out.add_op(opcode.LoadSynthetic(index=caught_exception))
                        out = out.with_expression_opcodes(
                            except_handler.type, renames=renames
                        )
                        out = out.add_op(
                            opcode.JumpIfNotMatchExceptType(target=next_handler)
                        )
                    if except_handler.name is not None:
                        out = out.add_op(opcode.LoadSynthetic(index=caught_exception))
                        out = out.add_op(opcode.StoreLocal(except_handler.name))

                    for statement in except_handler.body:
                        out = out.with_statement_opcodes(statement, renames)

                    out = out.add_op(opcode.JumpTo(try_finally))
                    out = out.add_label(next_handler)
                    next_handler = Label()

                out = out.add_op(opcode.LoadSynthetic(index=caught_exception))
                out = out.free_synthetic()
                out = out.add_label(except_finally)
                out = out.pop_finally_block()
                out = out.add_op(opcode.Pop())

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

            case ast.AsyncWith(items=with_items, body=body):
                return self._with(with_items, body, is_async=True, renames=renames)

            case ast.With(items=with_items, body=body):
                return self._with(with_items, body, is_async=False, renames=renames)

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

            case ast.For(target=target, iter=iter, body=body, orelse=orelse):
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
                out = out.add_op(
                    opcode.GetNextElseJumpTo(target=iterator_exhausted_label)
                )
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

            case ast.AsyncFor(target=target, iter=iter, body=body, orelse=orelse):
                out = self.new_synthetic()
                iterator_synthetic = out.current_synthetic
                next_label = Label()
                got_next_label = Label()
                iterator_exhausted_label = Label()
                break_label = Label()
                out = out.with_expression_opcodes(iter, renames)
                out = out.add_op(opcode.GetAsyncIterator())
                out = out.add_op(opcode.StoreSynthetic(iterator_synthetic))
                out = out.add_label(next_label)
                out = out.add_op(opcode.LoadSynthetic(iterator_synthetic))
                out = out.add_op(opcode.GetAsyncNext())

                await_start_label = Label()
                out = out.add_label(await_start_label)
                out = out._await_tos()
                await_end_label = Label()

                out = out.add_label(await_end_label)
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

                iteration_done_label = Label()
                out = out.add_label(iteration_done_label)
                out = out.add_exception_handler(
                    ExceptionHandler(
                        exception_class=StopAsyncIteration,
                        from_label=await_start_label,
                        to_label=await_end_label,
                        handler_label=iteration_done_label,
                    )
                )
                out = out.add_op(opcode.Pop())
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
                    out = out.with_expression_opcodes(decorator, renames)
                    out = out.add_ops(
                        opcode.CreateCallBuilder(),
                        opcode.Swap(),
                        opcode.WithPositionalArg(index=0),
                        opcode.CallWithBuilder(),
                    )
                out = out.with_assignment_opcodes(
                    ast.Name(id=func_def.name, ctx=ast.Store), renames
                )
                return out

            case ast.ClassDef(
                name=name,
                bases=bases,
                keywords=keywords,
                body=body,
                decorator_list=decorators,
            ):
                body_bytecode = self._create_inner_class_bytecode(statement)
                out = self
                out = out.add_op(opcode.NewList())

                for base in bases:
                    out = out.with_expression_opcodes(base, renames)
                    out = out.add_op(opcode.ListAppend())
                else:
                    out = out.add_op(opcode.ListToTuple())

                out = out.add_op(opcode.NewDict())
                for keyword in keywords:
                    out = out.add_op(opcode.LoadConstant(keyword.arg))
                    out = out.with_expression_opcodes(keyword.value, renames)
                    out = out.add_op(opcode.DictPut())

                out = out.add_op(
                    opcode.LoadAndBindInnerClass(
                        class_name=name, class_body=body_bytecode
                    )
                )

                for decorator in decorators:
                    out = out.with_expression_opcodes(decorator, renames)
                    out = out.add_ops(
                        opcode.CreateCallBuilder(),
                        opcode.Swap(),
                        opcode.WithPositionalArg(index=0),
                        opcode.CallWithBuilder(),
                    )

                out = out.with_assignment_opcodes(
                    ast.Name(id=name, ctx=ast.Store), renames
                )
                return out

            case ast.Import(names=aliases):
                out = self
                for alias in aliases:
                    out = out.add_statement(
                        statement,
                        opcode.ImportModule(name=alias.name, level=0, from_list=()),
                    )
                    asname = alias.name.split(".")[0]
                    if alias.asname is not None:
                        asname = alias.asname
                    out = out.with_assignment_opcodes(
                        ast.Name(id=asname, ctx=ast.Store), renames
                    )
                return out

            case ast.ImportFrom(module=module, names=aliases, level=level):
                out = self
                out = out.add_statement(
                    statement,
                    opcode.ImportModule(
                        name=module,
                        level=level,
                        from_list=tuple(alias.name for alias in aliases),
                    ),
                )
                for alias in aliases:
                    asname = alias.asname if alias.asname is not None else alias.name
                    out = out.add_ops(
                        opcode.Dup(),
                        opcode.LoadAttr(
                            name=alias.name,
                        ),
                    )
                    out = out.with_assignment_opcodes(
                        ast.Name(id=asname, ctx=ast.Store), renames
                    )
                out = out.add_op(opcode.Pop())
                return out

            case ast.Match(subject=subject, cases=cases):
                out = self.with_expression_opcodes(subject, renames)
                next_case_label = Label()
                match_end_label = Label()
                for case in cases:
                    out = out.with_match_opcodes(
                        case, next_case_label, match_end_label, renames
                    )
                    out = out.add_label(next_case_label)
                    next_case_label = Label()
                out = out.add_op(opcode.Pop())
                out = out.add_label(match_end_label)
                return out

            case _:
                raise NotImplementedError(
                    f"Not implemented statement: {type(statement)}"
                )

        raise RuntimeError(
            f"Missing return in case {type(statement)} in with_statement_opcodes"
        )

    def with_match_opcodes(
        self,
        match_case: ast.match_case,
        next_case_label: Label,
        match_end_label: Label,
        renames: dict[str, RenamedVariable],
    ) -> "Bytecode":
        out = self.with_match_pattern_opcodes(
            match_case.pattern, True, next_case_label, renames
        )
        if match_case.guard is not None:
            out = out.with_expression_opcodes(match_case.guard, renames)
            out = out.add_op(opcode.IfFalse(target=next_case_label))
        for statement in match_case.body:
            out = out.with_statement_opcodes(statement, renames)
        out = out.add_op(opcode.JumpTo(match_end_label))
        return out

    def with_match_pattern_opcodes(
        self,
        pattern: ast.pattern,
        should_pop: bool,
        next_case_label: Label,
        renames: dict[str, RenamedVariable],
    ):
        match pattern:
            case ast.MatchValue(value=value):
                out = self.add_ops(
                    opcode.Dup(),
                    opcode.LoadConstant(constant=self._evaluate_constant_expr(value)),
                    opcode.BinaryOp(operator=opcode.BinaryOperator.Eq),
                    opcode.IfFalse(target=next_case_label),
                )
                if should_pop:
                    out = out.add_op(opcode.Pop())
                return out
            case ast.MatchSingleton(value=value):
                out = self.add_ops(
                    opcode.Dup(),
                    opcode.LoadConstant(constant=value),
                    opcode.IsSameAs(negate=False),
                    opcode.IfFalse(target=next_case_label),
                )
                if should_pop:
                    out = out.add_op(opcode.Pop())
                return out
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
                out = self.add_ops(
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
                )
                element_values = []
                for i in range(len(subpatterns)):
                    out = out.new_synthetic()
                    element_values.append(out.current_synthetic)
                    out = out.add_op(opcode.StoreSynthetic(index=element_values[-1]))

                match_failed_label = Label()
                match_success_label = Label()
                for i in range(len(subpatterns)):
                    out = out.add_op(opcode.LoadSynthetic(index=element_values[i]))
                    out = out.with_match_pattern_opcodes(
                        subpatterns[i], True, match_failed_label, renames
                    )

                if should_pop:
                    out = out.add_op(opcode.Pop())

                out = out.add_op(opcode.JumpTo(target=match_success_label))
                out = out.add_label(match_failed_label)
                out = out.add_ops(opcode.Pop(), opcode.JumpTo(target=next_case_label))
                out = out.add_label(match_success_label)

                for i in range(len(subpatterns)):
                    out = out.free_synthetic()
                return out
            case ast.MatchMapping(keys=keys, patterns=subpatterns, rest=extras_name):
                has_extras = extras_name is not None
                # keys must be literals, so we can evaluate them
                const_keys = tuple(self._evaluate_constant_expr(key) for key in keys)
                out = self.add_ops(
                    opcode.MatchMapping(
                        keys=const_keys,
                        target=next_case_label,
                    ),
                    opcode.Dup(),
                    opcode.UnpackMapping(
                        keys=const_keys,
                        has_extras=has_extras,
                    ),
                )

                if has_extras:
                    out = out.with_assignment_opcodes(
                        ast.Name(id=extras_name, ctx=ast.Store()), renames
                    )

                key_values = []
                match_failed_label = Label()
                match_success_label = Label()
                for _ in subpatterns:
                    out = out.new_synthetic()
                    key_values.append(out.current_synthetic)
                    out = out.add_op(opcode.StoreSynthetic(index=key_values[-1]))

                for index, subpattern in enumerate(subpatterns):
                    out = out.add_op(opcode.LoadSynthetic(index=key_values[index]))
                    out = out.with_match_pattern_opcodes(
                        subpattern, True, match_failed_label, renames
                    )

                for _ in subpatterns:
                    out = out.free_synthetic()

                if should_pop:
                    out = out.add_op(opcode.Pop())
                out = out.add_op(opcode.JumpTo(target=match_success_label))
                out = out.add_label(match_failed_label)
                out = out.add_ops(opcode.Pop(), opcode.JumpTo(target=next_case_label))
                out = out.add_label(match_success_label)
                return out
            case ast.MatchClass(
                cls=matched_class,
                patterns=subpatterns,
                kwd_patterns=kwd_subpatterns,
                kwd_attrs=kwd_attrs,
            ):
                out = self.with_expression_opcodes(matched_class, renames)
                out = out.add_op(
                    opcode.MatchClass(
                        target=next_case_label,
                        attributes=tuple(kwd_attrs),
                        positional_count=len(subpatterns),
                    )
                )
                attribute_vars = {}
                positional_vars = {}

                # Store the values on the stack in reverse order
                for kwd_attr in reversed(kwd_attrs):
                    out = out.new_synthetic()
                    attribute_vars[kwd_attr] = out.current_synthetic
                    out = out.add_op(opcode.StoreSynthetic(index=out.current_synthetic))
                for index, _ in enumerate(subpatterns):
                    out = out.new_synthetic()
                    positional_vars[len(subpatterns) - index - 1] = (
                        out.current_synthetic
                    )
                    out = out.add_op(opcode.StoreSynthetic(index=out.current_synthetic))

                match_success_label = Label()
                match_failed_label = Label()

                for index, subpattern in enumerate(subpatterns):
                    out = out.add_op(opcode.LoadSynthetic(index=positional_vars[index]))
                    out = out.with_match_pattern_opcodes(
                        subpattern, True, match_failed_label, renames
                    )

                for index, subpattern in enumerate(kwd_subpatterns):
                    out = out.add_op(
                        opcode.LoadSynthetic(index=attribute_vars[kwd_attrs[index]])
                    )
                    out = out.with_match_pattern_opcodes(
                        subpattern, True, match_failed_label, renames
                    )

                out = out.add_op(opcode.JumpTo(target=match_success_label))
                out = out.add_label(match_failed_label)
                out = out.add_ops(opcode.Pop(), opcode.JumpTo(target=next_case_label))
                out = out.add_label(match_success_label)
                for _ in kwd_attrs:
                    out = out.free_synthetic()
                for _ in subpatterns:
                    out = out.free_synthetic()
                return out
            case ast.MatchStar(name=name):
                out = self
                if name is not None:
                    out = out.with_assignment_opcodes(
                        ast.Name(id=name, ctx=ast.Store()), renames
                    )
                else:
                    out = out.add_op(opcode.Pop())
                return out
            case ast.MatchAs(pattern=subpattern, name=name):
                out = self
                if subpattern is not None:
                    out = out.with_match_pattern_opcodes(
                        subpattern, False, next_case_label, renames
                    )
                if name is not None:
                    # TODO: Delay assignment until guard evaluation
                    out = out.with_assignment_opcodes(
                        ast.Name(id=name, ctx=ast.Store()), renames
                    )
                else:
                    out = out.add_op(opcode.Pop())
                return out
            case ast.MatchOr(patterns=subpatterns):
                out = self
                next_pattern_label = Label()
                pattern_end_label = Label()
                for subpattern in subpatterns:
                    out = out.with_match_pattern_opcodes(
                        subpattern, should_pop, next_pattern_label, renames
                    )
                    out = out.add_op(opcode.JumpTo(pattern_end_label))
                    out = out.add_label(next_pattern_label)
                    next_pattern_label = Label()
                out = out.add_label(next_pattern_label)
                out = out.add_op(opcode.JumpTo(next_case_label))
                out = out.add_label(pattern_end_label)
                return out
            case _:
                raise NotImplementedError(f"Not implemented pattern: {type(pattern)}")
        raise RuntimeError(
            f"Missing return in case {type(pattern)} in with_match_pattern_opcodes"
        )

    def _create_annotation_dict(
        self,
        all_args: dict[str, ast.expr],
        value_converter: Callable[["Bytecode", ast.expr], "Bytecode"],
    ) -> "Bytecode":
        annotate_function = self
        annotate_function = annotate_function.add_op(opcode.NewDict())
        for arg, annotation in all_args.items():
            annotate_function = annotate_function.add_op(opcode.LoadConstant(arg))
            annotate_function = value_converter(annotate_function, annotation)
            annotate_function = annotate_function.add_op(opcode.DictPut())

        annotate_function = annotate_function.add_op(opcode.ReturnValue())
        return annotate_function

    def _create_annotation_function(
        self,
        used_variables: "UsedVariables",
        free_name_set,
        func_def: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    ):
        annotate_function = Bytecode(
            function_name="__annotate__",
            function_type=opcode.FunctionType.FUNCTION,
            method_type=opcode.MethodType.VIRTUAL,
            generator_instance_parameter="_generator",
            signature=inspect.Signature(
                parameters=(
                    opcode.Parameter(
                        name="format", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD
                    ),
                )
            ),
            local_names=frozenset(),
            cell_names=frozenset(used_variables.cell_names),
            free_names=frozenset(free_name_set),
            global_names=self.global_names,
            globals=self.globals,
            closure=self.closure,
        )

        arg_info = _get_arg_dict(func_def)
        check_format_forward_ref = opcode.Label()
        check_format_source = opcode.Label()
        fail_label = opcode.Label()
        annotate_function = annotate_function.add_ops(
            opcode.LoadLocal(name="format"),
            opcode.LoadConstant(1),  # value
            opcode.BinaryOp(operator=opcode.BinaryOperator.Eq),
            opcode.IfFalse(target=check_format_forward_ref),
        )

        annotate_function = annotate_function._create_annotation_dict(
            arg_info, lambda b, e: b.with_expression_opcodes(e, {})
        )

        annotate_function = annotate_function.add_label(check_format_forward_ref)
        annotate_function = annotate_function.add_ops(
            opcode.LoadLocal(name="format"),
            opcode.LoadConstant(2),  # forward_ref
            opcode.BinaryOp(operator=opcode.BinaryOperator.Eq),
            opcode.IfFalse(target=check_format_source),
        )
        annotate_function = annotate_function.add_ops(
            opcode.LoadConstant(NotImplementedError), opcode.Raise()
        )

        annotate_function = annotate_function.add_label(check_format_source)
        annotate_function = annotate_function.add_ops(
            opcode.LoadLocal(name="format"),
            opcode.LoadConstant(3),  # source
            opcode.BinaryOp(operator=opcode.BinaryOperator.Eq),
            opcode.IfFalse(target=fail_label),
        )
        annotate_function = annotate_function.add_ops(
            opcode.LoadConstant(NotImplementedError), opcode.Raise()
        )

        annotate_function = annotate_function.add_label(fail_label)
        annotate_function = annotate_function.add_ops(
            opcode.LoadConstant(NotImplementedError), opcode.Raise()
        )
        return annotate_function

    def _create_inner_function(
        self, func_def: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> InnerFunction:
        signature_and_parameters = get_signature_from_arguments(func_def.args)
        used_variables = find_used_variables(
            func_def, self.local_names | self.cell_names | self.free_names
        )
        free_name_set = self.cell_names & used_variables.variable_names
        closure = (CellType(),) * len(free_name_set)

        inner_bytecode = ast_to_bytecode(
            func_def,
            signature_and_parameters.signature,
            self.globals,
            closure,
            tuple(free_name_set),
            tuple(used_variables.cell_names),
            tuple(used_variables.variable_names),
        )

        return InnerFunction(
            bytecode=inner_bytecode,
            annotate_function=self._create_annotation_function(
                used_variables, free_name_set, func_def
            ),
            parameters_with_defaults=tuple(
                signature_and_parameters.parameters_with_defaults
            ),
        )

    def _create_inner_class_bytecode(self, class_def: ast.ClassDef) -> Bytecode:
        signature = inspect.Signature(
            parameters=(
                inspect.Parameter(
                    "_attribute_mapping",
                    kind=inspect.Parameter.POSITIONAL_ONLY,
                ),
            )
        )
        used_variables = find_used_variables(
            class_def, self.local_names | self.cell_names | self.free_names
        )
        free_name_set = self.cell_names & used_variables.variable_names
        closure = (CellType(),) * len(free_name_set)

        inner_bytecode = ast_to_bytecode(
            class_def,
            signature,
            self.globals,
            closure,
            tuple(free_name_set),
            tuple(),
            tuple(used_variables.variable_names),
        )
        return inner_bytecode

    def _create_lambda(self, lambda_def: ast.Lambda) -> InnerFunction:
        signature_and_parameters = get_signature_from_arguments(lambda_def.args)
        used_variables = find_used_variables(
            lambda_def, self.local_names | self.cell_names | self.free_names
        )
        free_name_set = self.cell_names & used_variables.variable_names
        closure = (CellType(),) * len(free_name_set)
        inner_bytecode = lambda_ast_to_bytecode(
            lambda_def,
            signature_and_parameters.signature,
            self.globals,
            closure,
            tuple(free_name_set),
            tuple(used_variables.cell_names),
            tuple(used_variables.variable_names),
        )
        return InnerFunction(
            bytecode=inner_bytecode,
            annotate_function=self._create_annotation_function(
                used_variables, free_name_set, lambda_def
            ),
            parameters_with_defaults=signature_and_parameters.parameters_with_defaults,
        )


@cached_property
def annotations(self):
    from ._vm import CDisVM

    if self.annotate_function is None:
        out = {
            key: value.annotation
            for key, value in self.signature.parameters.items()
            if value.annotation is not inspect.Parameter.empty
        }
        if self.signature.return_annotation is not inspect.Parameter.empty:
            return out | {"return": self.signature.return_annotation}
        else:
            return out

    vm = CDisVM()
    return vm.run(self.annotate_function, 1)


Bytecode.__annotations__ = annotations
annotations.__set_name__(Bytecode, "__annotations__")


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
    func_def: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> dict[str, ast.expr]:
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


def unindent(code: str) -> str:
    lowest_indent = float("inf")
    for line in code.splitlines():
        indent = len(line) - len(line.lstrip())
        if indent == 0 and len(line) > 0:
            return code
        elif indent != 0:
            lowest_indent = min(lowest_indent, indent)

    out = ""
    for line in code.splitlines():
        if len(line) == 0:
            out += line
        else:
            out += line[lowest_indent:] + "\n"

    return out


def to_bytecode(function: FunctionType) -> Bytecode | opcode.ClassInfo:
    source = inspect.getsource(function)
    source = unindent(source)
    function_ast: ast.FunctionDef = cast(ast.FunctionDef, ast.parse(source).body[0])
    used_vars = find_used_variables(function_ast, frozenset())

    return ast_to_bytecode(
        function_ast,
        inspect.signature(
            function,
            globals=function.__globals__,
            locals=function.__globals__,
            eval_str=True,
        ),
        function.__globals__,
        function.__closure__,
        function.__code__.co_freevars,
        # TODO: Remove __code__ usage
        tuple(
            frozenset(used_vars.cell_names) | frozenset(function.__code__.co_cellvars)
        ),
        tuple(
            frozenset(used_vars.variable_names)
            | frozenset(function.__code__.co_varnames)
        ),
    )


def is_generator_or_async(function_ast: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if isinstance(function_ast, ast.AsyncFunctionDef):
        return True
    return is_generator(function_ast)


def is_generator(function_ast: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    generator_opcodes = ast.Yield, ast.YieldFrom
    for node in ast.walk(function_ast):
        if isinstance(node, generator_opcodes):
            return True
    return False


def _class_body_prologue(bytecode: Bytecode) -> Bytecode:
    # Make first synthetic store the generator object
    bytecode = bytecode.new_synthetic()
    bytecode = bytecode.add_ops(
        opcode.LoadLocal(name="_attribute_mapping"),
        opcode.StoreSynthetic(index=0),
    )
    return bytecode


def _generator_function_prologue(bytecode: Bytecode) -> Bytecode:
    # Make first synthetic store the generator object
    bytecode = bytecode.new_synthetic()
    bytecode = bytecode.add_ops(
        opcode.LoadLocal(name="_generator_instance"),
        opcode.StoreSynthetic(index=0),
    )
    # First opcode is to jump to the corresponding yield
    yield_index = bytecode.current_yield
    yield_label = Label()
    bytecode = bytecode.add_ops(opcode.JumpTo(target=yield_label))
    bytecode = bytecode.add_yield_label(yield_label)
    bytecode = bytecode.add_ops(
        opcode.LoadSynthetic(index=0),
        opcode.DelegateOrRestoreGeneratorState(
            state_id=yield_index, stack_metadata=cast(StackMetadata, None)
        ),
        opcode.Pop(),
    )

    def _deferred_processor(out_bytecode: Bytecode) -> Bytecode:
        out_bytecode = replace(
            out_bytecode,
            instructions=(
                *out_bytecode.instructions[:4],
                Instruction(
                    bytecode_index=4,
                    opcode=opcode.DelegateOrRestoreGeneratorState(
                        state_id=yield_index,
                        stack_metadata=out_bytecode.stack_metadata[2],
                    ),
                    lineno=out_bytecode.instructions[1].lineno,
                ),
                *out_bytecode.instructions[5:],
            ),
        )
        return out_bytecode

    bytecode = bytecode.add_deferred_processor(_deferred_processor)
    return bytecode


def ast_to_bytecode(
    function_ast: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    signature: inspect.Signature,
    function_globals: dict[str, object],
    function_closure: tuple[CellType, ...],
    free_names: tuple[str, ...],
    cell_names: tuple[str, ...],
    variable_names: tuple[str, ...],
) -> Bytecode | opcode.ClassInfo:
    free_vars = free_names
    cell_names = frozenset(cell_names + free_vars)
    local_names = frozenset(set(variable_names) - set(cell_names))
    is_generator_or_async_function = is_generator_or_async(function_ast)
    if is_generator_or_async_function:
        local_names |= frozenset({"_generator_instance"})

    closure_dict = {}
    for i in range(len(free_vars)):
        closure_dict[free_vars[i]] = function_closure[i]

    bytecode = Bytecode(
        function_name=function_ast.name,
        function_type=opcode.FunctionType.for_function_ast(function_ast),
        method_type=opcode.MethodType.for_function_ast(function_ast),
        generator_instance_parameter="_generator_instance"
        if is_generator_or_async_function
        else None,
        signature=signature,
        local_names=local_names,
        cell_names=cell_names,
        free_names=frozenset(free_vars),
        globals=function_globals,
        closure=closure_dict,
    )
    is_class_body = bytecode.function_type is opcode.FunctionType.CLASS_BODY
    if is_generator_or_async_function:
        bytecode = _generator_function_prologue(bytecode)
    elif is_class_body:
        bytecode = _class_body_prologue(bytecode)

    parameter_cells = (cell_names - frozenset(free_names)) & {
        name for name in variable_names if name in signature.parameters.keys()
    }

    for parameter_cell_var in parameter_cells:
        if not is_generator_or_async_function:
            bytecode = bytecode.add_op(opcode.LoadLocal(name=parameter_cell_var))
            bytecode = bytecode.add_op(
                opcode.StoreCell(name=parameter_cell_var, is_free=False)
            )
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
        ),
        {},
    )

    bytecode = bytecode.process_deferred()

    if not is_generator_or_async_function:
        return bytecode

    return _create_generator_class_from_bytecode(bytecode, parameter_cells)


def _create_generator_class_from_bytecode(
    bytecode: Bytecode, parameter_cells: frozenset[str]
) -> opcode.ClassInfo:
    # Create a class object that jumps to each yield label
    generator_class_attributes: dict[str, Bytecode] = dict()
    for yield_index, yield_label in enumerate(bytecode.yield_labels):
        method_bytecode = replace(
            bytecode,
            function_type=opcode.MethodType.VIRTUAL,
            # TODO: Generate a unique name incase '_generator_instance' was used already
            generator_instance_parameter="_generator_instance",
            signature=inspect.Signature(
                [
                    inspect.Parameter(
                        name="_generator_instance",
                        kind=inspect.Parameter.POSITIONAL_ONLY,
                    )
                ]
            ),
            instructions=(
                *bytecode.instructions[:2],
                Instruction(
                    bytecode_index=2,
                    lineno=bytecode.instructions[0].lineno,
                    opcode=opcode.JumpTo(target=yield_label),
                ),
                *bytecode.instructions[3:],
            ),
        )
        generator_class_attributes[f"_next_{yield_index}"] = method_bytecode
    init_bytecode = Bytecode(
        function_name="__init__",
        function_type=opcode.FunctionType.FUNCTION,
        method_type=opcode.MethodType.VIRTUAL,
        generator_instance_parameter="_generator_instance",
        signature=inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name="_generator_instance", kind=inspect.Parameter.POSITIONAL_ONLY
                ),
                *bytecode.signature.parameters.values(),
            ]
        ),
        instructions=(),
        local_names=bytecode.local_names,
        free_names=bytecode.free_names,
        cell_names=bytecode.cell_names,
        globals=bytecode.globals,
        closure=bytecode.closure,
    )
    for synthetic in range(bytecode.synthetic_count):
        init_bytecode = init_bytecode.new_synthetic()
    for parameter_cell_var in parameter_cells:
        init_bytecode = init_bytecode.add_op(opcode.LoadLocal(name=parameter_cell_var))
        init_bytecode = init_bytecode.add_op(
            opcode.StoreCell(name=parameter_cell_var, is_free=False)
        )
    init_bytecode = init_bytecode.add_op(opcode.LoadLocal(name="_generator_instance"))
    init_bytecode = init_bytecode.add_op(opcode.StoreSynthetic(index=0))
    # Need a value on the stack to match expected stack state for a yield
    init_bytecode = init_bytecode.add_op(opcode.LoadConstant(None))
    init_bytecode = init_bytecode.add_op(opcode.LoadSynthetic(index=0))
    init_bytecode = init_bytecode.add_op(
        opcode.SaveGeneratorState(
            state_id=0,
            stack_metadata=bytecode.stack_metadata[2],
        )
    )
    init_bytecode = init_bytecode.add_op(opcode.Pop())
    init_bytecode = init_bytecode.add_ops(
        opcode.LoadConstant(None),
        opcode.LoadLocal(name="_generator_instance"),
        opcode.StoreAttr(name="_sent_value"),
        opcode.LoadConstant(None),
        opcode.LoadLocal(name="_generator_instance"),
        opcode.StoreAttr(name="_thrown_value"),
        opcode.LoadConstant(False),
        opcode.LoadLocal(name="_generator_instance"),
        opcode.StoreAttr(name="_generator_finished"),
        opcode.LoadConstant(opcode.GeneratorOperation.NEXT.value),
        opcode.LoadLocal(name="_generator_instance"),
        opcode.StoreAttr(name="_operation"),
        opcode.LoadConstant(None),
        opcode.LoadLocal(name="_generator_instance"),
        opcode.StoreAttr(name="_sub_generator"),
    )
    init_bytecode = init_bytecode.add_op(opcode.LoadConstant(None))
    init_bytecode = init_bytecode.add_op(opcode.ReturnValue())
    generator_class_attributes["__init__"] = init_bytecode
    iter_bytecode = Bytecode(
        function_name="__iter__",
        function_type=opcode.FunctionType.FUNCTION,
        method_type=opcode.MethodType.VIRTUAL,
        generator_instance_parameter="_generator_instance",
        signature=inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name="_generator_instance", kind=inspect.Parameter.POSITIONAL_ONLY
                )
            ]
        ),
        instructions=(),
        local_names=frozenset({"_generator_instance"}),
        free_names=frozenset(),
        cell_names=frozenset(),
        globals=bytecode.globals,
        closure=bytecode.closure,
    )
    iter_bytecode = iter_bytecode.add_ops(
        opcode.LoadLocal(name="_generator_instance"), opcode.ReturnValue()
    )

    match bytecode.function_type:
        case opcode.FunctionType.GENERATOR:
            generator_class_attributes["__iter__"] = iter_bytecode
        case opcode.FunctionType.ASYNC_GENERATOR:
            generator_class_attributes["__aiter__"] = iter_bytecode
        case (
            opcode.FunctionType.COROUTINE_GENERATOR
            | opcode.FunctionType.ASYNC_FUNCTION
        ):
            #  Note: technically await should return a different object,
            #        but you cannot reuse an already run-coroutine
            generator_class_attributes["__iter__"] = iter_bytecode
            generator_class_attributes["__await__"] = iter_bytecode
        case _:
            raise RuntimeError(f"Unknown generator type {bytecode.function_type}")

    next_bytecode = Bytecode(
        function_name="__next__",
        function_type=opcode.FunctionType.FUNCTION,
        method_type=opcode.MethodType.VIRTUAL,
        generator_instance_parameter="_generator_instance",
        signature=inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name="_generator_instance", kind=inspect.Parameter.POSITIONAL_ONLY
                )
            ]
        ),
        instructions=(),
        local_names=frozenset({"_generator_instance", "state_id", "return_value"}),
        free_names=frozenset(),
        cell_names=frozenset(),
        globals=bytecode.globals,
        closure=bytecode.closure,
    )
    generator_not_finished_label = Label()
    next_bytecode = next_bytecode.add_ops(
        opcode.LoadLocal(name="_generator_instance"),
        opcode.LoadAttr(name="_generator_finished"),
        opcode.IfFalse(target=generator_not_finished_label),
        opcode.LoadConstant(StopIteration),
        opcode.CreateCallBuilder(),
        opcode.CallWithBuilder(),
        opcode.Raise(),
    )
    next_bytecode = next_bytecode.add_label(generator_not_finished_label)
    next_exception_handler_label = Label()
    next_bytecode = next_bytecode.add_exception_handler(
        ExceptionHandler(
            exception_class=BaseException,
            from_label=generator_not_finished_label,
            to_label=next_exception_handler_label,
            handler_label=next_exception_handler_label,
        )
    )
    next_bytecode = next_bytecode.add_ops(
        opcode.LoadLocal(name="_generator_instance"),
        opcode.LoadAttr(name="_state_id"),
        opcode.StoreLocal(name="state_id"),
    )
    # Note: a tuple lookup might be faster
    generator_finished_label = opcode.Label()
    for yield_index, yield_label in enumerate(bytecode.yield_labels):
        skip_label = opcode.Label()
        next_bytecode = next_bytecode.add_ops(
            opcode.LoadConstant(yield_index),
            opcode.LoadLocal(name="state_id"),
            opcode.BinaryOp(operator=opcode.BinaryOperator.Eq),
            opcode.IfFalse(target=skip_label),
            opcode.LoadLocal(name="_generator_instance"),
            opcode.LoadAttr(name=f"_next_{yield_index}"),
            opcode.CreateCallBuilder(),
            opcode.CallWithBuilder(),
            opcode.StoreLocal(name="return_value"),
            opcode.LoadLocal(name="_generator_instance"),
            opcode.LoadAttr(name="_generator_finished"),
            opcode.IfTrue(target=generator_finished_label),
            opcode.LoadLocal(name="return_value"),
            opcode.ReturnValue(),
        )
        next_bytecode = next_bytecode.add_label(skip_label)
    next_bytecode = next_bytecode.add_ops(
        opcode.LoadConstant(SystemError),
        opcode.CreateCallBuilder(),
        opcode.LoadConstant("Generated next method missing branch for state "),
        opcode.LoadLocal(name="state_id"),
        opcode.FormatValue(
            conversion=opcode.FormatConversion.TO_STRING, format_spec=""
        ),
        opcode.BinaryOp(operator=opcode.BinaryOperator.Add),
        opcode.WithPositionalArg(index=0),
        opcode.CallWithBuilder(),
        opcode.Raise(),
    )
    next_bytecode = next_bytecode.add_label(generator_finished_label)
    stop_iteration_has_value_label = Label()
    next_bytecode = next_bytecode.add_ops(
        opcode.LoadConstant(True),
        opcode.LoadLocal(name="_generator_instance"),
        opcode.StoreAttr(name="_generator_finished"),
        opcode.LoadLocal(name="return_value"),
        opcode.LoadConstant(None),
        opcode.IsSameAs(negate=False),
        opcode.IfFalse(target=stop_iteration_has_value_label),
        opcode.LoadConstant(StopIteration),
        opcode.CreateCallBuilder(),
        opcode.CallWithBuilder(),
        opcode.Raise(),
    )
    next_bytecode = next_bytecode.add_label(stop_iteration_has_value_label)
    next_bytecode = next_bytecode.add_ops(
        opcode.LoadConstant(StopIteration),
        opcode.CreateCallBuilder(),
        opcode.LoadLocal(name="return_value"),
        opcode.WithPositionalArg(index=0),
        opcode.CallWithBuilder(),
        opcode.Raise(),
    )
    next_bytecode = next_bytecode.add_label(next_exception_handler_label)
    next_bytecode = next_bytecode.add_ops(
        opcode.LoadConstant(True),
        opcode.LoadLocal(name="_generator_instance"),
        opcode.StoreAttr(name="_generator_finished"),
        opcode.ReraiseLast(),
    )

    match bytecode.function_type:
        case (
            opcode.FunctionType.GENERATOR
            | opcode.FunctionType.ASYNC_FUNCTION
            | opcode.FunctionType.COROUTINE_GENERATOR
        ):
            generator_class_attributes["__next__"] = next_bytecode
        case opcode.FunctionType.ASYNC_GENERATOR:
            generator_class_attributes["__anext__"] = next_bytecode
        case _:
            raise RuntimeError(f"Unknown generator type {bytecode.function_type}")

    send_bytecode = Bytecode(
        function_name="send",
        function_type=opcode.FunctionType.FUNCTION,
        method_type=opcode.MethodType.VIRTUAL,
        generator_instance_parameter="_generator_instance",
        signature=inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name="_generator_instance", kind=inspect.Parameter.POSITIONAL_ONLY
                ),
                inspect.Parameter(name="value", kind=inspect.Parameter.POSITIONAL_ONLY),
            ]
        ),
        instructions=(),
        local_names=frozenset({"_generator_instance"}),
        free_names=frozenset(),
        cell_names=frozenset(),
        globals=bytecode.globals,
        closure=bytecode.closure,
    )
    send_bytecode = send_bytecode.add_ops(
        opcode.LoadLocal(name="value"),
        opcode.LoadLocal(name="_generator_instance"),
        opcode.StoreAttr(name="_sent_value"),
        opcode.LoadConstant(opcode.GeneratorOperation.SEND.value),
        opcode.LoadLocal(name="_generator_instance"),
        opcode.StoreAttr(name="_operation"),
        opcode.LoadLocal(name="_generator_instance"),
        opcode.LoadAttr(name="__next__"),
        opcode.CreateCallBuilder(),
        opcode.CallWithBuilder(),
        opcode.ReturnValue(),
    )

    match bytecode.function_type:
        case (
            opcode.FunctionType.GENERATOR
            | opcode.FunctionType.ASYNC_FUNCTION
            | opcode.FunctionType.COROUTINE_GENERATOR
        ):
            generator_class_attributes["send"] = send_bytecode
        case opcode.FunctionType.ASYNC_GENERATOR:
            generator_class_attributes["asend"] = send_bytecode
        case _:
            raise RuntimeError(f"Unknown generator type {bytecode.function_type}")

    throw_bytecode = Bytecode(
        function_name="throw",
        function_type=opcode.FunctionType.FUNCTION,
        method_type=opcode.MethodType.VIRTUAL,
        generator_instance_parameter="_generator_instance",
        signature=inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name="_generator_instance", kind=inspect.Parameter.POSITIONAL_ONLY
                ),
                inspect.Parameter(name="error", kind=inspect.Parameter.POSITIONAL_ONLY),
            ]
        ),
        instructions=(),
        local_names=frozenset({"_generator_instance"}),
        free_names=frozenset(),
        cell_names=frozenset(),
        globals=bytecode.globals,
        closure=bytecode.closure,
    )
    throw_bytecode = throw_bytecode.add_ops(
        opcode.LoadLocal(name="error"),
        opcode.LoadLocal(name="_generator_instance"),
        opcode.StoreAttr(name="_thrown_value"),
        opcode.LoadConstant(opcode.GeneratorOperation.THROW.value),
        opcode.LoadLocal(name="_generator_instance"),
        opcode.StoreAttr(name="_operation"),
        opcode.LoadLocal(name="_generator_instance"),
        opcode.LoadAttr(name="__next__"),
        opcode.CreateCallBuilder(),
        opcode.CallWithBuilder(),
        opcode.ReturnValue(),
    )

    match bytecode.function_type:
        case (
            opcode.FunctionType.GENERATOR
            | opcode.FunctionType.ASYNC_FUNCTION
            | opcode.FunctionType.COROUTINE_GENERATOR
        ):
            generator_class_attributes["throw"] = throw_bytecode
        case opcode.FunctionType.ASYNC_GENERATOR:
            generator_class_attributes["athrow"] = throw_bytecode
        case _:
            raise RuntimeError(f"Unknown generator type {bytecode.function_type}")

    return opcode.ClassInfo(
        name=f"<generator {bytecode.function_name}>",
        qualname=f"<generator {bytecode.function_name}>",
        class_attributes={
            name: types.FunctionType for name in generator_class_attributes.keys()
        },
        instance_attributes={
            "_state_id": int,
            "_saved_state": object,
            "_sent_value": object,
            "_thrown_value": BaseException | None,
            "_sub_generator": Iterable | None,
            "_operation": int,
            "_generator_finished": bool,
        },
        class_attribute_defaults=generator_class_attributes,
    )


def lambda_ast_to_bytecode(
    lambda_ast: ast.Lambda,
    signature: inspect.Signature,
    function_globals: dict[str, object],
    function_closure: tuple[CellType, ...],
    free_names: tuple[str, ...],
    cell_names: tuple[str, ...],
    variable_names: tuple[str, ...],
) -> Bytecode:
    free_vars = free_names
    cell_names = frozenset(cell_names + free_vars)
    local_names = frozenset(set(variable_names) - set(cell_names))

    closure_dict = {}
    for i in range(len(free_vars)):
        closure_dict[free_vars[i]] = function_closure[i]

    bytecode = Bytecode(
        function_name="lambda",
        function_type=opcode.FunctionType.FUNCTION,
        method_type=opcode.MethodType.VIRTUAL,
        generator_instance_parameter=None,
        signature=signature,
        local_names=local_names,
        cell_names=cell_names,
        free_names=frozenset(free_vars),
        globals=function_globals,
        closure=closure_dict,
    )

    parameter_cells = (cell_names - frozenset(free_names)) & {
        name for name in variable_names if name in signature.parameters.keys()
    }
    for parameter_cell_var in parameter_cells:
        bytecode = bytecode.add_op(opcode.LoadLocal(name=parameter_cell_var))
        bytecode = bytecode.add_op(
            opcode.StoreCell(name=parameter_cell_var, is_free=False)
        )
        closure_dict[parameter_cell_var] = CellType()

    bytecode = bytecode.with_statement_opcodes(
        ast.Return(
            lambda_ast.body,
            lineno=lambda_ast.body.lineno,
        ),
        {},
    )
    return bytecode
