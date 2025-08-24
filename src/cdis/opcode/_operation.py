from ._api import StackMetadata, Opcode, ValueSource, Instruction
import ast
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, TYPE_CHECKING
import operator

if TYPE_CHECKING:
    from ..compiler._api import BytecodeDescriptor
    from .._vm import Frame


class OperatorType(Enum):
    """Represent if an operator is a unary, binary or comparison operator."""

    UNARY_OPERATION = ast.unaryop
    """A unary operator, such as negation (-x) or invert (~x).
    """

    BINARY_OPERATION = ast.operator
    """A binary operator, such as (x + y) or (x - y)."""

    COMPARISON = ast.Eq | ast.NotEq | ast.Lt | ast.Gt | ast.LtE | ast.GtE
    """A comparison operator, such as (x < y), (x == y)."""


@dataclass(frozen=True)
class Operator:
    """A unary, binary or comparison operator."""

    ast: ast.operator | ast.cmpop | ast.unaryop
    """The AST node corresponding to the operator.
    """

    type: OperatorType
    """The type of the operator (unary, binary or comparison).
    """

    function: Callable
    """A function that can be called to perform the operation.
    """

    dunder_name: str
    """The dunder method corresponding to the operation from the left side.

    For example, for (x + y), this would be "__add__".
    """

    flipped_dunder_name: str | None = None
    """The dunder method corresponding to the operation from the right side.

    For example, for (x + y), this would be "__radd__".

    Optional.
    """


class UnaryOperator(Enum):
    """A unary operator, such as negation (-x) or invert (~x)."""

    Invert = Operator(
        ast.Invert(), OperatorType.UNARY_OPERATION, operator.invert, "__invert__"
    )
    """The inversion operator (~x).
    """

    UAdd = Operator(ast.UAdd(), OperatorType.UNARY_OPERATION, operator.pos, "__pos__")
    """The positive operator (+x).
    """

    USub = Operator(ast.USub(), OperatorType.UNARY_OPERATION, operator.neg, "__neg__")
    """The negation operator (-x).
    """


class BinaryOperator(Enum):
    """A binary or comparison operator, such as (x + y) or (x < y)."""

    Add = Operator(
        ast.Add(), OperatorType.BINARY_OPERATION, operator.add, "__add__", "__radd__"
    )
    """The addition operator (x + y)."""

    Sub = Operator(
        ast.Sub(), OperatorType.BINARY_OPERATION, operator.sub, "__sub__", "__rsub__"
    )
    """The subtraction operator (x - y)."""

    Mult = Operator(
        ast.Mult(), OperatorType.BINARY_OPERATION, operator.mul, "__mul__", "__rmul__"
    )
    """The multiplication operator (x * y)."""

    Div = Operator(
        ast.Div(),
        OperatorType.BINARY_OPERATION,
        operator.truediv,
        "__truediv__",
        "__rtruediv__",
    )
    """The true division operator (x / y)."""

    FloorDiv = Operator(
        ast.FloorDiv(),
        OperatorType.BINARY_OPERATION,
        operator.floordiv,
        "__floordiv__",
        "__rfloordiv__",
    )
    """The floor division operator (x // y)."""

    Mod = Operator(
        ast.Mod(), OperatorType.BINARY_OPERATION, operator.mod, "__mod__", "__rmod__"
    )
    """The moduli operator (x % y)."""

    Pow = Operator(
        ast.Pow(), OperatorType.BINARY_OPERATION, operator.pow, "__pow__", "__rpow__"
    )
    """The power operator (x ** y)."""

    LShift = Operator(
        ast.LShift(),
        OperatorType.BINARY_OPERATION,
        operator.lshift,
        "__lshift__",
        "__rlshift__",
    )
    """The left shift operator (x << y)."""

    RShift = Operator(
        ast.RShift(),
        OperatorType.BINARY_OPERATION,
        operator.rshift,
        "__rshift__",
        "__rrshift__",
    )
    """The right shift operator (x >> y)."""

    BitOr = Operator(
        ast.BitOr(), OperatorType.BINARY_OPERATION, operator.or_, "__or__", "__ror__"
    )
    """The bitwise or operator (x | y)."""

    BitXor = Operator(
        ast.BitXor(), OperatorType.BINARY_OPERATION, operator.xor, "__xor__", "__rxor__"
    )
    """The bitwise xor operator (x ^ y)."""

    BitAnd = Operator(
        ast.BitAnd(),
        OperatorType.BINARY_OPERATION,
        operator.and_,
        "__and__",
        "__rand__",
    )
    """The bitwise and operator (x & y)."""

    MatMult = Operator(
        ast.MatMult(),
        OperatorType.BINARY_OPERATION,
        operator.matmul,
        "__matmul__",
        "__rmatmul__",
    )
    """The matrix multiplication operator (x @ y)."""

    # Inplace
    IAdd = Operator(
        ast.Add(), OperatorType.BINARY_OPERATION, operator.add, "__iadd__", "__radd__"
    )
    """The inplace addition operator (x += y)."""

    ISub = Operator(
        ast.Sub(), OperatorType.BINARY_OPERATION, operator.sub, "__isub__", "__rsub__"
    )
    """The inplace subtraction operator (x -= y)."""

    IMult = Operator(
        ast.Mult(), OperatorType.BINARY_OPERATION, operator.mul, "__imul__", "__rmul__"
    )
    """The inplace multiplication operator (x *= y)."""

    IDiv = Operator(
        ast.Div(),
        OperatorType.BINARY_OPERATION,
        operator.truediv,
        "__itruediv__",
        "__rtruediv__",
    )
    """The inplace true division operator (x /= y)."""

    IFloorDiv = Operator(
        ast.FloorDiv(),
        OperatorType.BINARY_OPERATION,
        operator.floordiv,
        "__ifloordiv__",
        "__rfloordiv__",
    )
    """The inplace floor division operator (x //= y)."""

    IMod = Operator(
        ast.Mod(), OperatorType.BINARY_OPERATION, operator.mod, "__imod__", "__rmod__"
    )
    """The inplace moduli operator (x %= y)."""

    IPow = Operator(
        ast.Pow(), OperatorType.BINARY_OPERATION, operator.pow, "__ipow__", "__rpow__"
    )
    """The inplace power operator (x **= y)."""

    ILShift = Operator(
        ast.LShift(),
        OperatorType.BINARY_OPERATION,
        operator.lshift,
        "__ilshift__",
        "__rlshift__",
    )
    """The inplace left shift operator (x <<= y)."""

    IRShift = Operator(
        ast.RShift(),
        OperatorType.BINARY_OPERATION,
        operator.rshift,
        "__irshift__",
        "__rrshift__",
    )
    """The inplace right shift operator (x >>= y)."""

    IBitOr = Operator(
        ast.BitOr(), OperatorType.BINARY_OPERATION, operator.or_, "__ior__", "__ror__"
    )
    """The inplace bitwise or operator (x |= y)."""

    IBitXor = Operator(
        ast.BitXor(),
        OperatorType.BINARY_OPERATION,
        operator.xor,
        "__ixor__",
        "__rxor__",
    )
    """The inplace bitwise xor operator (x ^= y)."""

    IBitAnd = Operator(
        ast.BitAnd(),
        OperatorType.BINARY_OPERATION,
        operator.and_,
        "__iand__",
        "__rand__",
    )
    """The inplace bitwise and operator (x &= y)."""

    IMatMult = Operator(
        ast.MatMult(),
        OperatorType.BINARY_OPERATION,
        operator.matmul,
        "__imatmul__",
        "__rmatmul__",
    )
    """The inplace matrix multiplication operator (x @= y)."""

    # Comparison operators
    Eq = Operator(ast.Eq(), OperatorType.COMPARISON, operator.eq, "__eq__", "__eq__")
    """The equality operator (x == y)."""

    NotEq = Operator(
        ast.NotEq(), OperatorType.COMPARISON, operator.ne, "__ne__", "__ne__"
    )
    """The inequality operator (x != y)."""

    Lt = Operator(ast.Lt(), OperatorType.COMPARISON, operator.lt, "__lt__", "__gt__")
    """The less than operator (x < y)."""

    LtE = Operator(ast.LtE(), OperatorType.COMPARISON, operator.le, "__le__", "__ge__")
    """The less than or equal operator (x <= y)."""

    Gt = Operator(ast.Gt(), OperatorType.COMPARISON, operator.gt, "__gt__", "__lt__")
    """The greater than operator (x > y)."""

    GtE = Operator(ast.GtE(), OperatorType.COMPARISON, operator.ge, "__ge__", "__le__")
    """The greater than or equal operator (x >= y)."""


@dataclass(frozen=True)
class UnaryOp(Opcode):
    """Performs a unary operation on the operand on the top of the stack.

    Notes
    -----
        | UnaryOp
        | Stack Effect: 0
        | Prior: ..., operand
        | After: ..., result

    Examples
    --------
    >>> -x
    LoadLocal(name="x")
    UnaryOp(operator=UnaryOperator.USub)
    """

    operator: UnaryOperator

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(1).push(
                ValueSource(
                    sources=(instruction,),
                    value_type=object,  # TODO
                )
            ),
        )

    def execute(self, frame: "Frame") -> None:
        top = frame.stack.pop()
        frame.stack.append(self.operator.value.function(top))


@dataclass(frozen=True)
class BinaryOp(Opcode):
    """Performs a binary operation on the two items on the top of the stack.

    Despite seemly simple, this is one of the most complex opcodes.
    First, get the types of the left and right operands. If the
    right operand is a more specific type than the left operand
    (i.e. is a subclass of the left operand's type), try the reflected
    operation first (ex: right.__radd__(left)), otherwise try the normal
    operation first (ex: left.__add__(right)). If the method corresponding
    to the operation is not present, the method returns `NotImplemented`,
    or the operand is a builtin type and raises `TypeError`,
    then try the other operation. Written as Python code, it looks like this:

    >>> def binary_op(forward_op, reverse_op, left, right):
    ...     left_type = type(left)
    ...     right_type = type(right)
    ...     def try_op(op, first, second):
    ...         method = getattr(type(first), op, None)
    ...         if method is None:
    ...             return NotImplemented
    ...         try:
    ...             return method(first, second)
    ...         except TypeError:
    ...             if type(first) in {int, float, str, ...}
    ...                 return NotImplemented
    ...             else:
    ...                 raise
    ...     if issubclass(right_type, left_type):
    ...         out = try_op(reverse_op, right, left)
    ...         if out is NotImplemented:
    ...             out = try_op(forward_op, left, right)
    ...         if out is NotImplemented:
    ...             raise TypeError
    ...         return out
    ...     else:
    ...         out = try_op(forward_op, left, right)
    ...         if out is NotImplemented:
    ...             out = try_op(reverse_op, right, left)
    ...         if out is NotImplemented:
    ...             raise TypeError
    ...         return out

    Notes
    -----
        | BinaryOp
        | Stack Effect: -1
        | Prior: ..., left, right
        | After: ..., result

        Comparisons are also BinaryOp, and can return any type
        (for instance, (a < b) can return an int).

    Examples
    --------
    >>> x + y
    LoadLocal(name="x")
    LoadLocal(name="y")
    BinaryOp(operator=BinaryOperator.Add)
    """

    operator: BinaryOperator

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(2).push(
                ValueSource(
                    sources=(instruction,),
                    value_type=object,  # TODO
                )
            ),
        )

    def execute(self, frame: "Frame") -> None:
        right = frame.stack.pop()
        left = frame.stack.pop()
        frame.stack.append(self.operator.value.function(left, right))


# TODO: Inplace operations should be their own opcode, since the logic is
#       different.


@dataclass(frozen=True)
class IsSameAs(Opcode):
    """Pops off the two top items on the stack and check if they are the same reference.
    If they are the same reference, `True` is pushed to the stack; otherwise
    `False` is pushed to the stack. If `negate` is set, then the result
    is negated before being pushed to the stack.

    Notes
    -----
        | IsSameAs
        | Stack Effect: -1
        | Prior: ..., left, right
        | After: ..., result

    Examples
    --------
    >>> x is y
    LoadLocal(name="x")
    LoadLocal(name="y")
    IsSameAs(negate=False)

    >>> x is not y
    LoadLocal(name="x")
    LoadLocal(name="y")
    IsSameAs(negate=True)
    """

    negate: bool

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(2).push(
                ValueSource(sources=(instruction,), value_type=bool)
            ),
        )

    def execute(self, frame: "Frame") -> None:
        right = frame.stack.pop()
        left = frame.stack.pop()
        if self.negate:
            frame.stack.append(left is not right)
        else:
            frame.stack.append(left is right)


class FormatConversion(Enum):
    """How a value should be converted before being formatted in a f-string."""

    # Values taken from
    # https://docs.python.org/3/library/ast.html#ast.FormattedValue
    NONE = -1
    """Do no conversion before formatting.
    """

    TO_STRING = 115
    """Call str on the value before formatting.
    """

    TO_REPR = 114
    """Call repr on the value before formatting.
    """

    TO_ASCII = 97
    """Call ascii on the value before formatting.
    """

    @staticmethod
    def from_int(value: int) -> "FormatConversion":
        """Gets a `FormatConversion` from the `conversion` attribute of an `ast.FormattedValue` object.

        Parameters
        ----------
        value: int
            The `conversion` attribute of an `ast.FormattedValue` object.

        Returns
        -------
        The `FormatConversion` corresponding to ast int `value`.
        """
        for conversion in FormatConversion:
            if conversion.value == value:
                return conversion
        raise ValueError(f"Invalid conversion: {value}")

    def convert(self, value: Any) -> Any:
        """Performs the conversion.

        Parameters
        ----------
        value
            The value to convert.

        Returns
        -------
        The converted value.
        """
        match self:
            case FormatConversion.NONE:
                return value
            case FormatConversion.TO_STRING:
                return str(value)
            case FormatConversion.TO_REPR:
                return repr(value)
            case FormatConversion.TO_ASCII:
                return ascii(value)
            case _:
                raise RuntimeError(f"Missing conversion: {self}")


@dataclass(frozen=True)
class FormatValue(Opcode):
    """Formats the value on the top of stack, performing a conversion if necessary.
    Raises `TypeError` if __format__ does not return a `str`.

    Notes
    -----
        | FormatValue
        | Stack Effect: 0
        | Prior: ..., value
        | After: ..., formatted_value

    Examples
    --------
    >>> f'{x}'
    LoadLocal(name="x")
    FormatValue(conversion=FormatConversion.NONE, format_spec='')

    >>> f'{x!s}'
    LoadLocal(name="x")
    FormatValue(conversion=FormatConversion.TO_STRING, format_spec='')

    >>> f'{x:spec}'
    LoadLocal(name="x")
    FormatValue(conversion=FormatConversion.NONE, format_spec='spec')

    >>> f'{x!s:spec}'
    LoadLocal(name="x")
    FormatValue(conversion=FormatConversion.TO_STRING, format_spec='spec')
    """

    conversion: FormatConversion
    format_spec: str

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(1).push(
                ValueSource(sources=(instruction,), value_type=str)
            ),
        )

    def execute(self, frame: "Frame") -> None:
        value = self.conversion.convert(frame.stack.pop())
        frame.stack.append(format(value, self.format_spec))


@dataclass(frozen=True)
class JoinStringValues(Opcode):
    """Joins the top count items on the stack into a single string.
    The items on the stack are guaranteed to be instances of `str`.

    Notes
    -----
        | JoinStringValues
        | Stack Effect: -count + 1
        | Prior: ..., str_1, str_2, ..., str_count
        | After: ..., combined_str

    Examples
    --------
    >>> f'{greetings} {noun}!'
    LoadLocal(name="greetings")
    FormatValue(conversion=FormatConversion.NONE, format_spec='')
    LoadConstant(constant=' ')
    LoadLocal(name="noun")
    FormatValue(conversion=FormatConversion.NONE, format_spec='')
    LoadConstant(constant='!')
    JoinStringValues(count=3)
    """

    count: int

    def next_stack_metadata(
        self,
        instruction: Instruction,
        bytecode: "BytecodeDescriptor",
        previous_stack_metadata: StackMetadata,
    ) -> tuple[StackMetadata, ...]:
        return (
            previous_stack_metadata.pop(self.count).push(
                ValueSource(sources=(instruction,), value_type=str)
            ),
        )

    def execute(self, frame: "Frame") -> None:
        values = frame.stack[-self.count :]
        del frame.stack[-self.count :]
        out = "".join(values)
        frame.stack.append(out)


__all__ = (
    "OperatorType",
    "Operator",
    "UnaryOperator",
    "BinaryOperator",
    "UnaryOp",
    "BinaryOp",
    "IsSameAs",
    "FormatConversion",
    "FormatValue",
    "JoinStringValues",
)
