from .conftest import assert_bytecode_for_args


def test_return_constant():
    def constant():
        return 42

    assert_bytecode_for_args(constant)


def test_return_closure():
    a = 10

    def closure():
        return a

    assert_bytecode_for_args(closure)


def test_return_arg():
    def identity(a):
        return a

    assert_bytecode_for_args(identity, 0)
    assert_bytecode_for_args(identity, 10)
    assert_bytecode_for_args(identity, "a")


def test_no_return():
    def no_return():
        pass

    assert_bytecode_for_args(no_return)


def test_single_assignment():
    def single_assignment(value):
        a = value
        return a

    assert_bytecode_for_args(single_assignment, 10)
    assert_bytecode_for_args(single_assignment, 20)
    assert_bytecode_for_args(single_assignment, "a")


def test_variable_deletion():
    def delete_variable():
        a = 1
        del a
        return a  # noqa

    assert_bytecode_for_args(delete_variable)


def test_free_variable_deletion():
    a = 0  # noqa
    del a

    def delete_variable():
        nonlocal a
        return a  # noqa

    assert_bytecode_for_args(delete_variable)


def test_multi_assignment():
    def multi_assignment(value):
        a = b = value
        return a + b

    assert_bytecode_for_args(multi_assignment, 10)
    assert_bytecode_for_args(multi_assignment, 20)
    assert_bytecode_for_args(multi_assignment, "a")


def test_multi_target_assignment():
    def multi_target_assignment(value):
        a, b = value
        return a + b

    assert_bytecode_for_args(multi_target_assignment, [1])
    assert_bytecode_for_args(multi_target_assignment, [1, 2])
    assert_bytecode_for_args(multi_target_assignment, [1, 2, 3])


def test_nested_multi_target_assignment():
    def nested_multi_target_assignment(value):
        a, (b, c) = value
        return a, b, c

    assert_bytecode_for_args(nested_multi_target_assignment, [1])
    assert_bytecode_for_args(nested_multi_target_assignment, [1, (2,)])
    assert_bytecode_for_args(nested_multi_target_assignment, [1, (2, 3)])
    assert_bytecode_for_args(nested_multi_target_assignment, [1, (2, 3, 4)])


def test_multi_target_extras_assignment():
    def multi_target_extras_assignment(value):
        a, *b, c = value
        return a, b, c

    assert_bytecode_for_args(multi_target_extras_assignment, [1])
    assert_bytecode_for_args(multi_target_extras_assignment, [1, 2])
    assert_bytecode_for_args(multi_target_extras_assignment, [1, 2, 3])
    assert_bytecode_for_args(multi_target_extras_assignment, [1, 2, 3, 4])


def test_attr_assignment():
    class A: ...

    def attr_assignment(value):
        a = A()
        a.x = value
        return a.x

    assert_bytecode_for_args(attr_assignment, 10)
    assert_bytecode_for_args(attr_assignment, 20)
    assert_bytecode_for_args(attr_assignment, "a")


def test_attr_deletion():
    class A:
        def __init__(self):
            self.x = 1

    def attr_delete():
        a = A()
        del a.x
        return a.x

    assert_bytecode_for_args(attr_delete)


def test_attr_inplace_add():
    class A: ...

    def attr_assignment(value):
        a = A()
        a.x = 1
        a.x += value
        return a.x

    assert_bytecode_for_args(attr_assignment, 10)
    assert_bytecode_for_args(attr_assignment, 20)


def test_add():
    def add(a, b):
        return a + b

    assert_bytecode_for_args(add, 1, 2)


def test_inplace_add():
    def add(a, b):
        a += b
        return a

    assert_bytecode_for_args(add, 1, 2)


def test_comparison():
    def lt(a, b):
        return a < b

    assert_bytecode_for_args(lt, 1, 2)


def test_chained_comparison():
    def lt(a, b, c):
        return a < b < c

    assert_bytecode_for_args(lt, 1, 2, 3)
    assert_bytecode_for_args(lt, 1, 3, 2)
    assert_bytecode_for_args(lt, 3, 2, 1)


def test_is():
    def is_(a, b):
        return a is b

    a = "1"
    b = "2"
    assert_bytecode_for_args(is_, a, a)
    assert_bytecode_for_args(is_, b, b)
    assert_bytecode_for_args(is_, a, b)


def test_is_not():
    def is_not(a, b):
        return a is not b

    a = "1"
    b = "2"
    assert_bytecode_for_args(is_not, a, a)
    assert_bytecode_for_args(is_not, b, b)
    assert_bytecode_for_args(is_not, a, b)


def test_in():
    def in_(a, b):
        return a in b

    a = "a"
    b = "abc"
    assert_bytecode_for_args(in_, a, a)
    assert_bytecode_for_args(in_, b, b)
    assert_bytecode_for_args(in_, a, b)
    assert_bytecode_for_args(in_, b, a)


def test_and():
    def and_(a, b):
        return a and b

    assert_bytecode_for_args(and_, 1, 2)
    assert_bytecode_for_args(and_, False, 2)
    assert_bytecode_for_args(and_, 0, False)
    assert_bytecode_for_args(and_, True, 5)


def test_or():
    def or_(a, b):
        return a or b

    assert_bytecode_for_args(or_, 1, 2)
    assert_bytecode_for_args(or_, False, 2)
    assert_bytecode_for_args(or_, 1, False)
    assert_bytecode_for_args(or_, True, 5)


def test_if():
    def if_(value):
        if value < 5:
            return -value
        else:
            return value

    assert_bytecode_for_args(if_, 1)
    assert_bytecode_for_args(if_, 3)
    assert_bytecode_for_args(if_, 5)
    assert_bytecode_for_args(if_, 6)
    assert_bytecode_for_args(if_, 8)


def test_if_expr():
    def if_(value):
        return -value if value < 5 else value

    assert_bytecode_for_args(if_, 1)
    assert_bytecode_for_args(if_, 3)
    assert_bytecode_for_args(if_, 5)
    assert_bytecode_for_args(if_, 6)
    assert_bytecode_for_args(if_, 8)


def test_while():
    def power(base, exponent):
        out = 1
        while exponent > 0:
            out *= base
            exponent -= 1
        return out

    assert_bytecode_for_args(power, 2, 0)
    assert_bytecode_for_args(power, 2, 1)
    assert_bytecode_for_args(power, 2, 2)
    assert_bytecode_for_args(power, 2, 3)

    assert_bytecode_for_args(power, 3, 0)
    assert_bytecode_for_args(power, 3, 1)
    assert_bytecode_for_args(power, 3, 2)
    assert_bytecode_for_args(power, 3, 3)


def test_while_else():
    def is_prime(number):
        tested = 2
        while tested < number:
            if number % tested == 0:
                break
            tested += 1
        else:
            return True
        return False

    assert_bytecode_for_args(is_prime, 2)
    assert_bytecode_for_args(is_prime, 3)
    assert_bytecode_for_args(is_prime, 4)
    assert_bytecode_for_args(is_prime, 5)
    assert_bytecode_for_args(is_prime, 6)
    assert_bytecode_for_args(is_prime, 7)


def test_while_continue():
    def count_even(end):
        current = 0
        out = 0
        while current < end:
            current += 1
            if current % 2 != 1:
                continue
            out += 1
        return out

    assert_bytecode_for_args(count_even, 0)
    assert_bytecode_for_args(count_even, 1)
    assert_bytecode_for_args(count_even, 2)
    assert_bytecode_for_args(count_even, 3)
    assert_bytecode_for_args(count_even, 4)
    assert_bytecode_for_args(count_even, 5)


def test_for():
    def power(base, exponent):
        out = 1
        for i in range(exponent):
            out *= base
        return out

    assert_bytecode_for_args(power, 2, 0)
    assert_bytecode_for_args(power, 2, 1)
    assert_bytecode_for_args(power, 2, 2)
    assert_bytecode_for_args(power, 2, 3)

    assert_bytecode_for_args(power, 3, 0)
    assert_bytecode_for_args(power, 3, 1)
    assert_bytecode_for_args(power, 3, 2)
    assert_bytecode_for_args(power, 3, 3)


def test_for_else():
    def is_prime(number):
        for tested in range(2, number):
            if number % tested == 0:
                break
            tested += 1
        else:
            return True
        return False

    assert_bytecode_for_args(is_prime, 2)
    assert_bytecode_for_args(is_prime, 3)
    assert_bytecode_for_args(is_prime, 4)
    assert_bytecode_for_args(is_prime, 5)
    assert_bytecode_for_args(is_prime, 6)
    assert_bytecode_for_args(is_prime, 7)


def test_for_continue():
    def count_even(end):
        out = 0
        for current in range(end):
            if current % 2 != 0:
                continue
            out += 1
        return out

    assert_bytecode_for_args(count_even, 0)
    assert_bytecode_for_args(count_even, 1)
    assert_bytecode_for_args(count_even, 2)
    assert_bytecode_for_args(count_even, 3)
    assert_bytecode_for_args(count_even, 4)
    assert_bytecode_for_args(count_even, 5)


def test_try():
    def try_(error):
        try:
            raise error
        except IOError:
            return 1
        except ValueError:
            return 2
        except:
            return 3

    assert_bytecode_for_args(try_, IOError)
    assert_bytecode_for_args(try_, ValueError)
    assert_bytecode_for_args(try_, SystemError)


def test_try_finally_return():
    def try_():
        try:
            return 1
        finally:
            return 2

    assert_bytecode_for_args(try_)


def test_try_finally_no_return():
    def try_():
        try:
            value = 1
            return value
        finally:
            value = 2

    assert_bytecode_for_args(try_)


def test_try_finally_return_raises():
    def try_():
        try:
            value = 1
            return value
        finally:
            raise ValueError

    assert_bytecode_for_args(try_)


def test_try_except_raises():
    def try_(error):
        failed = True
        try:
            raise error
        except IOError:
            raise ValueError
        except ValueError:
            failed = False
            raise
        finally:
            if failed:
                raise SystemError

    assert_bytecode_for_args(try_, IOError)
    assert_bytecode_for_args(try_, ValueError)
    assert_bytecode_for_args(try_, NameError)


def test_try_inner_try():
    def try_(error):
        try:
            raise error
        except ValueError:
            try:
                raise IOError
            except IOError:
                return 1
        except IOError:
            return 2

    assert_bytecode_for_args(try_, ValueError)
    assert_bytecode_for_args(try_, IOError)


def test_list():
    def list_(arg):
        return [1, arg, "3"]

    assert_bytecode_for_args(list_, 1)
    assert_bytecode_for_args(list_, 2)


def test_tuple():
    def tuple_(arg):
        return 1, arg, "3"

    assert_bytecode_for_args(tuple_, 1)
    assert_bytecode_for_args(tuple_, 2)


def test_set():
    def set_(arg):
        return {1, arg, "3"}

    assert_bytecode_for_args(set_, 1)
    assert_bytecode_for_args(set_, 2)


def test_dict():
    def dict_(arg):
        return {"a": 1, "b": arg, arg: "3"}

    assert_bytecode_for_args(dict_, "c")
    assert_bytecode_for_args(dict_, "b")


def test_list_extend():
    def list_(arg):
        return [1, *arg, "3"]

    assert_bytecode_for_args(list_, [1, 2])
    assert_bytecode_for_args(list_, [1, "3"])


def test_tuple_extend():
    def tuple_(arg):
        return 1, *arg, "3"

    assert_bytecode_for_args(tuple_, [1, 2])
    assert_bytecode_for_args(tuple_, [1, "3"])


def test_set_update():
    def set_(arg):
        return {1, *arg, "3"}

    assert_bytecode_for_args(set_, [1, 2])
    assert_bytecode_for_args(set_, [2, 4])


def test_dict_update():
    def dict_(arg):
        return {"a": 1, "b": arg, **arg}

    assert_bytecode_for_args(dict_, {"c": 3, "d": 4})
    assert_bytecode_for_args(dict_, {"c": 3, "a": 2})


def test_get_item():
    def get_item(items, index):
        return items[index]

    assert_bytecode_for_args(get_item, [1, 2, 3], 0)
    assert_bytecode_for_args(get_item, [1, 2, 3], 1)
    assert_bytecode_for_args(get_item, [1, 2, 3], 2)
    assert_bytecode_for_args(get_item, [1, 2, 3], -1)
    assert_bytecode_for_args(get_item, {"a": 10}, "a")


def test_get_item_chain():
    def get_item(items, index1, index2):
        return items[index1][index2]

    assert_bytecode_for_args(get_item, [[1]], 0, 0)
    assert_bytecode_for_args(get_item, [[0], [0, 1, 2]], 1, -1)


def test_set_item():
    def set_item(index, value):
        a = [1, 2, 3]
        a[index] = value
        return a

    assert_bytecode_for_args(set_item, 0, "a")
    assert_bytecode_for_args(set_item, 1, "b")
    assert_bytecode_for_args(set_item, 2, "c")
    assert_bytecode_for_args(set_item, -1, "d")


def test_set_item_chain():
    def set_item(index1, index2, value):
        a = [[1, 2, 3], [4, 5, 6]]
        a[index1][index2] = value
        return a

    assert_bytecode_for_args(set_item, 0, 0, "a")
    assert_bytecode_for_args(set_item, 1, 0, "b")
    assert_bytecode_for_args(set_item, 0, -1, "c")


def test_aug_set_item():
    def aug_set_item(index, value):
        a = [1, 2, 3]
        a[index] += value
        return a

    assert_bytecode_for_args(aug_set_item, 0, 1)
    assert_bytecode_for_args(aug_set_item, 1, 2)
    assert_bytecode_for_args(aug_set_item, 2, 3)
    assert_bytecode_for_args(aug_set_item, -1, 3)


def test_aug_set_item_chain():
    def aug_set_item(index1, index2, value):
        a = [[1, 2, 3], [4, 5, 6]]
        a[index1][index2] += value
        return a

    assert_bytecode_for_args(aug_set_item, 0, 0, 1)
    assert_bytecode_for_args(aug_set_item, 1, 0, 2)
    assert_bytecode_for_args(aug_set_item, 1, -1, 3)


def test_delete_item():
    def delete_item(index):
        a = [1, 2, 3]
        del a[index]
        return a

    assert_bytecode_for_args(delete_item, 0)
    assert_bytecode_for_args(delete_item, 1)
    assert_bytecode_for_args(delete_item, 2)
    assert_bytecode_for_args(delete_item, -1)


def test_delete_item_chain():
    def delete_item(index1, index2):
        a = [[1, 2, 3], [4, 5, 6]]
        del a[index1][index2]
        return a

    assert_bytecode_for_args(delete_item, 0, 0)
    assert_bytecode_for_args(delete_item, 1, 0)
    assert_bytecode_for_args(delete_item, 0, -1)


def test_f_string():
    def f_string(value):
        return f"Hello {value}!"

    assert_bytecode_for_args(f_string, "World")
    assert_bytecode_for_args(f_string, "Earth")
    assert_bytecode_for_args(f_string, 10)


def test_f_string_with_str_conversion():
    class A:
        def __str__(self):
            return "A"

        def __format__(self, format_spec):
            raise NotImplementedError

    def f_string(value):
        return f"Hello {value!s}!"

    assert_bytecode_for_args(f_string, 1)
    assert_bytecode_for_args(f_string, ["a", "b", "c"])
    assert_bytecode_for_args(f_string, A())


def test_f_string_with_repr_conversion():
    class A:
        def __repr__(self):
            return "A"

        def __format__(self, format_spec):
            raise NotImplementedError

    def f_string(value):
        return f"Hello {value!r}!"

    assert_bytecode_for_args(f_string, 1)
    assert_bytecode_for_args(f_string, ["a", "b", "c"])
    assert_bytecode_for_args(f_string, A())


def test_f_string_with_ascii_conversion():
    class A:
        def __str__(self):
            return "Å"

        def __format__(self, format_spec):
            raise NotImplementedError

    def f_string(value):
        return f"Hello {value!a}!"

    assert_bytecode_for_args(f_string, 1)
    assert_bytecode_for_args(f_string, ["Å", "b", "c"])
    assert_bytecode_for_args(f_string, A())


def test_f_string_with_spec():
    def f_string(value):
        return f"Hello {value:b}!"

    assert_bytecode_for_args(f_string, 1)
    assert_bytecode_for_args(f_string, 6)
    assert_bytecode_for_args(f_string, 12)


def test_f_string_multiple_joined():
    def f_string(greetings, name):
        return f"{greetings} {name}!"

    assert_bytecode_for_args(f_string, "Hello", "World")
    assert_bytecode_for_args(f_string, "Goodbye", "World")


def test_list_comprehension():
    def list_comp(numbers):
        x = 5
        out = [x * 2 for x in numbers]
        return x, out

    assert_bytecode_for_args(list_comp, [])
    assert_bytecode_for_args(list_comp, [1, 2, 3])


def test_set_comprehension():
    def set_comp(numbers):
        x = 5
        out = {x for x in numbers}
        return x, out

    assert_bytecode_for_args(set_comp, [])
    assert_bytecode_for_args(set_comp, [1, 2, 3])
    assert_bytecode_for_args(set_comp, [1, 2, 2])


def test_dict_comprehension():
    def dict_comp(numbers):
        x = 5
        out = {x: x**2 for x in numbers}
        return x, out

    assert_bytecode_for_args(dict_comp, [])
    assert_bytecode_for_args(dict_comp, [1, 2, 3])
    assert_bytecode_for_args(dict_comp, [1, 2, 2])


def test_list_comprehension_multiple_generators():
    def list_comp(xs, ys):
        return [(x, y, x * y) for x in xs for y in ys]

    assert_bytecode_for_args(list_comp, [], [])
    assert_bytecode_for_args(list_comp, [], [1, 2, 3])
    assert_bytecode_for_args(list_comp, [1, 2, 3], [])
    assert_bytecode_for_args(list_comp, [1, 2, 3], [4, 5, 6])
    assert_bytecode_for_args(list_comp, [1, 2], [3, 4, 5, 6])


def test_inner_function():
    def outer_function(x):
        def inner_function(y):
            return x + y
        return inner_function(x * 2)

    assert_bytecode_for_args(outer_function, 1)
    assert_bytecode_for_args(outer_function, 2)
    assert_bytecode_for_args(outer_function, 3)


def test_inner_function_with_defaults():
    def outer_function(x):
        def inner_function(y=x*2):
            return x + y
        x = 10
        return inner_function()

    assert_bytecode_for_args(outer_function, 1)
    assert_bytecode_for_args(outer_function, 2)
    assert_bytecode_for_args(outer_function, 3)


def test_lambda_function():
    def outer_function(x):
        adder = lambda y: x + y
        return adder(x * 2)

    assert_bytecode_for_args(outer_function, 1)
    assert_bytecode_for_args(outer_function, 2)
    assert_bytecode_for_args(outer_function, 3)


def test_lambda_function_with_defaults():
    def outer_function(x):
        adder = lambda y=x*2: x + y
        x = 10
        return adder()

    assert_bytecode_for_args(outer_function, 1)
    assert_bytecode_for_args(outer_function, 2)
    assert_bytecode_for_args(outer_function, 3)
