# Python Essentials Primer

This page introduces the Python features used in Project 1. You are not
expected to memorize everything. Keep it open while working and try the examples
in the notebook or a Python shell.

## Values, variables, and types

Python creates a variable when you assign a value. Unlike Java or C++, you do
not write the type before the variable name.

```python
count = 3             # int: a whole number
distance = 1.25       # float: a decimal number
name = "Ada"          # str: text
is_moving = True      # bool: True or False
```

Use `type(value)` to inspect a value's type. `int(...)`, `float(...)`, and
`str(...)` convert compatible values to another type.

## Lists: ordered and mutable collections

A list stores values in order and uses square brackets. **Mutable** means that
the list can change after it is created.

```python
prices = [1.2, 0.7, 2.5]

first = prices[0]          # 1.2; indexing begins at zero
last = prices[-1]          # 2.5; -1 means the final item
middle = prices[1:3]       # [0.7, 2.5]; stop index is excluded

prices.append(3.1)         # add one item at the end
prices[0] = 1.0            # replace an item
number_of_items = len(prices)
```

Useful list operations:

```python
values = [4, 1, 7]
sum(values)       # 12
min(values)       # 1
max(values)       # 7
len(values)       # 3
7 in values       # True
```

`sum`, `min`, and `max` need numeric/comparable items. `min([])` and `max([])`
raise an error because an empty list has no smallest or largest item.

## Tuples: ordered values that do not change

A tuple usually uses parentheses. It is **immutable**, so its items cannot be
replaced. Tuples are useful for a fixed group such as a 2D coordinate.

```python
position = (2.0, 3.0)
x = position[0]
x, y = position          # tuple unpacking: x gets 2.0 and y gets 3.0

def location():
    return 2.0, 3.0      # returning two values creates a tuple
```

Use a list when the collection should grow or change. Use a tuple when the
values form one fixed record. Both support `len`, indexing, slicing, and loops.

## Dictionaries: key-value mappings

A dictionary associates each unique key with a value. It uses braces and is
mutable.

```python
record = {
    "name": "Ada",
    "score": 85,
    "position": [1.0, 2.0],
}

name = record["name"]                # error if the key is absent
team = record.get("team", "none")    # use "none" if team is absent
record["score"] = 90                 # update an existing value
record["color"] = "blue"             # add a new key-value pair
```

`key in record` checks whether a key exists. A loop such as
`for key, value in record.items():` visits key-value pairs.

## Built-in functions used in this project

These functions are always available; no import is required.

| Expression | Meaning |
|---|---|
| `len(collection)` | number of items |
| `sum(numbers)` | total of numeric items |
| `min(values)` / `max(values)` | smallest / largest item |
| `range(n)` | integers from 0 through n−1, commonly used in loops |
| `enumerate(values)` | `(index, value)` pairs |
| `zip(a, b)` | pairs items from two collections, stopping at the shorter one |
| `type(value)` | the value's type |
| `isinstance(value, int)` | whether a value has a particular type |
| `print(value)` | display a value; it does not return that value |

Example loops:

```python
for index in range(3):
    print(index)                  # prints 0, 1, 2

names = ["apple", "banana"]
prices = [1.2, 0.7]
for index, (name, price) in enumerate(zip(names, prices)):
    print(index, name, price)
```

## Copying, aliasing, and mutation

Assignment does not copy a list or dictionary. Both names below refer to the
same object:

```python
original = [1, 2]
alias = original
alias.append(3)
print(original)       # [1, 2, 3]
```

Use a full slice or `.copy()` when you need a separate shallow copy:

```python
copy_a = original[:]
copy_b = original.copy()
```

This distinction explains why `update_position` intentionally changes its
input, while a function that should not affect its caller must build and
return a new list instead.

## Conditionals and boolean logic

Conditionals select which code runs. Indentation defines the body of each
branch.

```python
if temperature < 10:
    label = "cold"
elif temperature < 25:
    label = "mild"
else:
    label = "hot"
```

Comparisons (`<`, `<=`, `==`, `!=`, `>=`, `>`) produce booleans. Combine them
with `and`, `or`, and `not`. Be precise at boundaries: `< 25` excludes 25.

## For-loops, while-loops, and accumulators

A `for` loop visits each item. An accumulator stores a running result:

```python
count = 0
for price in prices:
    if price < 1.0:
        count += 1
```

A `while` loop repeats while its condition is true:

```python
steps = 0
while remaining > 0:
    remaining -= 10
    steps += 1
```

Make sure the loop changes the condition, or it may never terminate. Validate
values such as a zero step size before entering the loop.

## Functions, parameters, and return values

Functions package reusable behavior:

```python
def move(x, y, dx, dy):
    new_x = x + dx
    new_y = y + dy
    return new_x, new_y
```

Parameters are local names receiving arguments. A default makes an argument
optional: `def clamp(value, minimum=0, maximum=100)`. `return` gives a result to
the caller and ends the function. `print` only displays something; it does not
return that value. A function reaching its end without `return` returns `None`.

## Errors and validation

When an input value violates a function's documented rules, raise a useful
exception:

```python
if speed <= 0:
    raise ValueError("speed must be positive")
```

This stops the function immediately and explains the problem. Tests can verify
that the expected error occurs.

## Imports, modules, and packages

A `.py` file is a module containing reusable names. A package groups modules.

```python
import math
import numpy as np
from pathlib import Path
```

The first form uses names such as `math.sin`. The alias in the second form uses
`np.array`. The third imports one name directly. Python's standard library
ships with Python; external packages such as NumPy must be installed.

## NumPy arrays

NumPy arrays support efficient numerical operations over whole collections:

```python
import numpy as np

vector = np.array([1, 2, 3], dtype=float)
scaled = vector * 2                 # [2., 4., 6.]

matrix = np.array([[1, 2], [3, 6]])
matrix.shape                        # (2, 2)
matrix[0]                           # first row
matrix[:, 0]                        # first column
np.mean(matrix, axis=0)             # one mean per column
np.linalg.norm(np.array([3, 4]))    # 5.0
```

An **axis** says which dimension an operation combines. For a 2D matrix,
`axis=0` combines rows and leaves one result per column; `axis=1` combines
columns and leaves one result per row.

## Classes and objects

A class defines objects that combine state and behavior:

```python
class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        return self.x, self.y

point = Point(1, 2)
point.move(3, -1)
print(point.x)          # 4
```

`self` is the current object. Attributes such as `self.x` persist between
method calls. `__init__` initializes a newly created instance.

## Assertions, tests, and debugging

An assertion documents an expected condition:

```python
assert move(1, 2, 3, 4) == (4, 6)
```

Pytest finds test functions and reports failed assertions. When a test fails,
read the traceback from the bottom upward, reproduce its input in a small cell,
inspect intermediate values and types, fix one issue, reload the module, and
run the test again.

## Official Python tutorials

For more examples, use these official references:

- [An Informal Introduction to Python](https://docs.python.org/3/tutorial/introduction.html)
  covers numbers, strings, lists, indexing, and first programming steps.
- [Python Data Structures](https://docs.python.org/3/tutorial/datastructures.html)
  covers lists, tuples, dictionaries, looping techniques, and comprehensions.
- [Built-in Functions](https://docs.python.org/3/library/functions.html) documents
  `len`, `sum`, `min`, `max`, `range`, `enumerate`, `zip`, and other built-ins.
- [Defining Functions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions)
  explains parameters, defaults, and return values.
- [Errors and Exceptions](https://docs.python.org/3/tutorial/errors.html) explains
  exceptions, `raise`, and error handling.
- [Modules](https://docs.python.org/3/tutorial/modules.html) explains imports and
  reusable Python files.
- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html) introduces
  arrays, shapes, indexing, axes, and numerical operations.

The official tutorial contains more material than this project requires. Focus
on the linked topics and return here when you are ready to attempt each TODO.
