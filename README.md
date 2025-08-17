![cdis logo](https://raw.githubusercontent.com/cdis-vm/cdis/main/cdis-logo.png)

# cdis - a *consistent* Python disassembler

## Important

The `cdis` bytecode format will not be considered stable until the `1.0.0` release.
The `0.x.y` releases may have breaking changes in both the bytecode and the API exposed.

## What is it?

*cdis*, pronounced "see this", is a Python disassembler that produce consistent results across Python versions.
CPython bytecode is neither forward or backwards compatible, so it outputs bytecode for the "cdis Virtual Machine",
which when executed, has the exact same behaviour as CPython's bytecode.

## Why would I use it?

- To write a Python compiler to any language of your choice
- To determine what symbols a function uses
- As a compiler target to generate Python code from another language

## Known differences with CPython

- Different behaviour of `locals()` inside list comprehensions.
  In particular, we use a synthetic variable for the iterator variable
  instead of reusing the name.
  [CPython changed the behaviour of `locals()` in list comprehension
  in 3.12](https://docs.python.org/3/whatsnew/3.12.html#pep-709-comprehension-inlining),
  so this is undefined behaviour.
- Different behaviour of when match variables are bound.
  Technically, [Python allows binding even when subpatterns fail](https://docs.python.org/3/reference/compound_stmts.html#overview).
  That being said, CPython does appear to have consistent behaviour of delaying
  binding until all subpatterns succeed.
  As such, a future version of `cdis` will probably match
  CPython behaviour.


## Missing features

- User customizable optimization passes (Planned)
- User customizable VM tracer (Planned)
- A guide on how to actually use `cdis` (Planned).

# Run tests

Install test dependencies

```shell
pip install "pytest>8" "coverage" "tox"
pip install -e .
```

Run tests on current python version

```shell
pytest
```
