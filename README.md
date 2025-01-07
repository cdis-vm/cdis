# cdis - a *consistent* Python disassembler

## What is it?

*cdis*, pronounced "see this", is a Python disassembler that produce consistent results across Python versions.
CPython bytecode is neither forward or backwards compatible, so it outputs bytecode for the "cdis Virtual Machine",
which when executed, has the exact same behaviour as CPython's bytecode.

## Why would I use it?

- To write a Python compiler to any language of your choice
- To determine what symbols a function uses
- As a compiler target to generate Python code from another language


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