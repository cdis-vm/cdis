import cdis.opcode as opcode
import cdis.compiler as compiler
from ._vm import CDisVM

to_bytecode = compiler.to_bytecode


__all__ = ("opcode", "to_bytecode", "CDisVM")
