import inspect
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from functools import cached_property

from .opcode import Instruction, StackMetadata, ValueSource, FunctionType

if TYPE_CHECKING:
    from _compiler import Bytecode


@dataclass(frozen=True, kw_only=True)
class BasicBlock:
    bytecode: "Bytecode"
    start_index_inclusive: int
    end_index_exclusive: int

    @cached_property
    def instructions(self) -> tuple[Instruction, ...]:
        return tuple(
            self.bytecode.instructions[
                self.start_index_inclusive : self.end_index_exclusive
            ]
        )

    @property
    def leader(self):
        return self.instructions[0]

    @property
    def exit(self):
        return self.instructions[-1]

    def __contains__(self, item: Instruction | int) -> bool:
        match item:
            case Instruction(bytecode_index=index):
                return index in self
            case int(index):
                return self.start_index_inclusive <= index < self.end_index_exclusive
            case _:
                raise ValueError(f"{item} must be an int or an Instruction.")

    def __eq__(self, other):
        match other:
            case BasicBlock(start_index_inclusive=index):
                return self.start_index_inclusive == index
            case _:
                return False

    def __hash__(self):
        return hash(self.start_index_inclusive)


def resolve_stack_metadata(bytecode: "Bytecode") -> tuple[StackMetadata, ...]:
    leader_indices = []
    # Set initially to True so first instruction get added as leader
    next_instruction_was_skipped = True
    for instruction in bytecode.instructions:
        if next_instruction_was_skipped or any(
            instruction.bytecode_index == label.index for label in bytecode.labels
        ):
            leader_indices.append(instruction.bytecode_index)
        next_instruction_was_skipped = (
            instruction.bytecode_index + 1
            not in instruction.opcode.next_bytecode_indices(instruction)
        )
    basic_blocks: list[BasicBlock] = []
    jump_target_to_basic_block: dict[int, BasicBlock] = {}

    for index in range(len(leader_indices) - 1):
        basic_block = BasicBlock(
            bytecode=bytecode,
            start_index_inclusive=leader_indices[index],
            end_index_exclusive=leader_indices[index + 1],
        )
        jump_target_to_basic_block[leader_indices[index]] = basic_block
        basic_blocks.append(basic_block)
    else:
        basic_block = BasicBlock(
            bytecode=bytecode,
            start_index_inclusive=leader_indices[-1],
            end_index_exclusive=len(bytecode.instructions),
        )
        jump_target_to_basic_block[leader_indices[-1]] = basic_block
        basic_blocks.append(basic_block)

    initial_stack_metadata = StackMetadata(
        stack=(),
        variables={
            name: ValueSource(
                sources=(),
                value_type=parameter.annotation
                if parameter.annotation is not inspect._empty
                else object,
            )
            for (name, parameter) in bytecode.signature.parameters.items()
        },
        synthetic_variables=((ValueSource((), object)),)
        if bytecode.function_type is FunctionType.GENERATOR
        else (),
        dead=False,
    )
    opcode_index_to_stack_metadata: dict[int, StackMetadata] = {
        0: initial_stack_metadata,
    }

    for basic_block in basic_blocks:
        for instruction in basic_block.instructions:
            current_stack_metadata = (
                opcode_index_to_stack_metadata[instruction.bytecode_index]
                if instruction.bytecode_index in opcode_index_to_stack_metadata
                else StackMetadata.dead_code()
            )
            if current_stack_metadata.dead:
                opcode_index_to_stack_metadata[instruction.bytecode_index] = (
                    current_stack_metadata
                )

    for exception_handler in bytecode.exception_handlers:
        target_index = exception_handler.handler_label.index
        exception_stack_metadata = replace(
            opcode_index_to_stack_metadata[exception_handler.from_label.index],
            stack=(
                ValueSource(sources=(), value_type=exception_handler.exception_class),
            ),
        )
        if target_index in opcode_index_to_stack_metadata:
            opcode_index_to_stack_metadata[target_index] = (
                exception_stack_metadata.unify_with(
                    opcode_index_to_stack_metadata[target_index]
                )
            )
        else:
            opcode_index_to_stack_metadata[target_index] = exception_stack_metadata

    has_changed = True
    while has_changed:  # Keep unifying until no changes are detected
        has_changed = False
        for basic_block in basic_blocks:
            for instruction in basic_block.instructions:
                original_instruction_metadata = opcode_index_to_stack_metadata[
                    instruction.bytecode_index
                ]
                branches = instruction.opcode.next_bytecode_indices(instruction)
                next_stack_metadata_for_branches = (
                    (StackMetadata.dead_code(),) * len(branches)
                    if original_instruction_metadata.dead
                    else instruction.opcode.next_stack_metadata(
                        instruction, bytecode, original_instruction_metadata
                    )
                )

                for branch_index in range(len(branches)):
                    branch_target = branches[branch_index]
                    next_stack_metadata = next_stack_metadata_for_branches[branch_index]
                    original_opcode_metadata = opcode_index_to_stack_metadata[
                        branch_target
                    ]
                    try:
                        new_opcode_metadata = original_opcode_metadata.unify_with(
                            next_stack_metadata
                        )
                    except ValueError as e:
                        raise ValueError(
                            f"Stack metadata mismatch for bytecode index {branch_target} for bytecode\n{bytecode}"
                        ) from e
                    opcode_index_to_stack_metadata[branch_target] = new_opcode_metadata
                    has_changed |= new_opcode_metadata != original_opcode_metadata

        for exception_handler in bytecode.exception_handlers:
            target_index = exception_handler.handler_label.index
            original_exception_stack_metadata = opcode_index_to_stack_metadata[
                target_index
            ]
            new_exception_stack_metadata = replace(
                opcode_index_to_stack_metadata[exception_handler.from_label.index],
                stack=(
                    ValueSource(
                        sources=(), value_type=exception_handler.exception_class
                    ),
                ),
            ).unify_with(original_exception_stack_metadata)
            opcode_index_to_stack_metadata[target_index] = new_exception_stack_metadata
            has_changed |= (
                new_exception_stack_metadata != original_exception_stack_metadata
            )

    return tuple(
        value[1]
        for value in sorted(
            opcode_index_to_stack_metadata.items(), key=lambda item: item[0]
        )
    )
