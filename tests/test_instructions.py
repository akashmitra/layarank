"""Tests for instructions, presets, and validation logic."""

import pytest
from layarank.errors import ConfigurationError
from layarank.instructions import (
    PAIRWISE_INSTRUCTION,
    POINTWISE_RELEVANCE_INSTRUCTION,
    RELEVANCE_INSTRUCTION,
    RERANK_INSTRUCTION,
    validate_instruction,
)


def test_builtin_presets_are_valid():
    assert validate_instruction(RERANK_INSTRUCTION, "listwise")
    assert validate_instruction(RERANK_INSTRUCTION, "pointwise")
    assert validate_instruction(RELEVANCE_INSTRUCTION, "listwise")
    assert validate_instruction(POINTWISE_RELEVANCE_INSTRUCTION, "pointwise")
    assert validate_instruction(PAIRWISE_INSTRUCTION, "pairwise")


def test_invalid_instruction_structure():
    with pytest.raises(ConfigurationError, match="must be a dict"):
        validate_instruction("invalid", "listwise")

    with pytest.raises(ConfigurationError, match="exactly 'instructions' and 'criteria'"):
        validate_instruction({"instructions": "test"}, "listwise")


def test_invalid_placeholders():
    with pytest.raises(ConfigurationError, match="only these placeholders"):
        validate_instruction(
            {
                "instructions": "Does {wrong_key} answer the query?",
                "criteria": {"true": "yes", "false": "no"},
            },
            "listwise",
        )

    with pytest.raises(ConfigurationError, match="only these placeholders"):
        validate_instruction(
            {
                "instructions": "Does {document} answer the query?",
                "criteria": {"true": "yes", "false": "no"},
            },
            "pairwise",
        )


def test_invalid_criteria():
    with pytest.raises(ConfigurationError, match="'true' and 'false' keys"):
        validate_instruction(
            {
                "instructions": "Does {document} answer the query?",
                "criteria": {"yes": "correct", "no": "wrong"},
            },
            "listwise",
        )

    with pytest.raises(ConfigurationError, match="non-empty strings"):
        validate_instruction(
            {
                "instructions": "Does {document} answer the query?",
                "criteria": {"true": "", "false": "no"},
            },
            "listwise",
        )
