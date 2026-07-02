"""Tests for the tolerant JSON helpers used to parse LLM responses."""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph.intelligence.json_utils import (
    clamp_float,
    coerce_str,
    load_json_object,
)


class TestLoadJsonObject:
    def test_parses_clean_json_object(self):
        assert load_json_object('{"a": 1}', context="t") == {"a": 1}

    def test_strips_surrounding_whitespace(self):
        assert load_json_object('  \n {"a": 1}\n ', context="t") == {"a": 1}

    @pytest.mark.parametrize("content", [None, "", "   "])
    def test_empty_or_blank_returns_empty_dict(self, content):
        assert load_json_object(content, context="t") == {}

    def test_extracts_json_from_fenced_or_prose_wrapper(self):
        wrapped = 'Here is the result:\n```json\n{"score": 0.5}\n```\nThanks!'
        assert load_json_object(wrapped, context="t") == {"score": 0.5}

    def test_non_json_content_returns_empty_dict(self, caplog):
        with caplog.at_level("WARNING"):
            assert load_json_object("just some prose", context="reranker") == {}
        assert "non-JSON content" in caplog.text

    def test_malformed_embedded_json_returns_empty_dict(self, caplog):
        # Looks like an object (has braces) but is not valid JSON.
        with caplog.at_level("WARNING"):
            assert load_json_object("noise {a: 1, } trailing", context="x") == {}
        assert "malformed JSON" in caplog.text

    def test_json_array_is_rejected_as_non_object(self, caplog):
        with caplog.at_level("WARNING"):
            assert load_json_object("[1, 2, 3]", context="x") == {}
        assert "expected object" in caplog.text


class TestClampFloat:
    def test_passes_through_in_range(self):
        assert clamp_float(0.5) == 0.5

    def test_clamps_to_high_and_low_bounds(self):
        assert clamp_float(5.0) == 1.0
        assert clamp_float(-2.0) == 0.0

    def test_respects_custom_bounds(self):
        assert clamp_float(50, low=0, high=100) == 50.0
        assert clamp_float(500, low=0, high=100) == 100.0

    @pytest.mark.parametrize("bad", [None, "abc", object()])
    def test_non_numeric_returns_default(self, bad):
        assert clamp_float(bad, default=0.3) == 0.3

    def test_numeric_string_is_coerced(self):
        assert clamp_float("0.7") == 0.7


class TestCoerceStr:
    def test_strips_string(self):
        assert coerce_str("  hi  ") == "hi"

    def test_none_returns_default(self):
        assert coerce_str(None, default="fallback") == "fallback"

    def test_non_string_scalar_is_stringified(self):
        assert coerce_str(42) == "42"
        assert coerce_str(True) == "True"
