"""Adversarial contract for the Cypher-identifier shield and its resolvers.

``_constants`` is the canonical validation point for every value that reaches
Cypher as a *label / database / relationship type* — the positions that cannot
use ``$param`` binding and therefore land in the query by f-string
interpolation. A value that gets past ``validate_identifier`` is a value that
gets concatenated into a query.

Written 2026-07-25 from a mutmut campaign: 24 mutants of this module survived
the existing suite. Two distinct classes are covered here.

1. **Behaviour the mutants exposed as unpinned** — the alias/precedence chains
   in ``resolve_workspace`` and ``default_twin_folder``, and the error message
   that tells an operator *which* identifier was rejected.
2. **A defect the mutants did NOT expose**, found by probing the validator
   directly while triaging them: ``^\\w+$`` accepted ``"workspace\\n"``, because
   Python's ``$`` also matches just before a trailing newline. Exactly that one
   shape got through -- a single trailing newline. It was not statement
   injection (nothing may follow the newline), but it minted a second,
   visually identical namespace. Now matched with ``re.fullmatch``; pinned
   below so the contract cannot regress to ``^``/``$``.
"""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph import _constants
from twindb_lightrag_memgraph._constants import (
    DEFAULT_WORKSPACE,
    default_twin_folder,
    purge_llm_cache_on_failed_enabled,
    resolve_workspace,
    validate_identifier,
)

_ENV_KEYS = (
    _constants.MEMGRAPH_WORKSPACE_ENV,
    _constants.WORKSPACE_ENV,
    _constants.TWIN_DEFAULT_FOLDER_ENV,
)


@pytest.fixture(autouse=True)
def _clear_workspace_env(monkeypatch):
    """Neutral environment: these resolvers read process env directly."""
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


class TestValidateIdentifierAccepts:
    @pytest.mark.parametrize(
        "value", ["a", "Z", "0", "_", "base", "ws_1", "A_b_9", "_leading", "x" * 200]
    )
    def test_alphanumeric_and_underscore_pass_through_unchanged(self, value):
        assert validate_identifier(value) == value


class TestValidateIdentifierRejects:
    """Every rejection here is a character that would otherwise reach Cypher."""

    @pytest.mark.parametrize(
        ("value", "why"),
        [
            ("back`tick", "backtick closes a quoted label"),
            ("a`` OR 1=1 --", "backtick break-out attempt"),
            ("single'quote", "string delimiter"),
            ('double"quote', "string delimiter"),
            ("semi;colon", "statement separator"),
            ("with space", "clause separator"),
            ("dash-dash", "comment lead-in"),
            ("brace{}", "map literal"),
            ("paren()", "call syntax"),
            ("bracket[]", "index syntax"),
            ("dollar$param", "parameter sigil"),
            ("dot.walk", "property access"),
            ("colon:Label", "label sigil"),
            ("star*", "wildcard"),
            ("percent%", "wildcard"),
            ("slash/", "path separator"),
            ("back\\slash", "escape lead-in"),
            ("café", "non-ASCII: re.ASCII keeps \\w to [a-zA-Z0-9_]"),
            ("Ω", "non-ASCII"),
            ("a\tb", "tab"),
            ("a\rb", "carriage return"),
            ("a\x00b", "NUL"),
            ("a\x1bb", "escape byte"),
        ],
    )
    def test_unsafe_characters_raise(self, value, why):
        with pytest.raises(ValueError):
            validate_identifier(value)

    @pytest.mark.parametrize("value", ["", None])
    def test_empty_and_none_raise(self, value):
        with pytest.raises(ValueError):
            validate_identifier(value)

    @pytest.mark.parametrize(
        "value",
        [
            "workspace\n",  # the \A..\Z regression case
            "workspace\n\n",
            "\nworkspace",
            "\n",
            "workspace\nMATCH (n) DETACH DELETE n",
            "ws\r\n",
        ],
    )
    def test_newlines_raise_anywhere_including_a_single_trailing_one(self, value):
        """Guards the ``re.fullmatch`` contract.

        Only ONE of these got through the old ``^\\w+$``: ``"workspace\\n"``,
        returned intact and minting a distinct label. The rest were already
        rejected -- ``$`` tolerates a newline only as the very last character,
        so ``"workspace\\n\\n"`` never passed. They are kept as regression
        cover; the first case is the one that fails if someone reverts to
        ``^``/``$``.
        """
        with pytest.raises(ValueError):
            validate_identifier(value)


class TestValidateIdentifierErrorMessage:
    """The message names the rejected field — that is its operational job."""

    def test_message_carries_the_caller_supplied_name(self):
        with pytest.raises(ValueError) as excinfo:
            validate_identifier("bad;value", "workspace")
        # Anchored, not `"workspace" in message`: a substring check also passes
        # for a padded name like "XXworkspaceXX", so it pins nothing.
        assert str(excinfo.value).startswith("Invalid workspace:")

    def test_message_defaults_to_the_generic_field_name(self):
        with pytest.raises(ValueError) as excinfo:
            validate_identifier("bad;value")
        message = str(excinfo.value)
        # Anchored on the exact rendered prefix, not `"identifier" in message`:
        # a substring check also passes for a padded default like
        # "XXidentifierXX", so it would not pin the value at all.
        assert message.startswith("Invalid identifier:")
        assert "IDENTIFIER" not in message

    def test_message_is_non_empty_and_quotes_the_offending_value(self):
        with pytest.raises(ValueError) as excinfo:
            validate_identifier("bad;value", "database")
        message = str(excinfo.value)
        assert message and message != "None"
        assert repr("bad;value") in message


class TestResolveWorkspace:
    """Alias chain: MEMGRAPH_WORKSPACE > WORKSPACE > TWIN_DEFAULT_FOLDER > base."""

    def test_defaults_when_nothing_is_set(self):
        assert resolve_workspace() == DEFAULT_WORKSPACE

    @pytest.mark.parametrize("key", _ENV_KEYS)
    def test_each_alias_is_honoured_on_its_own(self, key, monkeypatch):
        monkeypatch.setenv(key, "solo_ws")
        assert resolve_workspace() == "solo_ws"

    def test_memgraph_workspace_wins_over_the_other_two(self, monkeypatch):
        monkeypatch.setenv(_constants.MEMGRAPH_WORKSPACE_ENV, "first")
        monkeypatch.setenv(_constants.WORKSPACE_ENV, "second")
        monkeypatch.setenv(_constants.TWIN_DEFAULT_FOLDER_ENV, "third")
        assert resolve_workspace() == "first"

    def test_workspace_wins_over_twin_default_folder(self, monkeypatch):
        monkeypatch.setenv(_constants.WORKSPACE_ENV, "second")
        monkeypatch.setenv(_constants.TWIN_DEFAULT_FOLDER_ENV, "third")
        assert resolve_workspace() == "second"

    @pytest.mark.parametrize("blank", ["", "   ", "\t", "\n"])
    def test_blank_value_falls_through_to_the_next_alias(self, blank, monkeypatch):
        monkeypatch.setenv(_constants.MEMGRAPH_WORKSPACE_ENV, blank)
        monkeypatch.setenv(_constants.WORKSPACE_ENV, "fallback_ws")
        assert resolve_workspace() == "fallback_ws"

    def test_surrounding_whitespace_is_stripped_before_validation(self, monkeypatch):
        monkeypatch.setenv(_constants.MEMGRAPH_WORKSPACE_ENV, "  padded_ws \n")
        assert resolve_workspace() == "padded_ws"

    def test_unsafe_workspace_raises_rather_than_falling_back(self, monkeypatch):
        """A hostile workspace must be a hard error, never a silent default."""
        monkeypatch.setenv(_constants.MEMGRAPH_WORKSPACE_ENV, "ws`; MATCH (n)")
        with pytest.raises(ValueError) as excinfo:
            resolve_workspace()
        # Anchored: resolve_workspace must label the failure "workspace", and a
        # substring check would accept a padded "XXworkspaceXX" just as happily.
        assert str(excinfo.value).startswith("Invalid workspace:")


class TestDefaultTwinFolder:
    """Same shape, but a shorter chain and a *forgiving* failure mode."""

    def test_defaults_when_nothing_is_set(self):
        assert default_twin_folder() == "default"

    def test_twin_default_folder_is_preferred(self, monkeypatch):
        monkeypatch.setenv(_constants.TWIN_DEFAULT_FOLDER_ENV, "twin_folder")
        monkeypatch.setenv(_constants.WORKSPACE_ENV, "ws_folder")
        assert default_twin_folder() == "twin_folder"

    def test_workspace_is_the_fallback_alias(self, monkeypatch):
        monkeypatch.setenv(_constants.WORKSPACE_ENV, "ws_folder")
        assert default_twin_folder() == "ws_folder"

    def test_memgraph_workspace_is_not_part_of_this_chain(self, monkeypatch):
        """Unlike resolve_workspace, MEMGRAPH_WORKSPACE is deliberately absent."""
        monkeypatch.setenv(_constants.MEMGRAPH_WORKSPACE_ENV, "ignored_here")
        assert default_twin_folder() == "default"

    def test_empty_twin_folder_falls_through_to_workspace(self, monkeypatch):
        monkeypatch.setenv(_constants.TWIN_DEFAULT_FOLDER_ENV, "")
        monkeypatch.setenv(_constants.WORKSPACE_ENV, "ws_folder")
        assert default_twin_folder() == "ws_folder"

    @pytest.mark.parametrize("blank", ["   ", "\n", "\t"])
    def test_whitespace_only_twin_folder_falls_through_to_workspace(
        self, blank, monkeypatch
    ):
        """Regression: a blank alias must not swallow the next one.

        This used to return "default". The ``or`` chain ran on RAW env values
        and stripped only the winner, so a whitespace-only TWIN_DEFAULT_FOLDER
        was truthy, captured the chain, stripped to empty and failed validation
        — silently ignoring a valid WORKSPACE. Since this value routes folder
        membership reads and writes, that is a wrong-folder risk, not a
        cosmetic one. Now normalized per alias, like ``resolve_workspace``.
        """
        monkeypatch.setenv(_constants.TWIN_DEFAULT_FOLDER_ENV, blank)
        monkeypatch.setenv(_constants.WORKSPACE_ENV, "ws_folder")
        assert default_twin_folder() == "ws_folder"

    def test_unsafe_alias_does_not_promote_the_next_one(self, monkeypatch):
        """A hostile value fails closed instead of handing over to WORKSPACE.

        Falling through here would let a bad TWIN_DEFAULT_FOLDER silently
        change which source of truth wins, which is a different surprise.
        """
        monkeypatch.setenv(_constants.TWIN_DEFAULT_FOLDER_ENV, "bad;folder")
        monkeypatch.setenv(_constants.WORKSPACE_ENV, "ws_folder")
        assert default_twin_folder() == "default"

    def test_surrounding_whitespace_is_stripped(self, monkeypatch):
        monkeypatch.setenv(_constants.TWIN_DEFAULT_FOLDER_ENV, "  padded \n")
        assert default_twin_folder() == "padded"

    @pytest.mark.parametrize(
        "hostile", ["bad;folder", "back`tick", "a b", "folder\n\nMATCH"]
    )
    def test_unsafe_folder_degrades_to_default_instead_of_raising(
        self, hostile, monkeypatch
    ):
        """Contrast with resolve_workspace: this one swallows ValueError.

        It is a fallback used where no request scope exists, so it must never
        raise into a caller — but it must also never return the hostile value.
        """
        monkeypatch.setenv(_constants.TWIN_DEFAULT_FOLDER_ENV, hostile)
        assert default_twin_folder() == "default"


class TestPurgeLlmCacheFlag:
    def test_defaults_to_enabled_when_unset(self, monkeypatch):
        monkeypatch.delenv(_constants.TWIN_PURGE_LLM_CACHE_ON_FAILED_ENV, raising=False)
        assert purge_llm_cache_on_failed_enabled() is True

    @pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", " false "])
    def test_recognised_false_values_disable_it(self, raw, monkeypatch):
        monkeypatch.setenv(_constants.TWIN_PURGE_LLM_CACHE_ON_FAILED_ENV, raw)
        assert purge_llm_cache_on_failed_enabled() is False

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", "anything", ""])
    def test_everything_else_leaves_it_enabled(self, raw, monkeypatch):
        monkeypatch.setenv(_constants.TWIN_PURGE_LLM_CACHE_ON_FAILED_ENV, raw)
        assert purge_llm_cache_on_failed_enabled() is True
