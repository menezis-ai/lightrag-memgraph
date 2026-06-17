"""Unit tests for DocStatus pagination bounds clamping.

No Memgraph required — exercises the pure ``_normalize_pagination`` helper.
"""

from twindb_lightrag_memgraph._constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage

_normalize = MemgraphDocStatusStorage._normalize_pagination


class TestNormalizePagination:
    def test_first_page(self):
        assert _normalize(1, 50) == (0, 50)

    def test_second_page_skip(self):
        assert _normalize(3, 20) == (40, 20)

    def test_zero_page_clamped_to_first(self):
        # page=0 would otherwise produce a negative SKIP → Cypher error.
        skip, limit = _normalize(0, 50)
        assert skip == 0
        assert limit == 50

    def test_negative_page_clamped_to_first(self):
        skip, _ = _normalize(-5, 50)
        assert skip == 0

    def test_page_size_capped(self):
        _, limit = _normalize(1, 10_000_000)
        assert limit == MAX_PAGE_SIZE

    def test_zero_page_size_clamped_to_one(self):
        _, limit = _normalize(1, 0)
        assert limit == 1

    def test_negative_page_size_clamped_to_one(self):
        _, limit = _normalize(1, -10)
        assert limit == 1

    def test_non_int_page_falls_back(self):
        skip, limit = _normalize("abc", "xyz")
        assert skip == 0
        assert limit == DEFAULT_PAGE_SIZE

    def test_none_inputs_fall_back(self):
        skip, limit = _normalize(None, None)
        assert skip == 0
        assert limit == DEFAULT_PAGE_SIZE
