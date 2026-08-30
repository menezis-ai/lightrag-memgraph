import pytest

from twindb_lightrag_memgraph._constants import upload_relative_path_context
from twindb_lightrag_memgraph._upload_paths import (
    canonical_upload_file_name,
    display_upload_file_path,
    normalize_relative_upload_path,
    relative_path_from_storage_name,
)
from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage


def test_nested_paths_are_distinct_lossless_storage_identities():
    first = canonical_upload_file_name("team-a/report.pdf")
    second = canonical_upload_file_name("team-b/report.pdf")

    assert first != second
    assert first.endswith(".pdf")
    assert second.endswith(".pdf")
    assert display_upload_file_path(first) == "team-a/report.pdf"
    assert display_upload_file_path(second) == "team-b/report.pdf"


@pytest.mark.parametrize(
    "path",
    [
        "../secret.txt",
        "root/../secret.txt",
        "/absolute.txt",
        "root\\windows.txt",
        "root//empty.txt",
        "root/./dot.txt",
        "root/line\nfeed.txt",
    ],
)
def test_relative_path_rejects_traversal_and_ambiguous_separators(path):
    with pytest.raises(ValueError):
        normalize_relative_upload_path(path)


def test_flat_upload_keeps_backward_compatible_file_name():
    assert canonical_upload_file_name("report.pdf") == "report.pdf"


def test_malformed_storage_token_is_not_decoded():
    assert relative_path_from_storage_name("twinrel_!!!!.pdf") is None


def test_docstatus_binds_only_matching_canonical_relative_path():
    relative_path = "root/team/report.pdf"
    props = {"file_path": canonical_upload_file_name(relative_path), "metadata": {}}
    with upload_relative_path_context(relative_path):
        MemgraphDocStatusStorage._attach_active_relative_path(props)
    assert '"relative_path": "root/team/report.pdf"' in props["metadata"]

    with upload_relative_path_context(relative_path):
        with pytest.raises(ValueError, match="does not match"):
            MemgraphDocStatusStorage._attach_active_relative_path(
                {"file_path": "other.pdf"}
            )
