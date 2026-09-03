"""Guards of the golden-fixture freezer (``scripts/portability_freeze_fixture.py``).

The freezer ends in ``shutil.rmtree(out_dir)`` and the fixture it writes is the
only artefact proving the v1 bundle format has not moved. Two things therefore
need their own tests, neither of which the fixture itself can catch:

* ``--out`` must never be able to name a directory outside the bundle root — a
  recursive delete is not what ``--force`` ("overwrite an existing fixture")
  promises;
* the procedure schematic must be a real image, so the "the fixture covers the
  procedure file plane" claim cannot silently decay back into opaque bytes.
"""

from __future__ import annotations

import pytest

from tests._repo_only import require_repo_path

require_repo_path("scripts")

from scripts.portability_freeze_fixture import (  # noqa: E402
    UnsafeOutputDir,
    resolve_out_dir,
)
from tests.test_portability._golden import (
    FIXTURES_ROOT,
    GOLDEN_V1_DIR,
    PNG_SIGNATURE,
    golden_png,
    png_chunks,
)


class TestResolveOutDir:
    def test_accepts_a_direct_child_of_the_bundle_root(self):
        assert resolve_out_dir(GOLDEN_V1_DIR) == GOLDEN_V1_DIR.resolve()
        assert (
            resolve_out_dir(FIXTURES_ROOT / "twin-kb-bundle-v2")
            == (FIXTURES_ROOT / "twin-kb-bundle-v2").resolve()
        )

    def test_refuses_the_bundle_root_itself(self):
        with pytest.raises(UnsafeOutputDir, match="not the root itself"):
            resolve_out_dir(FIXTURES_ROOT)

    @pytest.mark.parametrize(
        "relative",
        [
            ".",  # `--out . --force` would erase the checkout
            "..",
            "src",
            "tests",
            "tests/fixtures",  # the bundle root's parent
        ],
    )
    def test_refuses_paths_outside_the_bundle_root(self, relative):
        with pytest.raises(UnsafeOutputDir, match="direct child"):
            resolve_out_dir(FIXTURES_ROOT.parents[2] / relative)

    def test_refuses_a_grandchild(self, tmp_path):
        with pytest.raises(UnsafeOutputDir, match="direct child"):
            resolve_out_dir(GOLDEN_V1_DIR / "memgraph")

    def test_refuses_an_absolute_path_elsewhere(self, tmp_path):
        with pytest.raises(UnsafeOutputDir, match="direct child"):
            resolve_out_dir(tmp_path / "anywhere")

    def test_refuses_a_traversal_that_escapes_the_root(self):
        with pytest.raises(UnsafeOutputDir):
            resolve_out_dir(FIXTURES_ROOT / "twin-kb-bundle-v1" / ".." / "..")


class TestGoldenPng:
    def test_builds_a_structurally_valid_png(self):
        chunks = png_chunks(golden_png())
        tags = [tag for tag, _ in chunks]
        assert tags[0] == "IHDR"
        assert tags[-1] == "IEND"
        assert "IDAT" in tags

    def test_is_deterministic(self):
        assert golden_png() == golden_png()

    def test_refuses_the_signature_followed_by_arbitrary_bytes(self):
        # Exactly the shape the fixture used to carry: a PNG header glued to
        # text. No decoder accepts it, and it must not pass as an image again.
        with pytest.raises(ValueError):
            png_chunks(PNG_SIGNATURE + b"golden-schematic")

    def test_refuses_a_bad_signature(self):
        with pytest.raises(ValueError, match="signature"):
            png_chunks(b"GIF89a" + golden_png()[8:])

    def test_refuses_a_corrupted_chunk(self):
        data = bytearray(golden_png())
        # Flip a byte inside IHDR's payload; the length stays valid so only the
        # CRC can catch it.
        data[16] ^= 0xFF
        with pytest.raises(ValueError, match="CRC"):
            png_chunks(bytes(data))

    def test_refuses_a_truncated_file(self):
        with pytest.raises(ValueError, match="truncated"):
            png_chunks(golden_png()[:-4])

    def test_rejects_non_positive_dimensions(self):
        with pytest.raises(ValueError, match="positive"):
            golden_png(width=0)
