"""The catalogue admin UI ships a COPY of the Twin KMS charter (tokens.css +
app.css from the WebUI fork) because its Docker build context cannot see
``lightrag_webui_twin/``. The WebUI is the single source of truth: this test
fails as soon as the copy drifts — run ``scripts/sync_catalog_admin_charter.sh``.
It also pins that the catalogue defines no palette of its own: its stylesheet
may only reference charter tokens, never literal colours or font families.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "lightrag_webui_twin" / "src" / "styles"
COPY = ROOT / "services" / "twin_catalog" / "admin-ui" / "src" / "twin"
CATALOG_CSS = ROOT / "services" / "twin_catalog" / "admin-ui" / "src" / "catalog.css"
ADMIN_UI = ROOT / "services" / "twin_catalog" / "admin-ui"
ADMIN_PY = ROOT / "services" / "twin_catalog" / "twin_catalog" / "admin.py"
# tokens.css family → the self-hosted package main.tsx must import.
FONT_PACKAGES = {
    "Spectral": "@fontsource/spectral",
    "Hanken Grotesk": "@fontsource/hanken-grotesk",
    "IBM Plex Mono": "@fontsource/ibm-plex-mono",
}


@pytest.mark.parametrize("name", ["tokens.css", "app.css"])
def test_catalogue_charter_copy_is_identical_to_the_webui(name: str) -> None:
    assert (COPY / name).read_bytes() == (SOURCE / name).read_bytes(), (
        f"{name}: the catalogue copy drifted from lightrag_webui_twin/src/styles — "
        "run scripts/sync_catalog_admin_charter.sh"
    )


def test_catalogue_stylesheet_defines_no_palette_of_its_own() -> None:
    """Layout only: every colour/font must come from the charter tokens."""
    css = re.sub(r"/\*.*?\*/", "", CATALOG_CSS.read_text(), flags=re.S)
    hex_colours = re.findall(r"#[0-9a-fA-F]{3,8}\b", css)
    assert (
        not hex_colours
    ), f"literal colours in catalog.css: {sorted(set(hex_colours))}"
    rgb = re.findall(r"\b(?:rgb|rgba|hsl|hsla)\(", css)
    assert not rgb, "rgb()/hsl() literals in catalog.css — use the charter tokens"
    fonts = re.findall(r"font-family\s*:\s*([^;]+);", css)
    for value in fonts:
        assert "var(--" in value, f"literal font-family in catalog.css: {value!r}"
    assert "@font-face" not in css and "@import" not in css
    stale = re.findall(
        r"^\.(sidebar|login-story|story-\w+|nav-item|app-shell)\b", css, re.M
    )
    assert not stale, f"bespoke shell classes must not come back: {stale}"


_OFF_ORIGIN = re.compile(
    r"""(?:href|src)\s*=\s*["']https?://|@import\s+(?:url\()?["']?https?://|url\(\s*["']?https?://""",
    re.I,
)


def _admin_ui_sources() -> list[Path]:
    files = [ADMIN_UI / "index.html"]
    for pattern in ("*.ts", "*.tsx", "*.css"):
        files.extend((ADMIN_UI / "src").rglob(pattern))
    return sorted(files)


def test_catalogue_admin_ui_fetches_nothing_off_origin() -> None:
    """The admin CSP is ``default-src 'self'`` (no font-src / style-src
    exception) and the bank runtime is offline: every asset — fonts included —
    must ship in the bundle. Review of PR #454 caught a Google Fonts <link>
    that the CSP silently blocked (system-font fallback + console violations).
    """
    csp = re.search(
        r'"Content-Security-Policy":\s*\((.*?)\)', ADMIN_PY.read_text(), re.S
    )
    assert csp and "http" not in csp.group(
        1
    ), "the admin CSP must stay same-origin only"
    for path in _admin_ui_sources():
        hits = _OFF_ORIGIN.findall(_strip_comments(path.read_text()))
        assert not hits, f"{path.relative_to(ROOT)} fetches off-origin: {hits}"


def _strip_comments(source: str) -> str:
    """Block comments (``/* */``, ``<!-- -->``) span lines; ``//`` comments
    never do. Review round 2 of PR #454: a single ``re.S | re.M`` pass let the
    ``//`` branch swallow every line after the first comment, so main.tsx was
    inspected down to its third line and the font imports escaped the check.
    """
    text = re.sub(r"/\*.*?\*/|<!--.*?-->", "", source, flags=re.S)
    return re.sub(r"^\s*//[^\r\n]*$", "", text, flags=re.M)


def test_comment_stripper_keeps_code_after_a_line_comment() -> None:
    """The off-origin guard is only as good as its stripper: an external asset
    placed after a ``//`` comment must still be seen."""
    sample = (
        "// self-hosted fonts\n"
        'import "@fontsource/spectral/400.css";\n'
        "/* block\n comment */\n"
        'const bad = "url(https://fonts.googleapis.com/x)";\n'
    )
    stripped = _strip_comments(sample)
    assert "@fontsource/spectral" in stripped
    assert "block" not in stripped and "self-hosted" not in stripped
    assert _OFF_ORIGIN.findall(stripped) == ["url(https://"]


def test_catalogue_admin_ui_self_hosts_every_charter_font() -> None:
    tokens = (COPY / "tokens.css").read_text()
    families = {
        family
        for var in ("--serif", "--font-sans", "--font-mono")
        for family in re.findall(rf'{var}\s*:\s*"([^"]+)"', tokens)
    }
    assert families == set(FONT_PACKAGES), f"tokens.css families changed: {families}"
    main = (ADMIN_UI / "src" / "main.tsx").read_text()
    deps = (ADMIN_UI / "package.json").read_text()
    for family, package in FONT_PACKAGES.items():
        assert (
            f'import "{package}/' in main
        ), f"{family}: main.tsx must import {package}"
        assert f'"{package}"' in deps, f"{package} missing from admin-ui/package.json"
