"""Deterministic visual corpus with declared ground truth (OCR/Vision gates).

Why this module exists
----------------------
Before it, every offline vision test fed the pipeline ``b"\\x89PNG" + b"\\x00"
* 64``: real bytes never reached a decoder, RapidOCR was monkeypatched away in
100% of the free suites, and the only place real pixels ran was the *paid*
OpenRouter gate at the end of the pipeline. A broken ONNX model, a Pillow
decoder regression or a rotated-input crash could not be caught for free.

The corpus builds real PNG/JPEG/PDF bytes at test time and pairs each document
with the ground truth it must yield. Two consumers share it:

* ``tests/test_vision_offline.py`` — free gate. Real RapidOCR + real decoders,
  vision endpoint stubbed. Scores OCR anchor recall per case.
* ``tests/test_vision_live.py`` — paid gate. Same documents, real model, scored
  against the rubric in ``tests/vision_eval.py``.

Design constraints
------------------
* **No system font dependency.** Everything renders through Pillow's bundled
  face (``ImageFont.load_default(size=...)``). The previous fixture asked for
  ``DejaVuSans.ttf`` first, which is absent both on macOS and in
  ``python:3.12-bookworm`` — so the two environments silently diverged on
  whichever font each fell back to. Bundled-only removes that skew.
* **No customer data.** The documents reproduce the *hard parts* of BNP inputs
  (dense tables, low-density schematics, drop-class noise, degraded scans)
  rather than shipping real ones.
* **Anchors are compact-normalised** (``compact()``): OCR word spacing is not
  reproducible, character content is.
"""

from __future__ import annotations

import io
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def compact(value: str) -> str:
    """Uppercase alphanumerics only — OCR-stable comparison form."""
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def font(size: int):
    """Pillow's bundled face at ``size`` — identical on every platform."""
    return ImageFont.load_default(size=size)


def anchor_recall(text: str | None, anchors: tuple[str, ...]) -> tuple[float, list]:
    """Fraction of ``anchors`` present in ``text``; also returns the misses."""
    if not anchors:
        return 1.0, []
    haystack = compact(text or "")
    missing = [a for a in anchors if compact(a) not in haystack]
    return (len(anchors) - len(missing)) / len(anchors), missing


# ---------------------------------------------------------------------------
# Case declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisionCase:
    """One corpus document plus everything it is expected to produce."""

    key: str
    filename: str
    topic: str
    build: Callable[[Path], Path]
    #: Strings real OCR must transcribe (compact-normalised comparison).
    ocr_anchors: tuple[str, ...] = ()
    #: Per-case floor on OCR anchor recall, calibrated against real RapidOCR.
    min_ocr_recall: float = 0.0
    #: True when the pipeline must emit markdown; False when it must refuse.
    expect_ingest: bool = True
    #: Substring the refusal ``reason`` must carry (refusal cases only).
    expect_reason: str | None = None
    #: Classifications accepted from the model (lowercase, live gate).
    expected_classes: frozenset[str] = frozenset()
    #: Anchors the MODEL itself must carry in ``content`` (before the appended
    #: OCR section) — proves comprehension, not transcription of our own OCR.
    semantic_anchors: tuple[str, ...] = ()
    #: Strings whose presence means hallucination.
    forbidden: tuple[str, ...] = ()
    #: Included in the paid live evaluation.
    paid: bool = False
    #: Refusal is expected to happen at the free pre-filter (costs no call).
    free_refusal: bool = False
    notes: str = ""
    tags: frozenset[str] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

_INK = "#111827"
_BLUE = "#1D4ED8"
_GRID = "#64748B"

INVOICE_REFERENCE = "INV-2026-0719"
INVOICE_TOTAL = "12 480 EUR"


def _render_invoice() -> Image.Image:
    image = Image.new("RGB", (1600, 1100), "white")
    draw = ImageDraw.Draw(image)

    draw.text((90, 60), "FACTURE DATACENTER", font=font(70), fill=_BLUE)
    draw.text((92, 165), f"REFERENCE {INVOICE_REFERENCE}", font=font(46), fill=_INK)
    draw.text((92, 235), "CLIENT EQUIPE EXPLOITATION IT", font=font(40), fill=_INK)

    left, top, right, bottom = 90, 360, 1510, 810
    draw.rectangle((left, top, right, bottom), outline=_GRID, width=5)
    for y in (460, 575, 690):
        draw.line((left, y, right, y), fill=_GRID, width=4)
    for x in (850, 1120):
        draw.line((x, top, x, bottom), fill=_GRID, width=4)

    for x, text in ((120, "SERVICE"), (890, "QUANTITE"), (1170, "MONTANT")):
        draw.text((x, 390), text, font=font(36), fill=_INK)

    rows = (
        (500, "HEBERGEMENT SERVEUR", "12", "9 600 EUR"),
        (615, "SAUVEGARDE CHIFFREE", "12", "2 880 EUR"),
    )
    for y, service, quantity, amount in rows:
        draw.text((120, y), service, font=font(35), fill=_INK)
        draw.text((930, y), quantity, font=font(35), fill=_INK)
        draw.text((1180, y), amount, font=font(35), fill=_INK)

    draw.text((830, 860), "TOTAL HT", font=font(46), fill=_INK)
    draw.text((1150, 860), INVOICE_TOTAL, font=font(46), fill=_BLUE)
    draw.text((830, 935), "TVA", font=font(40), fill=_INK)
    draw.text((1150, 935), "2 496 EUR", font=font(40), fill=_INK)
    return image


def build_invoice(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _render_invoice().save(path, format="PNG", optimize=True)
    return path


def build_rotated_invoice(path: Path) -> Path:
    """Same invoice rotated a quarter turn — orientation robustness."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _render_invoice().rotate(90, expand=True).save(path, format="PNG")
    return path


def build_noisy_scan(path: Path) -> Path:
    """The invoice degraded like a fax: speckle plus heavy JPEG artifacts."""
    image = _render_invoice().convert("RGB")
    rng = random.Random(20260724)
    pixels = image.load()
    width, height = image.size
    for _ in range((width * height) // 260):
        x = rng.randrange(width)
        y = rng.randrange(height)
        shade = rng.choice((0, 255))
        pixels[x, y] = (shade, shade, shade)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=22, optimize=False)
    return path


def build_grayscale_memo(path: Path) -> Path:
    """8-bit greyscale ("L") — a decoder mode the RGB happy path never sees."""
    image = Image.new("L", (1400, 900), 250)
    draw = ImageDraw.Draw(image)
    draw.text((80, 70), "NOTE DE SERVICE INTERNE", font=font(62), fill=15)
    draw.text((80, 190), "OBJET MAINTENANCE DATACENTER", font=font(44), fill=25)
    draw.text((80, 270), "FENETRE 2026-08-14 DE 22H00 A 02H00", font=font(42), fill=25)
    draw.text((80, 350), "IMPACT SAUVEGARDE CHIFFREE SUSPENDUE", font=font(42), fill=25)
    draw.text((80, 430), "CONTACT EQUIPE EXPLOITATION IT", font=font(42), fill=25)
    draw.text((80, 540), "CLASSIFICATION INTERNE", font=font(38), fill=60)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def build_cmyk_jpeg(path: Path) -> Path:
    """CMYK JPEG — the classic "works on my RGB fixture" crash source."""
    image = Image.new("RGB", (1400, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.text((80, 80), "BON DE COMMANDE", font=font(64), fill=_INK)
    draw.text((80, 200), "NUMERO BC-2026-0442", font=font(46), fill=_INK)
    draw.text((80, 285), "FOURNISSEUR DATACENTER NORD", font=font(42), fill=_INK)
    draw.text((80, 370), "MONTANT ENGAGE 48 000 EUR", font=font(46), fill=_INK)
    draw.text((80, 460), "VALIDATION DIRECTION FINANCIERE", font=font(40), fill=_INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("CMYK").save(path, format="JPEG", quality=92)
    return path


def build_alpha_chart(path: Path) -> Path:
    """RGBA with real transparency — alpha handling on the decode path."""
    image = Image.new("RGBA", (1300, 900), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    draw.text((70, 60), "DISPONIBILITE MENSUELLE", font=font(58), fill=(17, 24, 39))
    bars = ((160, 520), (330, 610), (500, 470), (670, 640), (840, 700))
    for x, height in bars:
        draw.rectangle((x, 800 - height, x + 110, 800), fill=(29, 78, 216, 235))
    draw.line((120, 800, 1050, 800), fill=(17, 24, 39), width=6)
    labels = ("JANVIER", "FEVRIER", "MARS", "AVRIL", "MAI")
    for (x, _), label in zip(bars, labels):
        draw.text((x - 20, 815), label, font=font(30), fill=(17, 24, 39))
    draw.text(
        (70, 870), "SEUIL CONTRACTUEL 99 SUR 100", font=font(34), fill=(17, 24, 39)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def build_ops_screenshot(path: Path) -> Path:
    """Dark-theme console screenshot: small, low-contrast, UI-shaped text."""
    image = Image.new("RGB", (1500, 950), "#0F172A")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1500, 90), fill="#1E293B")
    draw.text(
        (40, 25), "SUPERVISION INCIDENTS PRODUCTION", font=font(40), fill="#E2E8F0"
    )

    headers = ((50, "TICKET"), (420, "SERVICE"), (860, "SEVERITE"), (1180, "ETAT"))
    for x, label in headers:
        draw.text((x, 140), label, font=font(30), fill="#94A3B8")
    rows = (
        (220, "INC0042318", "PAIEMENT", "MAJEUR", "EN COURS"),
        (300, "INC0042319", "REPORTING", "MINEUR", "RESOLU"),
        (380, "INC0042320", "SAUVEGARDE", "MAJEUR", "ESCALADE"),
    )
    for y, ticket, service, severity, state in rows:
        draw.text((50, y), ticket, font=font(32), fill="#F8FAFC")
        draw.text((420, y), service, font=font(32), fill="#F8FAFC")
        draw.text((860, y), severity, font=font(32), fill="#FCA5A5")
        draw.text((1180, y), state, font=font(32), fill="#86EFAC")
    draw.line((40, 190, 1460, 190), fill="#334155", width=3)
    draw.text((50, 520), "FILE ATTENTE 3 INCIDENTS", font=font(34), fill="#E2E8F0")
    draw.text((50, 590), "ASTREINTE NIVEAU 2 ACTIVEE", font=font(34), fill="#E2E8F0")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    return path


def build_architecture_diagram(path: Path) -> Path:
    """Low text density, high structural meaning — OCR cannot carry this one."""
    image = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 50), "ARCHITECTURE INDEXATION TWIN", font=font(52), fill=_BLUE)

    boxes = (
        (90, 300, 420, 470, "COLLECTE DOCUMENTS"),
        (560, 300, 900, 470, "MOTEUR INDEXATION"),
        (1040, 300, 1400, 470, "BASE MEMGRAPH"),
    )
    for x0, y0, x1, y1, label in boxes:
        draw.rectangle((x0, y0, x1, y1), outline=_INK, width=5, fill="#EFF6FF")
        draw.text((x0 + 24, y0 + 65), label, font=font(30), fill=_INK)
    for x0, x1 in ((420, 560), (900, 1040)):
        draw.line((x0, 385, x1, 385), fill=_INK, width=6)
        draw.polygon(((x1, 385), (x1 - 22, 370), (x1 - 22, 400)), fill=_INK)
    draw.text((430, 560), "FLUX ASYNCHRONE", font=font(32), fill=_INK)
    draw.text((90, 660), "SORTIE GRAPHE DE CONNAISSANCE", font=font(34), fill=_INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    return path


def build_logo(path: Path) -> Path:
    """Brand mark, a handful of glyphs — must die at the free pre-filter."""
    image = Image.new("RGB", (700, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((110, 110, 590, 590), outline="#047857", width=26)
    draw.polygon(((350, 210), (470, 430), (230, 430)), fill="#047857")
    draw.text((300, 460), "TWN", font=font(60), fill="#047857")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def build_signature(path: Path) -> Path:
    """Handwriting-like strokes, zero machine text — free pre-filter refusal."""
    image = Image.new("RGB", (900, 400), "white")
    draw = ImageDraw.Draw(image)
    rng = random.Random(4242)
    x, y = 90.0, 250.0
    points = []
    for step in range(240):
        x += 3.1
        y += rng.uniform(-9, 9) - 22 * (0.5 - abs((step % 60) / 60 - 0.5))
        y = max(110.0, min(330.0, y))
        points.append((x, y))
    draw.line(points, fill="#111827", width=5, joint="curve")
    draw.line((90, 350, 810, 350), fill="#9CA3AF", width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def build_blank_scan(path: Path) -> Path:
    """A blank page scanned with dust — the pathological empty input."""
    image = Image.new("RGB", (1200, 1600), "white")
    rng = random.Random(7)
    pixels = image.load()
    for _ in range(900):
        x = rng.randrange(1200)
        y = rng.randrange(1600)
        shade = rng.randrange(150, 235)
        pixels[x, y] = (shade, shade, shade)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


def build_tiny(path: Path) -> Path:
    """1x1 pixel — degenerate geometry through the whole decode chain."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1, 1), "white").save(path, format="PNG")
    return path


def build_visual_pdf(path: Path) -> Path:
    """A standard PDF whose page embeds a real raster image plus a text layer.

    The invoice is embedded as a genuine ``/DCTDecode`` XObject, so the generic
    PDF tier has something PDFium must actually rasterise and RapidOCR must
    actually read — as opposed to a vector-only page where "the render worked"
    proves nothing about the image path.
    """
    buffer = io.BytesIO()
    _render_invoice().save(buffer, format="JPEG", quality=88)
    jpeg = buffer.getvalue()
    width, height = _render_invoice().size

    text_ops = "\n".join(
        (
            "BT /F1 13 Tf 45 800 Td 18 TL",
            "(Annexe comptable - piece justificative) Tj T*",
            "(Le detail chiffre figure sur l image ci-dessous.) Tj T*",
            "ET",
        )
    )
    draw_ops = "q 500 0 0 340 48 380 cm /Im0 Do Q"
    stream = f"{text_ops}\n{draw_ops}".encode("latin-1", "replace")

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        4: (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 3 0 R >> "
            b"/XObject << /Im0 6 0 R >> >> /Contents 5 0 R >>"
        ),
        5: f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
        6: (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode "
            f"/Length {len(jpeg)} >>\nstream\n".encode() + jpeg + b"\nendstream"
        ),
    }

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"
    xref = len(out)
    total = max(objects) + 1
    out += f"xref\n0 {total}\n".encode()
    out += b"0000000000 65535 f \n"
    for number in range(1, total):
        out += f"{offsets[number]:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    ).encode()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


# ---------------------------------------------------------------------------
# Malformed inputs (anomaly injection — no ground truth, only "never crash")
# ---------------------------------------------------------------------------


def build_truncated_png(path: Path) -> Path:
    """Valid PNG header, stream cut mid-IDAT."""
    buffer = io.BytesIO()
    _render_invoice().save(buffer, format="PNG")
    payload = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload[: len(payload) // 2])
    return path


def build_empty_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def build_bogus_png(path: Path) -> Path:
    """PNG magic bytes followed by garbage — the old fixture's shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 8)
    return path


def build_decompression_bomb(path: Path) -> Path:
    """~100 megapixels of uniform colour: tiny on disk, huge once decoded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10000, 10000), "white").save(
        path, format="PNG", optimize=False, compress_level=9
    )
    return path


MALFORMED_BUILDERS: tuple[tuple[str, str, Callable[[Path], Path]], ...] = (
    ("truncated_png", "truncated.png", build_truncated_png),
    ("empty_file", "empty.png", build_empty_file),
    ("bogus_png", "bogus.png", build_bogus_png),
    ("decompression_bomb", "bomb.png", build_decompression_bomb),
)


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------
#
# ``min_ocr_recall`` floors are CALIBRATED, not guessed: they come from an
# actual RapidOCR run over these exact bytes (scripts/calibrate_vision_corpus.py)
# with headroom for engine-version drift. A case whose real recall sits at 1.0
# is floored below 1.0 on purpose — the gate must catch a collapse, not a
# one-anchor wobble from an onnxruntime bump.
#
# Measured 2026-07-24, rapidocr-onnxruntime on Python 3.12 (floor in brackets):
#
#     invoice_table         1.00 [0.83]     alpha_chart        0.67 [0.33]
#     ops_screenshot        1.00 [0.60]     rotated_invoice    1.00 [0.66]
#     architecture_diagram  0.75 [0.50]     noisy_fax          1.00 [0.50]
#     grayscale_memo        1.00 [0.50]     cmyk_purchase_..   1.00 [0.50]
#
# ``cmyk_purchase_order`` measured 0.00 before the ``_ocr_text_sync`` re-decode
# fix landed: RapidOCR's own file loader returns nothing for CMYK JPEG, and the
# empty result was indistinguishable from "no text", so the pre-filter silently
# discarded a legible document. That case is now this suite's regression guard.
# Misses are expected and intentional where the floor is well below 1.00:
# ``architecture_diagram`` drops "COLLECTE DOCUMENTS", ``alpha_chart`` drops
# "SEUIL CONTRACTUEL" — sparse-text documents are exactly where OCR alone is
# insufficient and the vision model has to carry the meaning.

CORPUS: tuple[VisionCase, ...] = (
    VisionCase(
        key="invoice_table",
        filename="invoice-table.png",
        topic="finance",
        build=build_invoice,
        ocr_anchors=(
            "FACTURE DATACENTER",
            INVOICE_REFERENCE,
            "HEBERGEMENT SERVEUR",
            "SAUVEGARDE CHIFFREE",
            INVOICE_TOTAL,
            "2 496 EUR",
        ),
        min_ocr_recall=0.83,
        expected_classes=frozenset(
            {"table", "invoice", "document", "scanned-document", "form", "receipt"}
        ),
        semantic_anchors=("12 480", "9 600"),
        forbidden=("Lorem ipsum",),
        paid=True,
        tags=frozenset({"dense-text", "table"}),
    ),
    VisionCase(
        key="ops_screenshot",
        filename="ops-screenshot.png",
        topic="it-ops",
        build=build_ops_screenshot,
        ocr_anchors=(
            "SUPERVISION INCIDENTS PRODUCTION",
            "INC0042318",
            "INC0042320",
            "ESCALADE",
            "ASTREINTE NIVEAU 2",
        ),
        min_ocr_recall=0.60,
        expected_classes=frozenset(
            {"screenshot", "table", "dashboard", "user-interface", "document"}
        ),
        semantic_anchors=("INC0042318",),
        forbidden=("Lorem ipsum",),
        paid=True,
        notes="Dark theme, small glyphs: the low-contrast OCR regression canary.",
        tags=frozenset({"screenshot", "low-contrast"}),
    ),
    VisionCase(
        key="architecture_diagram",
        filename="architecture-diagram.png",
        topic="it-ops",
        build=build_architecture_diagram,
        ocr_anchors=(
            "ARCHITECTURE INDEXATION TWIN",
            "COLLECTE DOCUMENTS",
            "MOTEUR INDEXATION",
            "BASE MEMGRAPH",
        ),
        min_ocr_recall=0.50,
        expected_classes=frozenset(
            {"diagram", "flowchart", "schematic", "graph", "chart", "illustration"}
        ),
        semantic_anchors=("MEMGRAPH",),
        forbidden=("Lorem ipsum",),
        paid=True,
        notes="Meaning lives in the arrows: OCR alone cannot carry this case.",
        tags=frozenset({"diagram", "sparse-text"}),
    ),
    VisionCase(
        key="grayscale_memo",
        filename="grayscale-memo.png",
        topic="corporate",
        build=build_grayscale_memo,
        ocr_anchors=(
            "NOTE DE SERVICE INTERNE",
            "MAINTENANCE DATACENTER",
            "2026-08-14",
            "CLASSIFICATION INTERNE",
        ),
        min_ocr_recall=0.50,
        expected_classes=frozenset({"document", "scanned-document", "text", "memo"}),
        tags=frozenset({"mode-L"}),
    ),
    VisionCase(
        key="cmyk_purchase_order",
        filename="cmyk-purchase-order.jpg",
        topic="finance",
        build=build_cmyk_jpeg,
        ocr_anchors=(
            "BON DE COMMANDE",
            "BC-2026-0442",
            "DATACENTER NORD",
            "48 000 EUR",
        ),
        min_ocr_recall=0.50,
        expected_classes=frozenset({"document", "form", "invoice", "scanned-document"}),
        tags=frozenset({"cmyk", "jpeg"}),
    ),
    VisionCase(
        key="alpha_chart",
        filename="alpha-chart.png",
        topic="reporting",
        build=build_alpha_chart,
        ocr_anchors=(
            "DISPONIBILITE MENSUELLE",
            "FEVRIER",
            "SEUIL CONTRACTUEL",
        ),
        min_ocr_recall=0.33,
        expected_classes=frozenset({"chart", "graph", "diagram", "illustration"}),
        tags=frozenset({"rgba", "chart"}),
    ),
    VisionCase(
        key="rotated_invoice",
        filename="rotated-invoice.png",
        topic="finance",
        build=build_rotated_invoice,
        ocr_anchors=(
            "FACTURE DATACENTER",
            INVOICE_REFERENCE,
            INVOICE_TOTAL,
        ),
        min_ocr_recall=0.66,
        expected_classes=frozenset(
            {"table", "invoice", "document", "scanned-document", "form"}
        ),
        notes=(
            "Quarter-turn input. Measured 1.00 — RapidOCR's orientation "
            "classifier recovers it — so the floor asserts the capability "
            "rather than merely 'does not crash'. Kept at 2/3 because "
            "orientation handling is an engine-model behaviour, not a "
            "documented contract."
        ),
        tags=frozenset({"rotated"}),
    ),
    VisionCase(
        key="noisy_fax",
        filename="noisy-fax.jpg",
        topic="finance",
        build=build_noisy_scan,
        ocr_anchors=(
            "FACTURE DATACENTER",
            INVOICE_REFERENCE,
            "HEBERGEMENT SERVEUR",
            INVOICE_TOTAL,
        ),
        min_ocr_recall=0.50,
        expected_classes=frozenset(
            {"scanned-document", "document", "table", "invoice", "form"}
        ),
        notes="Speckle + quality-22 JPEG: degraded but must stay above the floor.",
        tags=frozenset({"degraded", "jpeg"}),
    ),
    VisionCase(
        key="brand_logo",
        filename="brand-logo.png",
        topic="noise",
        build=build_logo,
        ocr_anchors=(),
        expect_ingest=False,
        expect_reason="vision-prefilter",
        free_refusal=True,
        paid=True,
        notes=(
            "The pre-filter's whole economic purpose. Real pixels, real OCR, "
            "and it must cost ZERO model calls."
        ),
        tags=frozenset({"drop-class"}),
    ),
    VisionCase(
        key="handwritten_signature",
        filename="handwritten-signature.png",
        topic="noise",
        build=build_signature,
        ocr_anchors=(),
        expect_ingest=False,
        expect_reason="vision-prefilter",
        free_refusal=True,
        paid=True,
        tags=frozenset({"drop-class"}),
    ),
    VisionCase(
        key="blank_scan",
        filename="blank-scan.png",
        topic="noise",
        build=build_blank_scan,
        ocr_anchors=(),
        expect_ingest=False,
        expect_reason="vision-prefilter",
        free_refusal=True,
        paid=True,
        tags=frozenset({"drop-class", "empty"}),
    ),
    VisionCase(
        key="tiny_pixel",
        filename="tiny-pixel.png",
        topic="noise",
        build=build_tiny,
        ocr_anchors=(),
        expect_ingest=False,
        expect_reason="vision-prefilter",
        free_refusal=True,
        notes="Degenerate geometry; must refuse cleanly, never raise.",
        tags=frozenset({"degenerate"}),
    ),
)

CORPUS_BY_KEY = {case.key: case for case in CORPUS}
PAID_CASES = tuple(case for case in CORPUS if case.paid)


def materialise(case: VisionCase, directory: Path) -> Path:
    """Write ``case`` into ``directory`` and return the path."""
    return case.build(Path(directory) / case.filename)
