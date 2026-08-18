"""Deterministic visual documents for the live OCR/Vision CI gate.

The fixtures contain no customer data.  They deliberately reproduce the
hard parts of the production inputs instead of shipping screenshots from BNP:

* a high-resolution French invoice with a table and stable OCR anchors;
* an IT Group level-2 procedure with a real PDF text layer and one vector
  process schematic carrying task boxes, roles, arrows and a condition.

Both documents are generated at test time so the semantic source stays easy to
review in Git while RapidOCR, PDFium and the Vision endpoint still receive real
PNG/PDF bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

INVOICE_REFERENCE = "INV-2026-0719"
INVOICE_TOTAL = "12 480 EUR"
PROCEDURE_TASK_IDS = frozenset({"T1.1", "T2.1", "T3.1"})


def _font(size: int, *, bold: bool = False):
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def write_invoice_png(path: Path) -> Path:
    """Write a legible table-like invoice used by both OCR and Vision."""
    image = Image.new("RGB", (1600, 1050), "white")
    draw = ImageDraw.Draw(image)
    ink = "#111827"
    blue = "#1D4ED8"
    grid = "#64748B"

    draw.text((90, 65), "FACTURE DATACENTER", font=_font(72, bold=True), fill=blue)
    draw.text(
        (92, 165),
        f"REFERENCE : {INVOICE_REFERENCE}",
        font=_font(45, bold=True),
        fill=ink,
    )
    draw.text(
        (92, 230),
        "CLIENT : EQUIPE EXPLOITATION IT",
        font=_font(40),
        fill=ink,
    )

    left, top, right, bottom = 90, 360, 1510, 810
    draw.rectangle((left, top, right, bottom), outline=grid, width=5)
    for y in (460, 575, 690):
        draw.line((left, y, right, y), fill=grid, width=4)
    for x in (850, 1120):
        draw.line((x, top, x, bottom), fill=grid, width=4)

    headers = ((120, "SERVICE"), (890, "QUANTITE"), (1170, "MONTANT"))
    for x, text in headers:
        draw.text((x, 385), text, font=_font(36, bold=True), fill=ink)

    rows = (
        (500, "HEBERGEMENT SERVEUR", "12", "9 600 EUR"),
        (615, "SAUVEGARDE CHIFFREE", "12", "2 880 EUR"),
    )
    for y, service, quantity, amount in rows:
        draw.text((120, y), service, font=_font(35), fill=ink)
        draw.text((930, y), quantity, font=_font(35), fill=ink)
        draw.text((1180, y), amount, font=_font(35), fill=ink)

    draw.text((830, 855), "TOTAL HT :", font=_font(46, bold=True), fill=ink)
    draw.text((1150, 855), INVOICE_TOTAL, font=_font(46, bold=True), fill=blue)
    draw.text((830, 925), "TVA :", font=_font(38), fill=ink)
    draw.text((1150, 925), "2 496 EUR", font=_font(38), fill=ink)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    return path


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _text_stream(lines: Iterable[str], *, size: int = 15) -> bytes:
    ops = [f"BT /F1 {size} Tf 45 790 Td 25 TL"]
    ops.extend(f"({_escape_pdf_text(line)}) Tj T*" for line in lines)
    ops.append("ET")
    return "\n".join(ops).encode("latin-1", "replace")


def _positioned_text(x: int, y: int, text: str, *, size: int = 10) -> str:
    return f"BT /F1 {size} Tf {x} {y} Td ({_escape_pdf_text(text)}) Tj ET"


def _schematic_stream() -> bytes:
    ops = [
        "0 0 0 RG 1.5 w",
        _positioned_text(
            35, 800, "Schematic: Qualify and resolve the incident", size=18
        ),
        _positioned_text(
            35, 765, "SUPPLIERS / INPUTS / PROCESS TASKS / OUTPUTS / CLIENTS", size=10
        ),
    ]
    boxes = (
        (25, 455, 165, 230, "T1.1 - Detect alert", "Monitoring", "Event Designer"),
        (
            215,
            455,
            165,
            230,
            "T2.1 - Qualify incident",
            "Incident Manager",
            "L1 Support",
        ),
        (
            405,
            455,
            165,
            230,
            "T3.1 - Resolve incident",
            "L2 Support",
            "Application Team",
        ),
    )
    for x, y, width, height, title, responsible, actors in boxes:
        ops.extend(
            (
                "0.92 0.96 1 rg",
                f"{x} {y} {width} {height} re B",
                "0 0 0 rg",
                _positioned_text(x + 9, y + 195, title, size=10),
                "0.78 0.93 0.80 rg",
                f"{x + 8} {y + 115} {width - 16} 43 re B",
                "0 0 0 rg",
                _positioned_text(
                    x + 14, y + 133, f"Responsible: {responsible}", size=8
                ),
                "0.80 0.88 1 rg",
                f"{x + 8} {y + 55} {width - 16} 43 re B",
                "0 0 0 rg",
                _positioned_text(x + 14, y + 73, f"Actors: {actors}", size=8),
            )
        )

    # Directional links between the three task boxes.
    ops.extend(
        (
            "0 0 0 RG 2 w",
            "190 570 m 215 570 l S",
            "208 575 m 215 570 l 208 565 l S",
            "380 570 m 405 570 l S",
            "398 575 m 405 570 l 398 565 l S",
            # The condition and cross-procedure link are deliberately inside
            # T2.1 so both vision passes must attribute them unambiguously.
            "0.75 0 0 rg",
            _positioned_text(225, 485, "Condition: If major incident", size=7),
            "0 0 0 rg",
            _positioned_text(225, 468, "Linked procedure:", size=6),
            "0 0.45 0 rg",
            "322 455 48 24 re B",
            "1 1 1 rg",
            _positioned_text(331, 463, "CONF", size=8),
            "0 0 0 rg",
            _positioned_text(
                35, 335, "Input: monitoring alert and affected service", size=11
            ),
            _positioned_text(
                35,
                310,
                "Output: qualified incident ticket and restored service",
                size=11,
            ),
            _positioned_text(35, 285, "Client: IT operations", size=11),
        )
    )
    return "\n".join(ops).encode("latin-1", "replace")


def _assemble_pdf(streams: tuple[bytes, ...]) -> bytes:
    count = len(streams)
    objects: dict[int, bytes] = {}
    kids = " ".join(f"{4 + 2 * index} 0 R" for index in range(count))
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode()
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for index, stream in enumerate(streams):
        page_obj = 4 + 2 * index
        content_obj = page_obj + 1
        objects[page_obj] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_obj} 0 R >>"
        ).encode()
        objects[content_obj] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )

    output = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number in sorted(objects):
        offsets[number] = len(output)
        output += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"
    xref = len(output)
    total = max(objects) + 1
    output += f"xref\n0 {total}\n".encode()
    output += b"0000000000 65535 f \n"
    for number in range(1, total):
        output += f"{offsets[number]:010d} 00000 n \n".encode()
    output += (
        f"trailer\n<< /Size {total} /Root 1 0 R >>\n" f"startxref\n{xref}\n%%EOF"
    ).encode()
    return bytes(output)


def write_visual_procedure_pdf(path: Path) -> Path:
    """Write a detected procedure with one vector schematic and text layer."""
    cover = _text_stream(
        (
            "IT GROUP - Manage Production Incidents Procedure",
            "Reference * ITG0420-Manage_Production_Incidents_Procedure",
            "Level * Level 2",
            "Procedure type * 4- Operational procedures",
            "Classification : Internal",
        )
    )
    body = _text_stream(
        (
            "4. Process activities",
            "T1.1 Detect alert - Responsible: Monitoring - Actor: Event Designer",
            "Input: monitoring alert - Output: incident candidate",
            "T2.1 Qualify incident - Responsible: Incident Manager - Actor: L1 Support",
            "Flow: T1.1 Detect alert then T2.1 Qualify incident then T3.1 Resolve incident",
            "Transition T2.1 to T3.1 condition: If major incident",
            "T2.1 linked procedure: CONF Manage Configurations",
            "T3.1 Resolve incident - Responsible: L2 Support - Actor: Application Team",
            "Output: restored service and qualified incident ticket",
        ),
        size=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_assemble_pdf((cover, body, _schematic_stream())))
    return path
