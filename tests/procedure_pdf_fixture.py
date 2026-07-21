"""Synthetic procedure-PDF fixture (PROCEDURE-PROFILE-PLAN.md).

The real BNP procedure PDFs never leave the bank, so CI runs against a
synthetic reproduction of the "IT Group" template markers calibrated on the
photographed ITG0162/ITG0160 samples (2026-07-20): metadata cover page with
the ``ITG\\d{4}`` reference, ``Schematic:`` page headings, mirrored §4 task
sections, "Classification : Internal" footers.

:func:`build_pdf` emits a dependency-free minimal PDF (raw objects, Helvetica
``Tj`` text operators) whose text layer pypdf extracts faithfully — enough
for the deterministic detection and page-location logic, which is exactly
what the fixture must exercise. Rendering fidelity is NOT the point (the
vision passes are monkeypatched in unit tests; the real render path is
covered by the pypdfium2-marked tests and the BNP recette).
"""

from __future__ import annotations

from collections.abc import Sequence


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(pages: Sequence[Sequence[str]]) -> bytes:
    """Build a minimal N-page PDF; each page is a sequence of text lines."""
    count = len(pages)
    objects: dict[int, bytes] = {}
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(count))
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode("ascii")
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for i, lines in enumerate(pages):
        page_obj = 4 + 2 * i
        content_obj = 5 + 2 * i
        ops = ["BT /F1 11 Tf 50 780 Td 14 TL"]
        ops.extend(f"({_escape(line)}) Tj T*" for line in lines)
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1", "replace")
        objects[page_obj] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_obj} 0 R >>"
        ).encode("ascii")
        objects[content_obj] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode("ascii") + objects[num] + b"\nendobj\n"
    xref_pos = len(out)
    total = max(objects) + 1
    out += f"xref\n0 {total}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for num in range(1, total):
        out += f"{offsets[num]:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {total} /Root 1 0 R >>\n" f"startxref\n{xref_pos}\n%%EOF"
    ).encode("ascii")
    return bytes(out)


#: Template reproduction: cover metadata, TOC, two schematics, §4 mirror,
#: closing page — the marker set the detection and page-location code keys on.
PROCEDURE_PAGES: tuple[tuple[str, ...], ...] = (
    (
        "BNP PARIBAS",
        "Manage Production Incidents Procedure",
        "Owner entity * IT Group Production [192 001]",
        "Level * Level 2",
        "Procedure type * 4- Operational procedures",
        "Classification rule * Internal",
        "Reference * ITG0162-Manage_Production_Incidents_Procedure",
        "IT GROUP    Classification : Internal    1/6",
    ),
    (
        "Version * V.1.0",
        "Affiliated parent procedure(s): ITG0160-Manage_IT_Operational_Changes",
        "TABLE OF CONTENTS",
        "3.2. Incident Management Process overview",
        "4. Process activities",
        "IT GROUP    Classification : Internal    2/6",
    ),
    (
        "3.2. Incident Management Process overview",
        "Schematic: Design the Monitoring, Monitor the components and Open"
        " the incident",
        "Legend  Model Version January 2025",
        "Suppliers  Inputs  Process Tasks  Outputs  Clients / Customers",
        "T1.1 - Define the components to be supervised  CTO  Event Designer",
        "IT GROUP    Classification : Internal    3/6",
    ),
    (
        "Schematic: Qualify the incident",
        "T4.1 - Categorize and Enrich  Incident Manager  L1 Support",
        "Trigrams of processes in the above flowcharts:",
        "CHG Manage IT Operational Changes",
        "CONF Manage Configurations",
        "IT GROUP    Classification : Internal    4/6",
    ),
    (
        "4. Process activities",
        "4.4. Activity: Qualify the incident",
        "4.4.1.Task: Categorize and enrich",
        "The L1 Support enriches the incident ticket with as much information"
        " as possible.",
        "IT GROUP    Classification : Internal    5/6",
    ),
    (
        "7. References used in this procedure",
        "- END OF THE DOCUMENT -",
        "IT GROUP    Classification : Internal    6/6",
    ),
)

#: Indexes (0-based) of the pages carrying a ``Schematic:`` heading above.
PROCEDURE_SCHEMATIC_PAGES = (2, 3)


def build_procedure_pdf() -> bytes:
    """The canonical procedure fixture (detection positive, 2 schematics)."""
    return build_pdf(PROCEDURE_PAGES)


def build_plain_pdf() -> bytes:
    """A non-procedure document (detection negative)."""
    return build_pdf(
        (
            ("Quarterly infrastructure report", "Prepared by the ops team"),
            ("Capacity figures and trends", "Nothing procedural here"),
        )
    )


def build_textonly_procedure_pdf() -> bytes:
    """A procedure document without any Schematic: page."""
    return build_pdf(
        (
            (
                "Manage Something Procedure",
                "Reference * ITG9999-Manage_Something_Procedure",
                "Level * Level 2",
                "Procedure type * 4- Operational procedures",
            ),
            ("4. Process activities", "4.1. Activity: Do the thing"),
        )
    )
