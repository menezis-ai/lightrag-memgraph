"""Canonical JSON for the ``twin-kb-bundle`` format — RFC 8785 (JCS).

Deliberate copy of ``services/twin_catalog/twin_catalog/canonical.py`` (the
catalogue is a separate distribution; the two packages never import each
other). The one difference is intentional: the KB bundle carries floats —
embeddings, anchor confidences, retrieval scores — so this module serialises
finite floats the way RFC 8785 §3.2.2.3 requires (ES6 ``Number.toString``,
shortest round-trip digits), where the catalogue refuses them outright.
NaN/Infinity are still refused: they have no JSON form.

Everything here is pure and deterministic; ``state_hash`` / ``manifest_hash``
are derived from it, so a change in this file is a change of the bundle
contract (bump ``FORMAT_VERSION`` in ``manifest.py``).
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from decimal import Decimal
from typing import Any


class CanonicalisationError(ValueError):
    """The value cannot be canonicalised (NaN, unsupported type, ...)."""


def _nfc(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, dict):
        return {unicodedata.normalize("NFC", str(k)): _nfc(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_nfc(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalisationError("NaN/Infinity are not allowed in a bundle")
        return value
    if value is None or isinstance(value, bool | int):
        return value
    raise CanonicalisationError(f"unsupported type {type(value).__name__}")


def _utf16_units(key: str) -> tuple[int, ...]:
    data = key.encode("utf-16-be")
    return tuple(int.from_bytes(data[i : i + 2], "big") for i in range(0, len(data), 2))


def es6_number(value: float) -> str:
    """ES6 ``Number.prototype.toString`` of a finite float (RFC 8785 §3.2.2.3).

    Python's ``repr`` already yields the shortest round-trip digits; only the
    layout differs (``1e+16`` vs ``10000000000000000``, ``1e-07`` vs ``1e-7``).
    """
    if not math.isfinite(value):
        raise CanonicalisationError("NaN/Infinity are not allowed in a bundle")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    digits_t, exponent = Decimal(repr(abs(value))).as_tuple()[1:]
    digits = "".join(map(str, digits_t)).lstrip("0")
    if not digits:
        return "0"
    trimmed = digits.rstrip("0")
    exponent += len(digits) - len(trimmed)
    digits = trimmed
    k = len(digits)
    n = k + exponent  # value = 0.d1..dk × 10^n
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    e = n - 1
    mantissa = digits if k == 1 else digits[0] + "." + digits[1:]
    return f"{sign}{mantissa}e{'+' if e >= 0 else '-'}{abs(e)}"


def _jcs(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return es6_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(_jcs(v) for v in value) + "]"
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: _utf16_units(kv[0]))
        return (
            "{"
            + ",".join(
                json.dumps(k, ensure_ascii=False, separators=(",", ":")) + ":" + _jcs(v)
                for k, v in items
            )
            + "}"
        )
    raise CanonicalisationError(f"unsupported type {type(value).__name__}")


def jcs_dumps(value: Any) -> str:
    """RFC 8785 serialisation of *value* after NFC normalisation."""
    return _jcs(_nfc(value))


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not allowed in a bundle")


def jcs_loads(text: str) -> Any:
    """``json.loads`` that refuses ``NaN``/``Infinity`` literals."""
    return json.loads(text, parse_constant=_reject_constant)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def jcs_sha256(value: Any) -> str:
    """``sha256`` of the JCS form of *value* (hex, lower case)."""
    return sha256_hex(jcs_dumps(value).encode("utf-8"))
