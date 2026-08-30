"""
PII detection + masking - the query is *masked*, never blocked.

Regex first (fast, zero-dependency). If `presidio-analyzer` happens to be
installed it is used as an additional pass, but it is never required.

Returns the masked text plus the list of entity types found (for state flags
and the dashboard).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)[\s.\-]?)?\d{3,5}[\s.\-]?\d{3,4}(?:[\s.\-]?\d{2,4})?(?!\d)"
)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_AADHAAR = re.compile(r"(?<![\d\-])\d{4}[ \-]?\d{4}[ \-]?\d{4}(?![ \-]?\d)")
_CREDIT = re.compile(r"(?<![\d\-])(?:\d[ \-]?){12,18}\d(?![ \-]?\d)")
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_PASSPORT = re.compile(r"\b[A-PR-WYa-pr-wy][0-9]{7}\b")


def _luhn_ok(digits: str) -> bool:
    d = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    checksum = 0
    parity = len(d) % 2
    for i, n in enumerate(d):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        checksum += n
    return checksum % 10 == 0


@dataclass
class PIIResult:
    text: str                                 # masked text
    original: str
    entities: List[str] = field(default_factory=list)   # e.g. ["EMAIL", "PHONE"]
    spans: List[Tuple[str, str]] = field(default_factory=list)  # (type, original_value)

    @property
    def found(self) -> bool:
        return bool(self.entities)

    def flags(self) -> List[str]:
        return [f"pii:{e}" for e in self.entities]


def _apply(pattern: re.Pattern, label: str, text: str, found: Dict[str, List[str]], validator=None) -> str:
    def repl(m: re.Match) -> str:
        val = m.group(0)
        if validator and not validator(val):
            return val
        if len(re.sub(r"\D", "", val)) < 4 and label in {"PHONE", "CREDIT_CARD", "SSN", "AADHAAR"}:
            return val
        found.setdefault(label, []).append(val)
        return f"[{label}]"

    return pattern.sub(repl, text)


def mask_pii(text: str) -> PIIResult:
    original = text or ""
    found: Dict[str, List[str]] = {}
    masked = original

    masked = _apply(_EMAIL, "EMAIL", masked, found)
    masked = _apply(_SSN, "SSN", masked, found)
    masked = _apply(_CREDIT, "CREDIT_CARD", masked, found, validator=_luhn_ok)
    masked = _apply(_AADHAAR, "AADHAAR", masked, found)
    masked = _apply(_IPV4, "IP_ADDRESS", masked, found)
    masked = _apply(_PASSPORT, "PASSPORT", masked, found)
    masked = _apply(_PHONE, "PHONE", masked, found)

    # optional presidio pass (best effort, additive)
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore

        analyzer = _get_presidio()
        results = analyzer.analyze(text=masked, language="en")
        # sort descending so index math stays valid while we splice
        for r in sorted(results, key=lambda x: x.start, reverse=True):
            if r.score < 0.6:
                continue
            label = r.entity_type
            value = masked[r.start : r.end]
            if value.startswith("[") and value.endswith("]"):
                continue
            found.setdefault(label, []).append(value)
            masked = masked[: r.start] + f"[{label}]" + masked[r.end :]
    except Exception:
        pass

    spans = [(lbl, v) for lbl, vals in found.items() for v in vals]
    return PIIResult(text=masked, original=original, entities=sorted(found.keys()), spans=spans)


_presidio_engine = None


def _get_presidio():
    global _presidio_engine
    if _presidio_engine is None:
        from presidio_analyzer import AnalyzerEngine  # type: ignore

        _presidio_engine = AnalyzerEngine()
    return _presidio_engine
