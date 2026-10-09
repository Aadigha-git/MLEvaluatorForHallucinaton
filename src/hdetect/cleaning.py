"""`clean_answer` v1 (docs/PREREGISTRATION.md, "Text normalization").

Applied before labeling and feature extraction; raw answers are stored unchanged.
"""

from __future__ import annotations

import re
import unicodedata

CLEAN_VERSION = "1"

_STRONG = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_EMPHASIS = re.compile(r"(?<![\w*])([*_])(?=\S)(.+?)(?<=\S)\1(?![\w*])", re.DOTALL)
_LINE_MARKER = re.compile(r"^[ \t]*(?:#{1,6}[ \t]+|>[ \t]?|[-*+][ \t]+)", re.MULTILINE)


def clean_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _STRONG.sub(r"\2", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = text.replace("`", "")
    text = _LINE_MARKER.sub("", text)
    return " ".join(text.split())
