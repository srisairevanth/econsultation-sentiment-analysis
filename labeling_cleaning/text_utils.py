"""
Shared text-normalization utilities used by BOTH the offline cleaning pipeline
(preprocessing/clean_dataset.py) and the online predictor (prediction/predictor.py).

Keeping this logic in one place guarantees that a comment is normalized the
SAME way at training time and at prediction time - a common source of subtle
train/serve skew bugs.

The normalization here is intentionally conservative: sentiment-bearing
punctuation (!, ?, ...) and negation words (not/no/never/cannot) are preserved.
"""
import re
import unicodedata

# Strings that consist only of these characters are treated as "meaningless"
_MEANINGLESS_PATTERN = re.compile(r"^[\s\.\-_=+*#~^`'\"|\\/,;:!?()\[\]{}<>]*$")

# Collapse runs of whitespace (including tabs/newlines) into a single space
_WHITESPACE_PATTERN = re.compile(r"\s+")

# Control characters (excluding the ones normal whitespace already handles)
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def is_null_or_empty(value) -> bool:
    """True for None/NaN/empty-string/whitespace-only/meaningless-characters-only."""
    if value is None:
        return True
    try:
        import math
        if isinstance(value, float) and math.isnan(value):
            return True
    except Exception:
        pass
    text = str(value)
    if text.strip() == "":
        return True
    if text.strip().lower() in {"nan", "none", "null", "n/a", "na"}:
        return True
    if _MEANINGLESS_PATTERN.match(text):
        return True
    return False


def normalize_text(text: str) -> str:
    """
    Conservative cleaning suitable for TF-IDF style sentiment models.
    - Fixes unicode (NFKC normalization) and encoding artifacts
    - Strips control characters
    - Collapses repeated whitespace
    - Trims leading/trailing whitespace
    Does NOT: lowercase (TfidfVectorizer handles that), strip punctuation, or
    remove negation/stopwords - these carry sentiment signal.
    """
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHAR_PATTERN.sub(" ", text)
    # common mojibake artifacts from bad encoding round-trips
    text = text.replace("﻿", "").replace("​", "")
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def word_count(text: str) -> int:
    text = normalize_text(text)
    return len(text.split()) if text else 0
