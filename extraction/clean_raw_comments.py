"""
Part 1.5 - Raw data cleaning.

Merges the raw Regulations.gov extraction batches, removes records that
cannot possibly carry usable sentiment (null/empty comment bodies, and
attachment-only placeholder text such as "See attached file(s)" with no
actual opinion text included inline), decodes HTML entities/tags left over
from the Regulations.gov API response, strips duplicate submissions, and
writes one clean, de-duplicated file ready for LLM labeling.

Kept: comments that reference an attachment but ALSO contain substantial
inline text (a cover letter, an excerpt of the argument, etc.) - these are
real opinions and are not discarded just because the word "attached"
appears somewhere in them.

Usage:
    python clean_raw_comments.py
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

EXTRACTION_DIR = Path(__file__).resolve().parent
RAW_OUTPUT_DIR = EXTRACTION_DIR / "raw_output"
INPUT_BATCHES = [
    RAW_OUTPUT_DIR / "batch1_ftc_111.json",
    RAW_OUTPUT_DIR / "batch2_ftc_1000.json",
]
OUTPUT_PATH = EXTRACTION_DIR.parent / "data" / "raw" / "raw_financial_comments.json"
REPORT_PATH = EXTRACTION_DIR / "raw_cleaning_report.txt"

BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"[ \t]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")

PLACEHOLDER_RE = re.compile(
    r"^(please\s+)?(see|find)\s+attach(ed|ment)\b.{0,60}$",
    re.IGNORECASE,
)


def clean_text(raw: str) -> str:
    text = html.unescape(raw)
    text = BR_TAG_RE.sub("\n", text)
    text = TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    text = BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def is_placeholder_only(text: str) -> bool:
    word_count = len(text.split())
    if word_count > 15:
        return False
    return bool(PLACEHOLDER_RE.match(text.strip()))


def main() -> None:
    merged: dict[str, dict] = {}
    for path in INPUT_BATCHES:
        batch = json.loads(path.read_text(encoding="utf-8"))
        for record in batch:
            rid = record.get("id")
            if rid:
                merged[rid] = record

    total_raw = len(merged)
    null_or_empty = 0
    placeholder_only = 0
    exact_duplicate = 0
    kept = 0

    seen_normalized_text: set[str] = set()
    cleaned_records = []

    for rid, record in merged.items():
        attrs = record.get("attributes", {}) or {}
        raw_text = attrs.get("comment")
        if raw_text is None or not raw_text.strip():
            null_or_empty += 1
            continue

        cleaned = clean_text(raw_text)
        if not cleaned:
            null_or_empty += 1
            continue

        if is_placeholder_only(cleaned):
            placeholder_only += 1
            continue

        dedupe_key = cleaned.lower().strip()
        if dedupe_key in seen_normalized_text:
            exact_duplicate += 1
            continue
        seen_normalized_text.add(dedupe_key)

        cleaned_records.append(
            {
                "id": rid,
                "comment": cleaned,
                "agencyId": attrs.get("agencyId"),
                "docketId": attrs.get("docketId") or attrs.get("commentOnDocumentId"),
                "postedDate": attrs.get("postedDate"),
            }
        )
        kept += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(cleaned_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report_lines = [
        "Raw Comment Cleaning Report",
        "=" * 40,
        f"Input batches: {[p.name for p in INPUT_BATCHES]}",
        f"Total unique raw records (by Regulations.gov comment ID): {total_raw}",
        f"Dropped - null/empty comment body: {null_or_empty}",
        f"Dropped - attachment-only placeholder (e.g. 'See attached'): {placeholder_only}",
        f"Dropped - exact duplicate text (after normalization): {exact_duplicate}",
        f"Kept - clean, unique, substantive comments: {kept}",
        f"Output: {OUTPUT_PATH}",
    ]
    report = "\n".join(report_lines)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
