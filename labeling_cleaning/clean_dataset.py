"""
Stage 2b: Dataset Verification and Cleaning.

Loads the LLM-labeled dataset (produced by labeling_cleaning/label_comments.py,
Ollama + Qwen3 8B), validates it, cleans it, and writes:
  - data/processed/cleaned_labeled_comments.csv
  - reports/dataset_quality_report.json
  - reports/dataset_quality_report.txt
  - reports/manual_review_sample.csv (optional random sample for human review)

Run:
    python -m labeling_cleaning.clean_dataset
    python -m labeling_cleaning.clean_dataset --input path/to/labeled.csv
    python -m labeling_cleaning.clean_dataset --manual-review-sample 50

If no labeled dataset can be found yet (because label_comments.py is still
running), this script exits cleanly with a clear message instead of crashing,
so it can simply be re-run once the file appears.
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from labeling_cleaning.text_utils import is_null_or_empty, normalize_text, word_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COMMENT_COLUMN_ALIASES = ["comment", "text", "comment_text", "body", "content"]
LABEL_COLUMN_ALIASES = ["label", "sentiment", "sentiment_label", "class"]


def find_labeled_dataset(explicit_path: Optional[str] = None) -> Optional[Path]:
    """Search known candidate locations for the LLM-labeled dataset."""
    if explicit_path:
        p = Path(explicit_path)
        return p if p.exists() else None
    for candidate in config.LABELED_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    if config.LABELED_DIR.exists():
        matches = sorted(config.LABELED_DIR.glob("*.csv"))
        if matches:
            return matches[0]
    return None


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV (or JSON records) labeled dataset without assuming exact column names."""
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_csv(path)


def resolve_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Find the actual comment/label column names, case-insensitively, among aliases."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    comment_col = next((lower_map[a] for a in COMMENT_COLUMN_ALIASES if a in lower_map), None)
    label_col = next((lower_map[a] for a in LABEL_COLUMN_ALIASES if a in lower_map), None)
    if comment_col is None or label_col is None:
        raise ValueError(
            f"Could not find required comment/label columns. Found columns: {list(df.columns)}. "
            f"Expected one of {COMMENT_COLUMN_ALIASES} for comment and one of "
            f"{LABEL_COLUMN_ALIASES} for label."
        )
    return comment_col, label_col


def validate_and_clean(df: pd.DataFrame, comment_col: str, label_col: str,
                        invalid_label_action: str = "remove"):
    stats = {"warnings": []}
    stats["original_size"] = len(df)

    df = df.rename(columns={comment_col: "comment", label_col: "label"})[["comment", "label"]].copy()
    df["original_comment"] = df["comment"]

    # --- 3.1 Label validation -------------------------------------------------
    df["label"] = df["label"].astype(str).str.strip()
    valid_mask = df["label"].isin(config.ALLOWED_LABELS)
    invalid_labels = df.loc[~valid_mask, "label"]
    invalid_label_counts = invalid_labels.value_counts().to_dict()
    stats["invalid_labels_found"] = invalid_label_counts
    stats["invalid_label_count"] = int((~valid_mask).sum())
    if stats["invalid_label_count"] > 0:
        logger.warning("Found %d rows with invalid labels: %s",
                        stats["invalid_label_count"], invalid_label_counts)
        stats["warnings"].append(
            f"{stats['invalid_label_count']} rows had labels outside "
            f"{config.ALLOWED_LABELS} and were {invalid_label_action}d."
        )
    if invalid_label_action == "remove":
        df = df[valid_mask].copy()
    else:
        df["label_flagged_invalid"] = ~valid_mask

    # --- 3.2 Empty/null comment removal ---------------------------------------
    empty_mask = df["comment"].apply(is_null_or_empty)
    stats["empty_or_null_comments_removed"] = int(empty_mask.sum())
    df = df[~empty_mask].copy()

    # --- 3.3 Duplicate detection ------------------------------------------------
    stats["size_before_duplicate_removal"] = len(df)
    exact_comment_dupes = df.duplicated(subset=["comment"], keep="first")
    stats["exact_duplicate_comment_count"] = int(exact_comment_dupes.sum())

    full_dupes = df.duplicated(subset=["comment", "label"], keep="first")
    stats["duplicate_comment_and_label_count"] = int(full_dupes.sum())

    df = df[~exact_comment_dupes].copy()
    stats["duplicates_removed"] = stats["size_before_duplicate_removal"] - len(df)

    # --- 3.4 Text normalization --------------------------------------------------
    df["comment"] = df["comment"].apply(normalize_text)
    post_norm_empty = df["comment"].apply(is_null_or_empty)
    stats["empty_after_normalization_removed"] = int(post_norm_empty.sum())
    df = df[~post_norm_empty].copy()

    # --- 3.5 Length analysis -----------------------------------------------------
    lengths_chars = df["comment"].str.len()
    lengths_words = df["comment"].apply(word_count)
    stats["length_stats_chars"] = {
        "min": int(lengths_chars.min()) if len(df) else 0,
        "max": int(lengths_chars.max()) if len(df) else 0,
        "mean": float(lengths_chars.mean()) if len(df) else 0.0,
        "median": float(lengths_chars.median()) if len(df) else 0.0,
        "p25": float(lengths_chars.quantile(0.25)) if len(df) else 0.0,
        "p75": float(lengths_chars.quantile(0.75)) if len(df) else 0.0,
        "p90": float(lengths_chars.quantile(0.90)) if len(df) else 0.0,
    }
    stats["length_stats_words"] = {
        "min": int(lengths_words.min()) if len(df) else 0,
        "max": int(lengths_words.max()) if len(df) else 0,
        "mean": float(lengths_words.mean()) if len(df) else 0.0,
        "median": float(lengths_words.median()) if len(df) else 0.0,
    }
    short_mask = (lengths_chars < config.MIN_COMMENT_LENGTH_CHARS) | \
                 (lengths_words < config.MIN_COMMENT_LENGTH_WORDS)
    stats["short_comments_flagged"] = int(short_mask.sum())
    stats["short_comment_threshold_chars"] = config.MIN_COMMENT_LENGTH_CHARS
    stats["short_comment_threshold_words"] = config.MIN_COMMENT_LENGTH_WORDS
    if stats["short_comments_flagged"] > 0:
        stats["warnings"].append(
            f"{stats['short_comments_flagged']} comments are shorter than "
            f"{config.MIN_COMMENT_LENGTH_WORDS} words / {config.MIN_COMMENT_LENGTH_CHARS} chars "
            f"and were flagged (not auto-removed) - review reports/dataset_quality_report.json."
        )
    df["is_short_comment"] = short_mask.values

    # --- 3.6 Label distribution ---------------------------------------------------
    label_counts = df["label"].value_counts()
    total = len(df)
    label_distribution = {
        label: {
            "count": int(label_counts.get(label, 0)),
            "percentage": round(100.0 * label_counts.get(label, 0) / total, 2) if total else 0.0,
        }
        for label in config.ALLOWED_LABELS
    }
    stats["label_distribution"] = label_distribution

    if total > 0:
        counts = [v["count"] for v in label_distribution.values() if v["count"] > 0]
        if counts:
            ratio = max(counts) / min(counts)
            stats["class_imbalance_ratio"] = round(ratio, 2)
            if ratio >= config.CLASS_IMBALANCE_RATIO_THRESHOLD:
                stats["warnings"].append(
                    f"Class imbalance detected (majority:minority ratio = {ratio:.2f}). "
                    "Consider class_weight='balanced' during training, macro-F1 as the "
                    "primary metric, or targeted collection of the minority class(es). "
                    "Synthetic oversampling was NOT applied automatically."
                )

    stats["final_size"] = len(df)
    return df.reset_index(drop=True), stats


def write_reports(stats: dict, source_path: Path) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source_path),
        "note": (
            "Initial training labels were automatically generated by a Local Large "
            "Language Model (Ollama + Qwen3 8B) and may contain labeling noise. "
            "Labels have not been exhaustively manually verified; see "
            "reports/manual_review_sample.csv for an optional spot-check sample."
        ),
        **stats,
    }
    with open(config.DATASET_QUALITY_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    lines = [
        "DATASET QUALITY REPORT",
        "=" * 60,
        f"Generated: {report['generated_at_utc']}",
        f"Source file: {report['source_file']}",
        "",
        f"Original dataset size:            {stats['original_size']}",
        f"Invalid labels found:             {stats['invalid_label_count']} {stats['invalid_labels_found']}",
        f"Empty/null comments removed:      {stats['empty_or_null_comments_removed']}",
        f"Exact duplicate comments:         {stats['exact_duplicate_comment_count']}",
        f"Duplicate (comment+label) rows:   {stats['duplicate_comment_and_label_count']}",
        f"Duplicates removed:               {stats['duplicates_removed']}",
        f"Empty after normalization:        {stats['empty_after_normalization_removed']}",
        f"Final dataset size:               {stats['final_size']}",
        "",
        "LABEL DISTRIBUTION",
        "-" * 60,
    ]
    for label, d in stats["label_distribution"].items():
        lines.append(f"  {label:10s}: {d['count']:6d}  ({d['percentage']}%)")
    lines += [
        "",
        "TEXT LENGTH (characters)",
        "-" * 60,
        f"  min={stats['length_stats_chars']['min']}  max={stats['length_stats_chars']['max']}  "
        f"mean={stats['length_stats_chars']['mean']:.1f}  median={stats['length_stats_chars']['median']:.1f}",
        f"  Short comments flagged (< {stats['short_comment_threshold_words']} words): "
        f"{stats['short_comments_flagged']}",
        "",
        "WARNINGS",
        "-" * 60,
    ]
    lines += [f"  - {w}" for w in stats["warnings"]] or ["  (none)"]
    with open(config.DATASET_QUALITY_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Quality reports written to %s and %s",
                config.DATASET_QUALITY_REPORT_JSON, config.DATASET_QUALITY_REPORT_TXT)


def export_manual_review_sample(df: pd.DataFrame, n: int, seed: int) -> None:
    if len(df) == 0 or n <= 0:
        return
    n = min(n, len(df))
    sample = df.sample(n=n, random_state=seed)[["comment", "label"]].copy()
    sample["human_reviewed_label"] = ""
    sample.to_csv(config.MANUAL_REVIEW_SAMPLE_PATH, index=False)
    logger.info("Manual review sample (%d rows) written to %s", n, config.MANUAL_REVIEW_SAMPLE_PATH)


def main():
    parser = argparse.ArgumentParser(description="Verify and clean the LLM-labeled dataset.")
    parser.add_argument("--input", type=str, default=None,
                         help="Explicit path to the labeled dataset (CSV or JSON). "
                              "If omitted, known candidate paths are searched automatically.")
    parser.add_argument("--invalid-label-action", choices=["remove", "flag"],
                         default=config.INVALID_LABEL_ACTION)
    parser.add_argument("--manual-review-sample", type=int, default=50,
                         help="Number of rows to export for optional manual label review (0 to skip).")
    args = parser.parse_args()

    dataset_path = find_labeled_dataset(args.input)
    if dataset_path is None:
        logger.error(
            "No labeled dataset found yet. Checked: %s. "
            "This is expected if the Local LLM labeling module is still running - "
            "re-run this script once labeled_comments.csv is available.",
            [str(p) for p in config.LABELED_DATA_CANDIDATES],
        )
        sys.exit(2)

    logger.info("Loading labeled dataset from %s", dataset_path)
    df = load_dataset(dataset_path)
    logger.info("Loaded %d rows, columns=%s", len(df), list(df.columns))

    comment_col, label_col = resolve_columns(df)
    cleaned_df, stats = validate_and_clean(df, comment_col, label_col, args.invalid_label_action)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_cols = ["comment", "label", "original_comment"]
    if "label_flagged_invalid" in cleaned_df.columns:
        output_cols.append("label_flagged_invalid")
    if "is_short_comment" in cleaned_df.columns:
        output_cols.append("is_short_comment")
    cleaned_df[output_cols].to_csv(config.CLEANED_DATA_PATH, index=False)
    logger.info("Cleaned dataset (%d rows) written to %s", len(cleaned_df), config.CLEANED_DATA_PATH)

    write_reports(stats, dataset_path)
    if args.manual_review_sample > 0:
        export_manual_review_sample(cleaned_df, args.manual_review_sample, config.RANDOM_STATE)

    logger.info("Done. %d -> %d rows after cleaning.", stats["original_size"], stats["final_size"])


if __name__ == "__main__":
    main()
