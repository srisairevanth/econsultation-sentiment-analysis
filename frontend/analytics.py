"""
Part 5.3 / 5.5: Sentiment analytics and descriptive (non-causal) insights.

Operates on any DataFrame containing a sentiment column (e.g. the output of
frontend/batch_predict.py, with column 'predicted_sentiment', or a labeled
dataset with column 'label'). Never fabricates data - if a requested signal
(e.g. dates) isn't present, it is reported as unavailable rather than invented.
"""
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def compute_summary(df: pd.DataFrame, sentiment_column: str = "predicted_sentiment") -> dict:
    if sentiment_column not in df.columns:
        raise ValueError(f"Column '{sentiment_column}' not found. Available: {list(df.columns)}")

    valid = df[df[sentiment_column].isin(config.ALLOWED_LABELS)]
    total = len(valid)
    counts = {label: int((valid[sentiment_column] == label).sum()) for label in config.ALLOWED_LABELS}
    percentages = {label: round(100.0 * c / total, 2) if total else 0.0 for label, c in counts.items()}

    summary = {
        "total_comments": total,
        "counts": counts,
        "percentages": percentages,
    }

    if total > 0:
        most_common = max(counts, key=counts.get)
        summary["most_common_sentiment"] = most_common
        summary["insights"] = [
            f"Of the {total} analyzed comments, {most_common} is the most frequently observed "
            f"sentiment ({percentages[most_common]}%).",
            f"Negative sentiment was observed in {percentages['Negative']}% of analyzed comments.",
            f"Positive sentiment was observed in {percentages['Positive']}% of analyzed comments.",
            "These are descriptive patterns observed in the analyzed comments only, "
            "not causal findings about public opinion at large.",
        ]
    else:
        summary["most_common_sentiment"] = None
        summary["insights"] = ["No valid predictions available to summarize."]

    return summary


def find_date_column(df: pd.DataFrame) -> Optional[str]:
    """Looks for a plausible date column without fabricating one. Returns None if absent."""
    candidates = ["date", "posted_date", "postedDate", "received_date", "receivedDate",
                  "comment_date", "submission_date", "created_at", "attributes.postedDate"]
    lower_map = {c.lower().replace("_", "").replace(".", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace("_", "").replace(".", "")
        if key in lower_map:
            col = lower_map[key]
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0:
                return col
    return None
