"""
Part 5.4: Sentiment visualizations.

Generates a bar chart, a donut chart, and (only if real date data is present)
a sentiment-trend-over-time chart. Nothing is fabricated: the trend chart is
skipped with a clear message when no usable date column exists.

Charts are saved under reports/visualizations/.
"""
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from analytics import find_date_column

SENTIMENT_COLORS = {"Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#7f7f7f"}


def plot_sentiment_bar(counts: dict, out_path: Path = config.VISUALIZATIONS_DIR / "sentiment_bar_chart.png"):
    labels = config.ALLOWED_LABELS
    values = [counts.get(l, 0) for l in labels]
    colors = [SENTIMENT_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Sentiment Distribution of Analyzed Comments")
    ax.set_ylabel("Number of Comments")
    ax.bar_label(bars, padding=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sentiment_donut(counts: dict, out_path: Path = config.VISUALIZATIONS_DIR / "sentiment_donut_chart.png"):
    labels = [l for l in config.ALLOWED_LABELS if counts.get(l, 0) > 0]
    values = [counts[l] for l in labels]
    colors = [SENTIMENT_COLORS[l] for l in labels]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if values:
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, wedgeprops=dict(width=0.4),
        )
    ax.set_title("Sentiment Percentage Summary")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sentiment_trend(df: pd.DataFrame, sentiment_column: str,
                          out_path: Path = config.VISUALIZATIONS_DIR / "sentiment_trend.png") -> Optional[Path]:
    """Only produced when a real, parseable date column exists. Returns None otherwise."""
    date_col = find_date_column(df)
    if date_col is None:
        return None

    tmp = df.copy()
    tmp["_date"] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=["_date"])
    if tmp.empty:
        return None

    tmp["_period"] = tmp["_date"].dt.to_period("M").dt.to_timestamp()
    trend = tmp.groupby(["_period", sentiment_column]).size().unstack(fill_value=0)
    for label in config.ALLOWED_LABELS:
        if label not in trend.columns:
            trend[label] = 0
    trend = trend[config.ALLOWED_LABELS]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label in config.ALLOWED_LABELS:
        ax.plot(trend.index, trend[label], marker="o", label=label, color=SENTIMENT_COLORS[label])
    ax.set_title("Sentiment Trend Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Comments")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_all_visualizations(df: pd.DataFrame, sentiment_column: str = "predicted_sentiment") -> dict:
    from analytics import compute_summary
    summary = compute_summary(df, sentiment_column)
    bar_path = plot_sentiment_bar(summary["counts"])
    donut_path = plot_sentiment_donut(summary["counts"])
    trend_path = plot_sentiment_trend(df, sentiment_column)
    return {
        "bar_chart": str(bar_path),
        "donut_chart": str(donut_path),
        "trend_chart": str(trend_path) if trend_path else None,
        "trend_chart_skipped_reason": None if trend_path else
            "No usable date column found in the data - trend chart was not fabricated.",
    }
