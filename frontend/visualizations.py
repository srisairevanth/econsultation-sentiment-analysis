"""
Interactive sentiment visualizations (Plotly), for the Streamlit app.

Generates a bar chart, a donut chart, a sentiment-trend-over-time chart (only
if real date data is present), a confusion-matrix heatmap, and a model
comparison chart. Nothing is fabricated: the trend chart returns None with a
clear reason when no usable date column exists.
"""
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from analytics import find_date_column
from theme import SENTIMENT_COLORS


def plot_sentiment_bar(counts: dict) -> go.Figure:
    labels = config.ALLOWED_LABELS
    values = [counts.get(l, 0) for l in labels]
    fig = px.bar(
        x=labels, y=values, color=labels, color_discrete_map=SENTIMENT_COLORS,
        text=values, labels={"x": "Sentiment", "y": "Number of Comments"},
        title="Sentiment Distribution",
    )
    fig.update_traces(textposition="outside", showlegend=False)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=360)
    return fig


def plot_sentiment_donut(counts: dict) -> go.Figure:
    labels = [l for l in config.ALLOWED_LABELS if counts.get(l, 0) > 0]
    values = [counts[l] for l in labels]
    fig = px.pie(
        values=values, names=labels, hole=0.55,
        color=labels, color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Percentage Summary",
    )
    fig.update_traces(textinfo="label+percent")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=360)
    return fig


def plot_sentiment_trend(df: pd.DataFrame, sentiment_column: str) -> Optional[go.Figure]:
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
    trend = trend[config.ALLOWED_LABELS].reset_index()

    fig = go.Figure()
    for label in config.ALLOWED_LABELS:
        fig.add_trace(go.Scatter(
            x=trend["_period"], y=trend[label], mode="lines+markers",
            name=label, line=dict(color=SENTIMENT_COLORS[label]),
        ))
    fig.update_layout(
        title="Sentiment Trend Over Time", xaxis_title="Month", yaxis_title="Number of Comments",
        margin=dict(l=10, r=10, t=40, b=10), height=380,
    )
    return fig


def plot_confusion_matrix_interactive(cm: List[List[int]], labels: List[str]) -> go.Figure:
    fig = px.imshow(
        cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Predicted label", y="True label", color="Count"),
        title="Confusion Matrix (test set)",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
    return fig


def plot_model_comparison_interactive(comp_rows: List[dict]) -> go.Figure:
    """comp_rows: [{'model': str, 'accuracy': float, 'macro_f1': float, 'weighted_f1': float}, ...]"""
    df = pd.DataFrame(comp_rows)
    melted = df.melt(id_vars="model", var_name="metric", value_name="score")
    fig = px.bar(
        melted, x="model", y="score", color="metric", barmode="group",
        title="Model Comparison (validation split)",
        labels={"score": "Score", "model": "", "metric": "Metric"},
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380, xaxis_tickangle=-15)
    return fig
