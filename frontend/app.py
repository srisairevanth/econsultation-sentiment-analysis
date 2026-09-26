"""
Sentiment Analysis of Comments Received Through an E-Consultation Module
Streamlit demonstration app.

MAIN FEATURE: Sentiment Analysis (single + batch) and Analytics.
ADDITIONAL FEATURE: U.S. Financial Regulatory AI Assistant (separate tab).

Run (from the project root):
    streamlit run frontend/app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import model_insights
import theme
import wordclouds
from analytics import compute_summary
from visualizations import (plot_confusion_matrix_interactive, plot_model_comparison_interactive,
                             plot_sentiment_bar, plot_sentiment_donut, plot_sentiment_trend)
from chatbot_service import ConversationManager, ask_regulatory_assistant
from context_manager import list_available_regulations
from batch_predict import predict_batch
from predictor import is_model_available, predict_sentiment

st.set_page_config(page_title="E-Consultation Sentiment Analysis", page_icon="\U0001F4CA", layout="wide")
theme.inject_custom_css()

EXAMPLE_COMMENTS = {
    # Real comments from the project's own dataset (data/processed/cleaned_labeled_comments.csv),
    # picked because the trained model classifies each correctly with reasonable confidence.
    "Positive": "I believe the extension should be granted to reduce the burden of information collection and cut costs.",
    "Negative": "This practice is illegal and constitutes predatory pricing. I urge the Commission to rule against personalized pricing.",
    "Neutral": "Please see the attached file for comments, concerns, and suggestions.",
}


@st.cache_data(show_spinner=False)
def _pipeline_status() -> dict:
    """Read-only summary of what each of the 4 pipeline stages has produced on disk."""
    status = {"raw": None, "labeled": None, "cleaned": None, "model": None}

    for candidate in config.RAW_DATA_CANDIDATES:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    status["raw"] = len(json.load(f))
            except Exception:
                pass
            break

    for candidate in config.LABELED_DATA_CANDIDATES:
        if candidate.exists():
            try:
                status["labeled"] = len(pd.read_csv(candidate))
            except Exception:
                pass
            break

    if config.CLEANED_DATA_PATH.exists():
        try:
            status["cleaned"] = len(pd.read_csv(config.CLEANED_DATA_PATH))
        except Exception:
            pass

    report = model_insights.load_model_report()
    if report:
        status["model"] = {
            "name": model_insights.display_name(report["model_type"]),
            "accuracy": report["test_set_metrics"]["accuracy"],
        }

    return status


@st.cache_data(show_spinner=False)
def _load_real_comments_pool() -> pd.DataFrame:
    if config.CLEANED_DATA_PATH.exists():
        return pd.read_csv(config.CLEANED_DATA_PATH)
    return pd.DataFrame(columns=["comment", "label"])


def _confidence_gauge(confidence: float, sentiment: str) -> go.Figure:
    color = theme.SENTIMENT_COLORS.get(sentiment, theme.PRIMARY)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#E2E8F0",
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=10))
    return fig


status = _pipeline_status()

with st.sidebar:
    st.markdown("### \U0001F4CB Pipeline Status")
    stage_rows = [
        ("Extraction", status["raw"], f"{status['raw']} real comments" if status["raw"] else None),
        ("Labeling", status["labeled"], f"{status['labeled']} labeled (Ollama+Qwen3)" if status["labeled"] else None),
        ("Cleaning", status["cleaned"], f"{status['cleaned']} rows" if status["cleaned"] else None),
        ("Model", status["model"], f"{status['model']['name']} • {status['model']['accuracy']*100:.1f}% acc." if status["model"] else None),
    ]
    for name, ok, detail in stage_rows:
        icon = "✅" if ok else "⬜"
        st.markdown(f"**{icon} {name}**")
        st.caption(detail or "not run yet")
    st.divider()
    st.caption("Academic mini project — sentiment analysis of U.S. financial-regulatory public comments.")

theme.hero_header(
    "Sentiment Analysis of E-Consultation Comments",
    "U.S. financial-regulatory public comments — classical ML sentiment classifier + Gemini-powered regulatory assistant",
)

if not is_model_available():
    st.warning(
        "No trained sentiment model found yet at `models/best_sentiment_model.joblib`.\n\n"
        "Run these commands once the labeled dataset is available:\n\n"
        "```\npython -m labeling_cleaning.clean_dataset\npython -m ml_pipeline.train_model\n```"
    )

tab_overview, tab_single, tab_batch, tab_analytics, tab_performance, tab_assistant = st.tabs(
    ["\U0001F3E0 Overview", "\U0001F50D Single Comment Analysis", "\U0001F4C2 Batch Analysis",
     "\U0001F4C8 Sentiment Analytics", "\U0001F4CA Model Performance",
     "\U0001F916 Regulatory AI Assistant"]
)

# ---------------------------------------------------------------------------
# TAB 0: Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.markdown("#### What this project does")
    st.write(
        "An end-to-end pipeline that pulls real public comments on U.S. financial regulations, "
        "labels their sentiment with a local LLM, trains a classical ML classifier on the result, "
        "and serves it through this dashboard — plus a Gemini-powered assistant that answers "
        "questions about the regulations themselves."
    )

    stat_cols = st.columns(4)
    stat_cols[0].markdown(theme.stat_card("\U0001F4C4", f"{status['raw'] or '—'}", "Real comments"), unsafe_allow_html=True)
    model_acc = f"{status['model']['accuracy']*100:.1f}%" if status["model"] else "—"
    stat_cols[1].markdown(theme.stat_card("\U0001F3AF", model_acc, "Test accuracy"), unsafe_allow_html=True)
    stat_cols[2].markdown(theme.stat_card("\U0001F3F7️", "3", "Sentiment classes"), unsafe_allow_html=True)
    stat_cols[3].markdown(theme.stat_card("\U0001F916", "Gemini", "AI regulatory assistant"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### How the pipeline works")
    pcols = st.columns(4)
    pcols[0].markdown(theme.pipeline_card("Stage 1", "Extraction", "Regulations.gov API → real FTC public comments"), unsafe_allow_html=True)
    pcols[1].markdown(theme.pipeline_card("Stage 2", "Labeling + Cleaning", "Local Ollama + Qwen3 8B LLM, then validated & de-duplicated"), unsafe_allow_html=True)
    pcols[2].markdown(theme.pipeline_card("Stage 3", "ML Pipeline", "TF-IDF + Linear SVM / Logistic Regression / Naive Bayes, fairly compared"), unsafe_allow_html=True)
    pcols[3].markdown(theme.pipeline_card("Stage 4", "Frontend", "This app — predictions, analytics, and an AI assistant"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "\U0001F449 Head to **Single Comment Analysis** to try it live, or **Model Performance** "
        "for the full accuracy/F1 breakdown."
    )

# ---------------------------------------------------------------------------
# TAB 1: Single comment sentiment analysis
# ---------------------------------------------------------------------------
with tab_single:
    st.subheader("Analyze a Single Comment")
    st.caption("Try an example, or paste your own comment below.")

    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
    if ex_col1.button("\U0001F4D7 Positive example"):
        st.session_state["single_comment_input"] = EXAMPLE_COMMENTS["Positive"]
    if ex_col2.button("\U0001F4D5 Negative example"):
        st.session_state["single_comment_input"] = EXAMPLE_COMMENTS["Negative"]
    if ex_col3.button("\U0001F4D8 Neutral example"):
        st.session_state["single_comment_input"] = EXAMPLE_COMMENTS["Neutral"]
    if ex_col4.button("\U0001F3B2 Surprise me"):
        pool = _load_real_comments_pool()
        if len(pool):
            st.session_state["single_comment_input"] = str(pool.sample(1).iloc[0]["comment"])

    comment_text = st.text_area("Enter an e-consultation comment", height=120, key="single_comment_input",
                                 placeholder="e.g. This proposal creates unnecessary compliance costs.")
    if st.button("Analyze", type="primary", key="analyze_single"):
        result = predict_sentiment(comment_text)
        if result["error"]:
            st.error(result["error"])
        else:
            sentiment = result["sentiment"]
            color = theme.SENTIMENT_COLORS[sentiment]
            bg = theme.SENTIMENT_BG[sentiment]
            st.markdown(
                f'<div class="result-card" style="background:{bg}; border-color:{color}44;">'
                f'<span style="font-size:0.85rem; color:#64748B; font-weight:600;">PREDICTED SENTIMENT</span><br>'
                f'<span style="font-size:1.6rem; font-weight:800; color:{color};">{sentiment}</span></div>',
                unsafe_allow_html=True,
            )
            if result["confidence"] is not None:
                gcol, _ = st.columns([1, 1])
                with gcol:
                    st.plotly_chart(_confidence_gauge(result["confidence"], sentiment), use_container_width=True)
            else:
                st.caption("Confidence score not available for this model type.")

# ---------------------------------------------------------------------------
# TAB 2: Batch analysis
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Analyze a Batch of Comments")
    st.caption("Upload a CSV with a 'comment' column (or a similarly named column). "
               "Tip: `data/processed/test_split.csv` has 130 real, held-out comments ready to use.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            column = "comment" if "comment" in df.columns else df.columns[0]
            st.write(f"Loaded {len(df)} rows. Using column: `{column}`")
            if st.button("Analyze All Comments", type="primary"):
                with st.spinner("Running predictions..."):
                    result_df = predict_batch(df, column)
                st.session_state["batch_results"] = result_df
                st.success(f"Analyzed {len(result_df)} comments.")
        except Exception as exc:
            st.error(f"Could not process file: {exc}")

    if "batch_results" in st.session_state:
        result_df = st.session_state["batch_results"]
        st.dataframe(result_df, use_container_width=True)
        st.download_button("Download Results as CSV", result_df.to_csv(index=False),
                            file_name="sentiment_predictions.csv", mime="text/csv")

        if "predicted_sentiment" in result_df.columns:
            batch_summary = compute_summary(result_df, "predicted_sentiment")
            if batch_summary["total_comments"] > 0:
                st.markdown("**This batch's sentiment breakdown:**")
                bcol_a, bcol_b = st.columns(2)
                with bcol_a:
                    st.plotly_chart(plot_sentiment_bar(batch_summary["counts"]), use_container_width=True)
                with bcol_b:
                    st.plotly_chart(plot_sentiment_donut(batch_summary["counts"]), use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 3: Analytics dashboard
# ---------------------------------------------------------------------------
with tab_analytics:
    st.subheader("Sentiment Analytics Dashboard")
    source_df = st.session_state.get("batch_results")
    sentiment_col = "predicted_sentiment"
    if source_df is None and config.CLEANED_DATA_PATH.exists():
        st.caption("No batch results yet in this session - showing analytics for the cleaned "
                   "labeled dataset instead.")
        source_df = pd.read_csv(config.CLEANED_DATA_PATH)
        sentiment_col = "label"

    if source_df is None or sentiment_col not in source_df.columns:
        st.info("Run a batch analysis (or complete dataset cleaning) to see analytics here.")
    else:
        summary = compute_summary(source_df, sentiment_col)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Comments", summary["total_comments"])
        c2.metric("Positive", f"{summary['counts']['Positive']} ({summary['percentages']['Positive']}%)")
        c3.metric("Negative", f"{summary['counts']['Negative']} ({summary['percentages']['Negative']}%)")
        c4.metric("Neutral", f"{summary['counts']['Neutral']} ({summary['percentages']['Neutral']}%)")

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_sentiment_bar(summary["counts"]), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_sentiment_donut(summary["counts"]), use_container_width=True)

        trend_fig = plot_sentiment_trend(source_df, sentiment_col)
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True)
        else:
            st.caption("Sentiment-over-time chart skipped: no usable date column found in this data.")

        st.markdown("**Observed patterns (descriptive, not causal):**")
        for insight in summary["insights"]:
            st.write(f"- {insight}")

        if sentiment_col == "label" and {"comment", "label"}.issubset(source_df.columns):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### \U0001F4AC Most distinctive words by sentiment")
            st.caption("Words that are disproportionately common in each sentiment class vs. the rest "
                       "of the dataset (not just the most frequent words overall).")
            wc_cols = st.columns(3)
            for i, label in enumerate(config.ALLOWED_LABELS):
                with wc_cols[i]:
                    st.markdown(f"**{label}**")
                    img = wordclouds.generate_wordcloud_image(source_df, label, theme.SENTIMENT_COLORMAPS[label])
                    if img:
                        st.image(img, use_container_width=True)
                    else:
                        st.caption("Not enough data for this class yet.")

# ---------------------------------------------------------------------------
# TAB 4: Model Performance (transparency dashboard for the trained classifier)
# ---------------------------------------------------------------------------
with tab_performance:
    st.subheader("Model Performance")
    if not model_insights.is_report_available():
        st.info("No trained model found yet. Run `python -m ml_pipeline.train_model` first.")
    else:
        report = model_insights.load_model_report()
        metrics = report["test_set_metrics"]

        st.markdown(f"**Best model selected:** {model_insights.display_name(report['model_type'])}")
        st.caption(report["selection_reason"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Accuracy", f"{metrics['accuracy']*100:.1f}%")
        m2.metric("Macro F1", f"{metrics['f1_macro']:.3f}")
        m3.metric("Weighted F1", f"{metrics['f1_weighted']:.3f}")
        m4.metric("Dataset Size", report["dataset_size_total"])

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Per-class performance (test set):**")
            per_class_df = pd.DataFrame(metrics["per_class"]).T
            st.dataframe(per_class_df, use_container_width=True)
        with col_b:
            st.plotly_chart(
                plot_confusion_matrix_interactive(metrics["confusion_matrix"], metrics["confusion_matrix_labels"]),
                use_container_width=True,
            )

        st.markdown("**Model comparison (validation split):**")
        comp_rows = [
            {
                "model": model_insights.display_name(name),
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("f1_macro"),
                "weighted_f1": m.get("f1_weighted"),
            }
            for name, m in report["candidate_comparison_on_eval_split"].items()
        ]
        cc1, cc2 = st.columns([3, 2])
        with cc1:
            st.plotly_chart(plot_model_comparison_interactive(comp_rows), use_container_width=True)
        with cc2:
            comp_df = pd.DataFrame(comp_rows).sort_values("macro_f1", ascending=False)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("**Class distribution (full dataset):**")
        dist_df = pd.DataFrame.from_dict(
            report["dataset_class_distribution"], orient="index", columns=["count"])
        st.bar_chart(dist_df)

        st.divider()
        split_info = report["split_info"]
        st.caption(
            f"Split: {split_info['strategy']} (train={split_info['train_size']}, "
            f"val={split_info['val_size']}, test={split_info['test_size']}, "
            f"random_state={split_info['random_state']})"
        )
        st.caption(report.get("label_noise_disclosure", ""))

# ---------------------------------------------------------------------------
# TAB 5: U.S. Financial Regulatory AI Assistant (additional feature)
# ---------------------------------------------------------------------------
with tab_assistant:
    st.subheader("U.S. Financial Regulatory AI Assistant")
    st.caption(
        "ADDITIONAL FEATURE - separate from the main sentiment analysis system. "
        "Uses the Gemini API with a selected regulatory context; not fine-tuned, "
        "and not a source of legal advice."
    )

    regulations = list_available_regulations()
    if not regulations:
        st.warning("No regulatory context files found under data/regulatory_context/.")
    else:
        options = {r["title"]: r["id"] for r in regulations}
        selected_title = st.selectbox("Select a regulation/topic", list(options.keys()))
        selected_id = options[selected_title]

        if "chat_manager" not in st.session_state:
            st.session_state["chat_manager"] = ConversationManager()
        if "chat_log" not in st.session_state:
            st.session_state["chat_log"] = []

        for role, content in st.session_state["chat_log"]:
            with st.chat_message(role):
                st.write(content)

        question = st.chat_input("Ask a question about the selected regulation...")
        if question:
            st.session_state["chat_log"].append(("user", question))
            with st.spinner("Thinking..."):
                result = ask_regulatory_assistant(question, selected_id, st.session_state["chat_manager"])
            if result["error"]:
                st.session_state["chat_log"].append(("assistant", f"⚠️ {result['error']}"))
            else:
                st.session_state["chat_log"].append(("assistant", result["answer"]))
            st.rerun()

        st.info(
            "AI-generated information for educational and informational purposes. "
            "Refer to official regulatory sources for authoritative information."
        )
        if st.button("Clear Conversation"):
            st.session_state["chat_manager"].clear()
            st.session_state["chat_log"] = []
            st.rerun()
