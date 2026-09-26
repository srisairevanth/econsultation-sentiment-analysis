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
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
import model_insights
from analytics import compute_summary
from visualizations import plot_sentiment_bar, plot_sentiment_donut, plot_sentiment_trend
from chatbot_service import ConversationManager, ask_regulatory_assistant
from context_manager import list_available_regulations
from batch_predict import predict_batch
from predictor import is_model_available, predict_sentiment

st.set_page_config(page_title="E-Consultation Sentiment Analysis", page_icon="\U0001F4CA", layout="wide")

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


st.title("Sentiment Analysis of Comments Received Through an E-Consultation Module")
st.caption(
    "MAIN FEATURE: Sentiment Analysis, Batch Analysis & Analytics  |  "
    "ADDITIONAL FEATURE: U.S. Financial Regulatory AI Assistant"
)

with st.sidebar:
    st.markdown("### Pipeline status")
    status = _pipeline_status()
    st.markdown(
        f"- {'✅' if status['raw'] else '⬜'} **Extraction:** "
        + (f"{status['raw']} real comments" if status['raw'] else "not run yet")
    )
    st.markdown(
        f"- {'✅' if status['labeled'] else '⬜'} **Labeling (Ollama + Qwen3):** "
        + (f"{status['labeled']} labeled" if status['labeled'] else "not run yet")
    )
    st.markdown(
        f"- {'✅' if status['cleaned'] else '⬜'} **Cleaning:** "
        + (f"{status['cleaned']} rows" if status['cleaned'] else "not run yet")
    )
    st.markdown(
        f"- {'✅' if status['model'] else '⬜'} **Model:** "
        + (f"{status['model']['name']} ({status['model']['accuracy']*100:.1f}% acc.)"
           if status['model'] else "not trained yet")
    )
    st.divider()
    st.caption("Academic mini project - sentiment analysis of U.S. financial-regulatory public comments.")

if not is_model_available():
    st.warning(
        "No trained sentiment model found yet at `models/best_sentiment_model.joblib`.\n\n"
        "Run these commands once the labeled dataset is available:\n\n"
        "```\npython -m labeling_cleaning.clean_dataset\npython -m ml_pipeline.train_model\n```"
    )

tab_single, tab_batch, tab_analytics, tab_performance, tab_assistant = st.tabs(
    ["\U0001F50D Single Comment Analysis", "\U0001F4C2 Batch Analysis",
     "\U0001F4C8 Sentiment Analytics", "\U0001F4CA Model Performance",
     "\U0001F916 Regulatory AI Assistant"]
)

# ---------------------------------------------------------------------------
# TAB 1: Single comment sentiment analysis
# ---------------------------------------------------------------------------
with tab_single:
    st.subheader("Analyze a Single Comment")
    st.caption("Try an example, or paste your own comment below.")

    ex_col1, ex_col2, ex_col3 = st.columns(3)
    if ex_col1.button("📗 Try a positive example"):
        st.session_state["single_comment_input"] = EXAMPLE_COMMENTS["Positive"]
    if ex_col2.button("📕 Try a negative example"):
        st.session_state["single_comment_input"] = EXAMPLE_COMMENTS["Negative"]
    if ex_col3.button("📘 Try a neutral example"):
        st.session_state["single_comment_input"] = EXAMPLE_COMMENTS["Neutral"]

    comment_text = st.text_area("Enter an e-consultation comment", height=120, key="single_comment_input",
                                 placeholder="e.g. This proposal creates unnecessary compliance costs.")
    if st.button("Analyze", type="primary", key="analyze_single"):
        result = predict_sentiment(comment_text)
        if result["error"]:
            st.error(result["error"])
        else:
            color = {"Positive": "green", "Negative": "red", "Neutral": "gray"}[result["sentiment"]]
            st.markdown(f"**Predicted Sentiment:** :{color}[{result['sentiment']}]")
            if result["confidence"] is not None:
                st.progress(result["confidence"], text=f"Confidence: {result['confidence']*100:.1f}%")
            else:
                st.caption("Confidence score not available for this model type.")

# ---------------------------------------------------------------------------
# TAB 2: Batch analysis
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Analyze a Batch of Comments")
    st.caption("Upload a CSV with a 'comment' column (or a similarly named column).")
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
                    st.image(str(plot_sentiment_bar(batch_summary["counts"])))
                with bcol_b:
                    st.image(str(plot_sentiment_donut(batch_summary["counts"])))

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
            bar_path = plot_sentiment_bar(summary["counts"])
            st.image(str(bar_path))
        with col_b:
            donut_path = plot_sentiment_donut(summary["counts"])
            st.image(str(donut_path))

        trend_path = plot_sentiment_trend(source_df, sentiment_col)
        if trend_path:
            st.image(str(trend_path))
        else:
            st.caption("Sentiment-over-time chart skipped: no usable date column found in this data.")

        st.markdown("**Observed patterns (descriptive, not causal):**")
        for insight in summary["insights"]:
            st.write(f"- {insight}")

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
            comp_df = pd.DataFrame(comp_rows).sort_values("macro_f1", ascending=False)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        with col_b:
            cm_path = model_insights.confusion_matrix_image_path()
            if cm_path:
                st.image(str(cm_path), caption="Confusion Matrix (test set)")
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
