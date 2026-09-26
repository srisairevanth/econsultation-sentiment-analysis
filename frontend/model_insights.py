"""
Model Performance tab backend.

Loads the metrics/metadata already written to disk by ml_pipeline/train_model.py
(models/model_metadata.json) so the Streamlit app can show them live instead of
them sitting unseen in a JSON file. Never re-runs training or fabricates numbers -
if the model hasn't been trained yet, callers get a clear "not available" signal.
"""
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def is_report_available() -> bool:
    return config.MODEL_METADATA_PATH.exists()


def load_model_report() -> Optional[dict]:
    """Returns the full model_metadata.json content, or None if not trained yet."""
    if not config.MODEL_METADATA_PATH.exists():
        return None
    with open(config.MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def confusion_matrix_image_path() -> Optional[Path]:
    path = config.VISUALIZATIONS_DIR / "confusion_matrix_test.png"
    return path if path.exists() else None


MODEL_DISPLAY_NAMES = {
    "tfidf_logreg_tuned": "TF-IDF + Logistic Regression (tuned)",
    "tfidf_logreg": "TF-IDF + Logistic Regression",
    "tfidf_linear_svm": "TF-IDF + Linear SVM",
    "tfidf_multinomial_nb": "TF-IDF + Multinomial Naive Bayes",
}


def display_name(model_key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)
