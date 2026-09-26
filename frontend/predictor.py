"""
Single comment sentiment prediction.

Loads the model trained by ml_pipeline/train_model.py (TF-IDF + classical ML,
NOT the Local LLM) and exposes predict_sentiment(text) for reuse by the
batch predictor, the analytics module, and the Streamlit app.
"""
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from labeling_cleaning.text_utils import is_null_or_empty, normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Raised when models/best_sentiment_model.joblib has not been trained yet."""


@lru_cache(maxsize=1)
def _load_model():
    import joblib
    if not config.MODEL_PATH.exists():
        raise ModelNotAvailableError(
            f"No trained model found at {config.MODEL_PATH}. Run "
            f"'python -m labeling_cleaning.clean_dataset' then 'python -m ml_pipeline.train_model' first."
        )
    return joblib.load(config.MODEL_PATH)


def is_model_available() -> bool:
    return config.MODEL_PATH.exists()


def _confidence_from_pipeline(pipeline, text: str, predicted_label: str) -> Optional[float]:
    """
    Returns a probability-like confidence score for the predicted label if the
    underlying classifier supports it (predict_proba / decision_function).
    Documented limitation: for models without predict_proba and without our
    CalibratedClassifierCV wrapper, confidence is not available (returns None).
    """
    clf = pipeline.named_steps.get("clf") if hasattr(pipeline, "named_steps") else None
    try:
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba([text])[0]
            classes = list(pipeline.classes_)
            return float(proba[classes.index(predicted_label)])
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("predict_proba unavailable/failed: %s", exc)
    return None


def predict_sentiment(comment_text: str) -> dict:
    """
    Predict Positive/Negative/Neutral for a single comment.

    Returns:
        {"comment": str, "sentiment": str, "confidence": float|None, "error": str|None}
    """
    if is_null_or_empty(comment_text):
        return {"comment": comment_text, "sentiment": None, "confidence": None,
                "error": "Comment is empty or contains no meaningful text."}

    try:
        pipeline = _load_model()
    except ModelNotAvailableError as exc:
        return {"comment": comment_text, "sentiment": None, "confidence": None, "error": str(exc)}

    cleaned = normalize_text(comment_text)
    predicted_label = pipeline.predict([cleaned])[0]
    confidence = _confidence_from_pipeline(pipeline, cleaned, predicted_label)

    return {
        "comment": comment_text,
        "sentiment": predicted_label,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "error": None,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict sentiment for a single comment.")
    parser.add_argument("text", type=str, help="Comment text to classify.")
    args = parser.parse_args()
    print(predict_sentiment(args.text))
