"""
Standalone evaluation of the SAVED model (models/best_sentiment_model.joblib).

Useful to re-check the model's performance independently of the training run,
e.g. after moving the model file, or to evaluate on a different held-out CSV.

Run:
    python -m ml_pipeline.evaluate_model
    python -m ml_pipeline.evaluate_model --dataset path/to/other_labeled.csv
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from ml_pipeline.model_utils import evaluate_predictions, plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved sentiment model.")
    parser.add_argument("--dataset", type=str, default=None,
                         help="CSV with 'comment'/'label' columns. Defaults to the held-out "
                              "test_split.csv produced by ml_pipeline.train_model.")
    args = parser.parse_args()

    if not config.MODEL_PATH.exists():
        logger.error("No trained model found at %s. Run 'python -m ml_pipeline.train_model' first.",
                      config.MODEL_PATH)
        sys.exit(2)

    dataset_path = Path(args.dataset) if args.dataset else config.PROCESSED_DIR / "test_split.csv"
    if not dataset_path.exists():
        logger.error("Evaluation dataset not found at %s.", dataset_path)
        sys.exit(2)

    model = joblib.load(config.MODEL_PATH)
    df = pd.read_csv(dataset_path).dropna(subset=["comment", "label"])
    df = df[df["label"].isin(config.ALLOWED_LABELS)]
    logger.info("Evaluating on %d rows from %s", len(df), dataset_path)

    preds = model.predict(df["comment"])
    metrics = evaluate_predictions(df["label"], preds)
    text_report = classification_report(df["label"], preds, labels=config.ALLOWED_LABELS, zero_division=0)

    logger.info("accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f",
                 metrics["accuracy"], metrics["f1_macro"], metrics["f1_weighted"])
    print(text_report)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.REPORTS_DIR / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": str(dataset_path), "metrics": metrics}, f, indent=2, default=str)
    with open(config.REPORTS_DIR / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(text_report)

    config.VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(metrics["confusion_matrix"], metrics["confusion_matrix_labels"],
                           "Confusion Matrix (evaluate_model.py)",
                           config.VISUALIZATIONS_DIR / "confusion_matrix_eval.png")
    logger.info("Reports written to %s", config.REPORTS_DIR)


if __name__ == "__main__":
    main()
