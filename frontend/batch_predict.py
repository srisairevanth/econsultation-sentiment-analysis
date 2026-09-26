"""
Part 5.2: Batch sentiment prediction for multiple comments (CSV or JSON input).

Output preserves the original input columns and adds:
    predicted_sentiment, confidence

Run:
    python frontend/batch_predict.py --input comments.csv --output results.csv
    python frontend/batch_predict.py --input comments.json --column comment
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from predictor import ModelNotAvailableError, _load_model
from labeling_cleaning.text_utils import is_null_or_empty, normalize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _read_input(path: Path, column: str) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)
    if column not in df.columns:
        # try common aliases
        lower_map = {c.lower().strip(): c for c in df.columns}
        for alias in ["comment", "text", "comment_text", "body", "content"]:
            if alias in lower_map:
                column = lower_map[alias]
                break
        else:
            raise ValueError(f"Could not find a '{column}' column. Available columns: {list(df.columns)}")
    return df, column


def predict_batch(df: pd.DataFrame, column: str) -> pd.DataFrame:
    try:
        pipeline = _load_model()
    except ModelNotAvailableError as exc:
        logger.error(str(exc))
        raise

    df = df.copy()
    valid_mask = ~df[column].apply(is_null_or_empty)
    cleaned_texts = df.loc[valid_mask, column].apply(normalize_text)

    df["predicted_sentiment"] = None
    df["confidence"] = None

    if valid_mask.any():
        preds = pipeline.predict(cleaned_texts)
        df.loc[valid_mask, "predicted_sentiment"] = preds

        if hasattr(pipeline, "predict_proba"):
            probas = pipeline.predict_proba(cleaned_texts)
            classes = list(pipeline.classes_)
            confidences = [round(float(p[classes.index(pred)]), 4) for p, pred in zip(probas, preds)]
            df.loc[valid_mask, "confidence"] = confidences

    skipped = (~valid_mask).sum()
    if skipped:
        logger.warning("%d rows had empty/invalid comments and were skipped (no prediction).", skipped)

    return df


def main():
    parser = argparse.ArgumentParser(description="Batch-predict sentiment for a CSV/JSON of comments.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV or JSON file.")
    parser.add_argument("--output", type=str, default=None,
                         help="Path to write results CSV (default: <input>_predictions.csv).")
    parser.add_argument("--column", type=str, default="comment",
                         help="Name of the column containing comment text.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(2)

    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_predictions.csv")

    df, column = _read_input(input_path, args.column)
    logger.info("Loaded %d rows from %s", len(df), input_path)

    result_df = predict_batch(df, column)
    result_df.to_csv(output_path, index=False)
    logger.info("Predictions written to %s", output_path)


if __name__ == "__main__":
    main()
