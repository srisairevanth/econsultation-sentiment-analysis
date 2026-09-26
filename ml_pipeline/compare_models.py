"""
Part 4.4: Optional model comparison.

Trains the TF-IDF + Logistic Regression baseline alongside two lightweight
classical alternatives (Linear SVM, Multinomial Naive Bayes) on the SAME
train split, evaluates all of them on the SAME validation split, and reports
a fair comparison. Selection priority: Macro F1 -> Weighted F1 -> Accuracy.

Can be run standalone for inspection:
    python -m ml_pipeline.compare_models
It is also imported by train_model.py as part of the full pipeline.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from ml_pipeline.model_utils import (build_candidate_pipelines, evaluate_predictions,
                                      load_cleaned_dataset, stratified_split)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def rank_key(metrics: dict):
    return (metrics["f1_macro"], metrics["f1_weighted"], metrics["accuracy"])


def compare_models(train_df, eval_df, random_state: int = config.RANDOM_STATE) -> dict:
    """
    Fits each candidate pipeline on train_df ONLY, evaluates on eval_df
    (validation set, or train-CV mean if eval_df is empty), and returns a
    comparison report plus the fitted pipelines.
    """
    pipelines = build_candidate_pipelines(random_state)
    results = {}
    fitted = {}

    use_cv = eval_df is None or len(eval_df) == 0
    if use_cv:
        from sklearn.model_selection import cross_val_predict
        logger.info("No validation split available - using 3-fold CV on the training set for comparison.")

    for name, pipeline in pipelines.items():
        logger.info("Training candidate model: %s", name)
        pipeline.fit(train_df["comment"], train_df["label"])
        fitted[name] = pipeline

        if use_cv:
            from sklearn.model_selection import cross_val_predict
            preds = cross_val_predict(pipeline, train_df["comment"], train_df["label"], cv=3)
            metrics = evaluate_predictions(train_df["label"], preds)
            metrics["evaluated_on"] = "3fold_cv_on_train"
        else:
            preds = pipeline.predict(eval_df["comment"])
            metrics = evaluate_predictions(eval_df["label"], preds)
            metrics["evaluated_on"] = "validation_split"
        results[name] = metrics
        logger.info("%s -> accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f",
                     name, metrics["accuracy"], metrics["f1_macro"], metrics["f1_weighted"])

    best_name = max(results, key=lambda n: rank_key(results[n]))
    logger.info("Best model by (macro_f1, weighted_f1, accuracy): %s", best_name)

    return {
        "results": results,
        "best_model_name": best_name,
        "fitted_pipelines": fitted,
    }


def main():
    df = load_cleaned_dataset()
    train_df, val_df, test_df, split_info = stratified_split(df)
    comparison = compare_models(train_df, val_df)

    report = {
        "split_info": split_info,
        "results": comparison["results"],
        "best_model_name": comparison["best_model_name"],
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.MODEL_COMPARISON_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Model comparison report written to %s", config.MODEL_COMPARISON_REPORT)


if __name__ == "__main__":
    main()
