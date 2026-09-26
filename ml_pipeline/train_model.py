"""
Part 4: Train the actual sentiment analysis model.

End-to-end reproducible pipeline:
  1. Load cleaned dataset (data/processed/cleaned_labeled_comments.csv)
  2. Validate it
  3. Stratified 70/15/15 train/val/test split (random_state=42), with a
     graceful 80/20 fallback for small datasets
  4. Train TF-IDF + Logistic Regression baseline
  5. Lightweight hyperparameter tuning (GridSearchCV on the TRAIN split only)
  6. Compare against Linear SVM and Multinomial Naive Bayes on the SAME split
  7. Select the best model (Macro F1 -> Weighted F1 -> Accuracy)
  8. Refit the best model on train+val, evaluate ONCE on the untouched test set
  9. Save the model, metadata, evaluation reports and confusion matrix

Run:
    python -m ml_pipeline.train_model
    python -m ml_pipeline.train_model --skip-comparison   (baseline only, faster)
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from ml_pipeline.compare_models import compare_models, rank_key
from ml_pipeline.model_utils import (evaluate_predictions, load_cleaned_dataset,
                                      plot_confusion_matrix, stratified_split)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def tune_logreg_baseline(train_df: pd.DataFrame, random_state: int = config.RANDOM_STATE) -> Pipeline:
    """Lightweight GridSearchCV over TF-IDF + LogisticRegression, fit on TRAIN only."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
    ])
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "tfidf__max_df": [0.9, 0.95],
        "clf__C": [0.5, 1.0, 5.0],
    }
    n_splits = min(3, train_df["label"].value_counts().min())
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(pipeline, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1)
    logger.info("Running lightweight GridSearchCV (%d-fold, %d combinations) on the training split only...",
                n_splits, len(param_grid["tfidf__ngram_range"]) * len(param_grid["tfidf__min_df"]) *
                len(param_grid["tfidf__max_df"]) * len(param_grid["clf__C"]))
    search.fit(train_df["comment"], train_df["label"])
    logger.info("Best params: %s (cv macro-f1=%.4f)", search.best_params_, search.best_score_)
    return search.best_estimator_, search.best_params_


def main():
    parser = argparse.ArgumentParser(description="Train the sentiment classification model.")
    parser.add_argument("--skip-comparison", action="store_true",
                         help="Skip SVM/NB comparison and use the tuned TF-IDF+LogReg baseline directly.")
    args = parser.parse_args()

    df = load_cleaned_dataset()
    logger.info("Loaded cleaned dataset: %d rows", len(df))
    if len(df) < 30:
        logger.error("Dataset has only %d rows - too small to train a meaningful model. "
                      "Wait for more labeled data or check preprocessing output.", len(df))
        sys.exit(2)

    train_df, val_df, test_df, split_info = stratified_split(df)
    logger.info("Split: %s", split_info)

    # persist the actual split files for reproducible downstream evaluation/demo
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.PROCESSED_DIR / "train_split.csv", index=False)
    if len(val_df):
        val_df.to_csv(config.PROCESSED_DIR / "val_split.csv", index=False)
    test_df.to_csv(config.PROCESSED_DIR / "test_split.csv", index=False)

    # --- Step 1: tune the required baseline (TF-IDF + Logistic Regression) ------
    tuned_baseline, best_params = tune_logreg_baseline(train_df)

    candidates = {"tfidf_logreg_tuned": tuned_baseline}
    comparison_results = {}

    if not args.skip_comparison:
        # --- Step 2: compare against Linear SVM / Multinomial NB on the SAME split
        comparison = compare_models(train_df, val_df)
        candidates.update(comparison["fitted_pipelines"])
        comparison_results = comparison["results"]

    # Evaluate the tuned baseline on the same eval set used for comparison, so all
    # candidates are judged identically.
    eval_df = val_df if len(val_df) else train_df
    eval_method = "validation_split" if len(val_df) else "train_cv_not_applicable_using_train_refit_check"
    baseline_preds = tuned_baseline.predict(eval_df["comment"])
    comparison_results["tfidf_logreg_tuned"] = evaluate_predictions(eval_df["label"], baseline_preds)
    comparison_results["tfidf_logreg_tuned"]["evaluated_on"] = eval_method

    best_name = max(comparison_results, key=lambda n: rank_key(comparison_results[n]))
    logger.info("Selected best model: %s", best_name)

    best_pipeline = candidates[best_name]

    # --- Refit best model on train+val (still excluding the untouched test set) --
    if len(val_df):
        train_plus_val = pd.concat([train_df, val_df], ignore_index=True)
    else:
        train_plus_val = train_df
    best_pipeline.fit(train_plus_val["comment"], train_plus_val["label"])

    # --- Final, single evaluation on the untouched test set -----------------------
    test_preds = best_pipeline.predict(test_df["comment"])
    test_metrics = evaluate_predictions(test_df["label"], test_preds)
    logger.info("TEST SET -> accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f",
                 test_metrics["accuracy"], test_metrics["f1_macro"], test_metrics["f1_weighted"])

    config.VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = config.VISUALIZATIONS_DIR / "confusion_matrix_test.png"
    plot_confusion_matrix(test_metrics["confusion_matrix"], test_metrics["confusion_matrix_labels"],
                           f"Confusion Matrix - {best_name} (test set)", cm_path)

    # --- Save model + metadata -----------------------------------------------------
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, config.MODEL_PATH)
    logger.info("Model saved to %s", config.MODEL_PATH)

    label_dist = df["label"].value_counts().to_dict()
    metadata = {
        "model_type": best_name,
        "selection_reason": "Highest Macro F1 (tie-break: Weighted F1, then Accuracy) "
                             "on the validation split, compared fairly against "
                             "TF-IDF+LogReg, Linear SVM, and Multinomial NB.",
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": config.RANDOM_STATE,
        "dataset_size_total": len(df),
        "dataset_class_distribution": label_dist,
        "split_info": split_info,
        "tuned_baseline_best_params": best_params,
        "test_set_metrics": test_metrics,
        "candidate_comparison_on_eval_split": comparison_results,
        "label_noise_disclosure": (
            "Initial labels were generated automatically by a Local LLM "
            "(Ollama + Qwen3 8B) and may contain labeling noise; see "
            "reports/manual_review_sample.csv for a manual spot-check sample."
        ),
    }
    with open(config.MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Model metadata saved to %s", config.MODEL_METADATA_PATH)

    with open(config.BASELINE_EVALUATION_REPORT, "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_name,
            "test_set_metrics": test_metrics,
            "candidate_comparison_on_eval_split": comparison_results,
        }, f, indent=2, default=str)
    logger.info("Evaluation report saved to %s", config.BASELINE_EVALUATION_REPORT)

    print("\n=== TRAINING COMPLETE ===")
    print(f"Best model: {best_name}")
    print(f"Test accuracy: {test_metrics['accuracy']}  Test macro-F1: {test_metrics['f1_macro']}  "
          f"Test weighted-F1: {test_metrics['f1_weighted']}")


if __name__ == "__main__":
    main()
