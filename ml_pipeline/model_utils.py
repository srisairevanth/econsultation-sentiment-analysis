"""
Shared helpers for the training pipeline: data loading/splitting, candidate
model definitions, and evaluation. Used by train_model.py, compare_models.py
and evaluate_model.py so all three stay consistent.
"""
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


def load_cleaned_dataset(path: Path = config.CLEANED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {path}. Run "
            f"'python -m labeling_cleaning.clean_dataset' first."
        )
    df = pd.read_csv(path)
    if "comment" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Cleaned dataset at {path} is missing required 'comment'/'label' columns.")
    df = df.dropna(subset=["comment", "label"])
    df = df[df["label"].isin(config.ALLOWED_LABELS)]
    return df.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, random_state: int = config.RANDOM_STATE):
    """
    Returns (train_df, val_df, test_df, split_info).
    Uses a 70/15/15 stratified split when the dataset is large enough to make a
    3-way split meaningful; otherwise falls back to an 80/20 train/test split
    and hyperparameter tuning is done via cross-validation on the train set
    instead of a dedicated validation split.
    """
    n = len(df)
    class_counts = df["label"].value_counts()
    min_class_count = class_counts.min() if len(class_counts) else 0

    if n >= config.MIN_ROWS_FOR_3WAY_SPLIT and min_class_count >= 10:
        train_df, temp_df = train_test_split(
            df, test_size=(config.VAL_SIZE + config.TEST_SIZE),
            stratify=df["label"], random_state=random_state,
        )
        relative_test_size = config.TEST_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
        val_df, test_df = train_test_split(
            temp_df, test_size=relative_test_size,
            stratify=temp_df["label"], random_state=random_state,
        )
        strategy = "stratified_70_15_15"
    else:
        logger.warning(
            "Dataset too small (%d rows, smallest class=%d) for a reliable 3-way "
            "70/15/15 split. Falling back to an 80/20 stratified train/test split; "
            "hyperparameter tuning will use cross-validation on the training set.",
            n, min_class_count,
        )
        try:
            train_df, test_df = train_test_split(
                df, test_size=0.20, stratify=df["label"], random_state=random_state,
            )
        except ValueError:
            # a class has too few members even for stratified 80/20 - fall back further
            logger.warning("Stratified split failed (a class has too few members); using a plain split.")
            train_df, test_df = train_test_split(df, test_size=0.20, random_state=random_state)
        val_df = pd.DataFrame(columns=df.columns)
        strategy = "fallback_80_20"

    split_info = {
        "strategy": strategy,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "random_state": random_state,
    }
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True),
            test_df.reset_index(drop=True), split_info)


def build_candidate_pipelines(random_state: int = config.RANDOM_STATE) -> Dict[str, Pipeline]:
    """
    Candidate models, all sharing the same conservative TF-IDF configuration by
    default (individually tunable in train_model.py via GridSearchCV).
    """
    tfidf_kwargs = dict(ngram_range=(1, 2), min_df=2, max_df=0.95,
                         sublinear_tf=True, max_features=20000)

    logreg_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                    random_state=random_state)),
    ])

    # LinearSVC has no predict_proba; wrap with Platt-scaling calibration so we
    # can still report a confidence score (Part 5.1 requires this where possible).
    svm_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
        ("clf", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=random_state), cv=3)),
    ])

    nb_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
        ("clf", MultinomialNB()),
    ])

    return {
        "tfidf_logreg": logreg_pipeline,
        "tfidf_linear_svm": svm_pipeline,
        "tfidf_multinomial_nb": nb_pipeline,
    }


def evaluate_predictions(y_true, y_pred, labels=config.ALLOWED_LABELS) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    per_class = {
        labels[i]: {
            "precision": round(float(precision_per_class[i]), 4),
            "recall": round(float(recall_per_class[i]), 4),
            "f1": round(float(f1_per_class[i]), 4),
            "support": int(support_per_class[i]),
        }
        for i in range(len(labels))
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(precision_weighted), 4),
        "recall_weighted": round(float(recall_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
    }


def plot_confusion_matrix(cm: list, labels: list, title: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
