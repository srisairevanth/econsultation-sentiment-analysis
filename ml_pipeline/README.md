# Stage 3 - ML Pipeline (Training)

Trains the actual sentiment classifier on `data/processed/cleaned_labeled_comments.csv`
(produced by `labeling_cleaning/`).

```
ml_pipeline/
├── train_model.py       End-to-end pipeline: split -> tune -> compare -> select -> evaluate -> save
├── compare_models.py    TF-IDF+LogReg vs Linear SVM vs Multinomial NB on the same split
├── model_utils.py       Shared split/evaluation/plotting helpers
└── evaluate_model.py    Re-evaluate the already-saved model independently
```

## Run it (from the project root)

```bash
python -m ml_pipeline.train_model
python -m ml_pipeline.train_model --skip-comparison       # baseline only, faster
python -m ml_pipeline.evaluate_model                       # re-evaluate the saved model
python -m ml_pipeline.evaluate_model --dataset other.csv   # or on any other labeled CSV
```

Already run for real: best model **TF-IDF + Linear SVM**, test accuracy **89.2%**, macro-F1
**0.78** on the real 862-comment dataset (see `models/model_metadata.json` and
`reports/baseline_evaluation.json` for the full breakdown, or the "Model Performance" tab in the
frontend app).

## Model selection

Candidates (all TF-IDF-based, `random_state=42`, stratified 70/15/15 split with an 80/20 +
cross-validation fallback for small datasets): TF-IDF + Logistic Regression (tuned via a
lightweight `GridSearchCV` on the training split only), TF-IDF + Linear SVM (wrapped in
`CalibratedClassifierCV` so it can still report a confidence score), and TF-IDF + Multinomial
Naive Bayes. All three are trained on the identical training split and compared on the identical
validation split; the model with the highest Macro F1 (ties broken by Weighted F1, then Accuracy)
is refit on train+validation and evaluated once on the untouched test set. This avoids data
leakage: the test set is never seen during training, tuning, or model selection.
