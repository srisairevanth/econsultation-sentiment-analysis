# Sentiment Analysis of Comments Received Through an E-Consultation Module

Academic mini project: automated sentiment analysis (Positive / Negative / Neutral) of
U.S. financial-regulatory public comments, plus an additional AI Regulatory Assistant.

## Pipeline

```
STAGE 1   extraction/                      Regulations.gov API -> data/raw/raw_financial_comments.json
STAGE 2   labeling_cleaning/               Ollama + Qwen3 8B, then cleaning -> data/processed/cleaned_labeled_comments.csv
STAGE 3   ml_pipeline/                     Cleaned dataset -> models/best_sentiment_model.joblib
STAGE 4   frontend/                        Streamlit app - sentiment classifier + regulatory chatbot (additional feature)
```

Four self-contained stage folders, one shared `config.py`, one shared `data/`. Each stage only
reads artifacts the previous stage already wrote to disk - running the app (Stage 4) never
re-runs extraction, labeling, or training.

> **The whole pipeline has been run for real, end to end**: 1,111 raw FTC (Federal Trade
> Commission) comments were pulled across two extraction batches, cleaned down to **862 real,
> substantive, de-duplicated comments**, labeled locally via Ollama + Qwen3 8B (zero failed
> comments), cleaned again (862 -> 862 rows, no duplicates or invalid labels), and used to train
> the production model - **89.2% test accuracy** (see **Most recent run on file** below).

> **Label noise disclosure:** training labels are generated automatically by a Local LLM
> (Ollama + Qwen3 8B) and may contain labeling noise. They have not been exhaustively manually
> verified. `labeling_cleaning/clean_dataset.py` exports `reports/manual_review_sample.csv` (50
> random rows with a blank `human_reviewed_label` column) as an optional manual spot-check
> workflow.

## Project structure

```
config.py                          central paths & constants shared by all 4 stages (RANDOM_STATE=42, allowed labels, Ollama/Gemini settings, etc.)
requirements.txt, .env.example, .gitignore, .streamlit/config.toml

extraction/                        STAGE 1 - Regulations.gov pull + raw cleaning (see its README)
  comments_extractor.py
  clean_raw_comments.py
  analyze_extracted_comments.py
  raw_output/                      untouched extraction batches, kept for provenance

labeling_cleaning/                 STAGE 2 - local Ollama/Qwen3 labeling + dataset cleaning (see its README)
  label_comments.py                offline sentiment labeling via local Ollama
  clean_dataset.py                 validation, cleaning, quality report
  text_utils.py                    shared normalization used by cleaning AND prediction

ml_pipeline/                       STAGE 3 - model training (see its README)
  train_model.py                   full pipeline: split -> tune -> compare -> select -> save
  compare_models.py                fair comparison (LogReg vs Linear SVM vs Naive Bayes)
  model_utils.py                   split logic, candidate pipelines, metrics, confusion-matrix plot
  evaluate_model.py                standalone re-evaluation of the saved model

frontend/                          STAGE 4 - Streamlit app, "modern dashboard" UI, 6 tabs (see its README)
  app.py                           entrypoint
  theme.py                         color palette + CSS (dark sidebar, gradient header, card stats)
  predictor.py, batch_predict.py   single / batch sentiment prediction
  analytics.py, visualizations.py  summary stats + interactive Plotly charts
  wordclouds.py                    "most distinctive words per sentiment" word clouds
  model_insights.py                loads models/model_metadata.json for the Model Performance tab
  chatbot_service.py, context_manager.py, prompts.py   Gemini-backed regulatory Q&A (additional feature)

scripts/
  make_sample_dataset.py           synthetic smoke-test data generator (NOT real data, optional dev utility)

data/
  raw/                             862 real, cleaned FTC comments (Stage 1 output)
  labeled/                         862 LLM-labeled comments (Stage 2a output)
  processed/                       cleaned_labeled_comments.csv + train/val/test_split.csv (Stage 2b/3 output)
  sample/                          synthetic smoke-test dataset (optional, not used by the real pipeline)
  regulatory_context/              starter regulation summary files for the AI assistant

models/                            best_sentiment_model.joblib + model_metadata.json (Stage 3 output)
reports/                           quality/evaluation reports + visualizations/ (Stage 2b/3 output)
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then fill in GEMINI_API_KEY (and REGULATIONS_API_KEY if re-extracting)
```

## Quickstart (evaluator demo - nothing needs to be re-run)

The trained model and real data are already committed, so you can go straight to the app:

```bash
pip install -r requirements.txt
streamlit run frontend/app.py
```

## Ollama setup (Stage 2, only needed to re-label from scratch)

```bash
ollama pull qwen3:8b
ollama serve
```

See `labeling_cleaning/README.md` for the exact labeling command.

## Gemini setup (Stage 4's additional feature)

1. Get a free API key at https://aistudio.google.com/apikey (no payment method required for the
   free tier; `gemini-3.8-flash`, the default model here, is free-of-charge with generous limits).
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY=...` (never commit `.env`).
3. Optionally set `GEMINI_MODEL` (defaults to `gemini-3.8-flash`). Google periodically retires
   older model names (e.g. `gemini-2.5-flash` was retired for new users after this project was
   first built) - if the chatbot tab ever starts returning errors, check
   https://ai.google.dev/gemini-api/docs/models for the current free-tier flash model name and
   update `GEMINI_MODEL` in `.env` (no code changes needed).

## Commands (full pipeline, if re-running from scratch)

```bash
# Stage 1 (already run) - pull more comments if you want to expand the dataset later
python extraction/comments_extractor.py --agency-id CFPB --limit 2000 --output extraction/raw_output/batch3_cfpb.json
python extraction/clean_raw_comments.py

# Stage 2a - label the real data (see labeling_cleaning/README.md - requires local Ollama)
python labeling_cleaning/label_comments.py --input data/raw/raw_financial_comments.json --output data/labeled/labeled_comments.csv

# Stage 2b - verify & clean the labeled dataset (auto-discovers data/labeled/labeled_comments.csv)
python -m labeling_cleaning.clean_dataset

# Stage 3 - train the model (TF-IDF+LogReg baseline, tuned, compared against SVM/NB)
python -m ml_pipeline.train_model
python -m ml_pipeline.evaluate_model                       # re-evaluate the saved model on the test split

# Stage 4 - prediction (CLI) or the full app
python frontend/predictor.py "This proposal creates unnecessary compliance costs."
streamlit run frontend/app.py

# (optional) regenerate the synthetic smoke-test dataset used to verify the pipeline wiring
python scripts/make_sample_dataset.py
```

## Model selection

Candidates (all TF-IDF-based, `random_state=42`, stratified 70/15/15 split with an 80/20 +
cross-validation fallback for small datasets): **TF-IDF + Logistic Regression** (tuned via a
lightweight `GridSearchCV` on the training split only), **TF-IDF + Linear SVM** (wrapped in
`CalibratedClassifierCV` so it can still report a confidence score), and **TF-IDF + Multinomial
Naive Bayes**. All three are trained on the identical training split and compared on the
identical validation split; the model with the highest **Macro F1** (ties broken by Weighted F1,
then Accuracy) is refit on train+validation and evaluated once on the untouched test set. This
avoids data leakage: the test set is never seen during training, tuning, or model selection.

### Most recent run on file (real 862-comment dataset)

- Best model selected: **TF-IDF + Linear SVM** (beat tuned Logistic Regression and Multinomial NB
  on the validation split's Macro F1)
- Dataset: 862 real FTC comments, labeled by Ollama + Qwen3 8B -> 862 after cleaning (no
  duplicates, no invalid labels, 3 short comments flagged but not removed)
- Label distribution: Negative 602 (69.8%), Neutral 177 (20.5%), Positive 83 (9.6%) - class
  imbalance ratio 7.25:1, mitigated with `class_weight="balanced"` and Macro F1 as the selection
  metric
- Split: stratified 70/15/15 (train 603 / val 129 / test 130), `random_state=42`
- Test set: **Accuracy 0.892, Macro F1 0.779, Weighted F1 0.884**
- Full breakdown (per-class precision/recall/F1, confusion matrix) in
  `reports/baseline_evaluation.json`, `reports/visualizations/confusion_matrix_test.png`, or live
  in the app's **Model Performance** tab

## Deploy to Streamlit Community Cloud

The model, cleaned data, and reports are committed to the repo (see `.gitignore`), so a fresh
deploy needs no GPU, no Ollama, and no training step - it just loads what's already there.

1. Push this folder to a GitHub repository (the local folder is named `MINI Project`; the GitHub
   repo name can be anything, e.g. `sentiment-econsultation`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and click "New app".
3. Pick the repo/branch, and set **Main file path** to `frontend/app.py`.
4. Under the app's **Settings -> Secrets**, add (TOML format):
   ```toml
   GEMINI_API_KEY = "your_real_key_here"
   GEMINI_MODEL = "gemini-3.8-flash"
   ```
   Streamlit Cloud exposes Secrets as environment variables to the running app, so
   `config.py`'s existing `os.getenv("GEMINI_API_KEY", "")` picks it up with no code changes.
5. Deploy. Everything except the Regulatory AI Assistant tab works with zero configuration; that
   tab needs the secret above to answer questions (it shows a clear in-app message until then).

## Limitations / known issues

- Confidence scores for Linear SVM come from Platt-scaling calibration (`CalibratedClassifierCV`),
  an approximation, not a native probability.
- The regulatory context store is intentionally simple (plain text files, keyword matching, no
  embeddings/vector DB) per the agreed architecture; the three provided files are general
  educational placeholders and should be replaced/extended with content matched to the actual
  dockets collected by the extraction module.
- Sentiment-trend-over-time charts require a real date column in the input data; none is
  fabricated when one isn't present.
- The Gemini-powered assistant requires network access and a valid `GEMINI_API_KEY`; without one
  it returns a clear in-app error rather than failing silently.
- Class imbalance (Negative comments dominate the real dataset, 7.25:1 vs the minority class) is
  mitigated with `class_weight="balanced"` and Macro F1 as the model-selection metric, but is not
  eliminated - Positive-class recall (0.42 on the test set) is the model's weakest point, a direct
  consequence of only 83 Positive examples in the full dataset.
