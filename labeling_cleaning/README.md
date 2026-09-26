# Stage 2 - Local LLM Labeling (Ollama + Qwen3 8B) + Dataset Cleaning

Two steps that together turn the raw comments from `extraction/` into a clean, trained-model-ready
labeled dataset:

1. **`label_comments.py`** - one-time, fully offline sentiment labeling. Nothing is sent to any
   cloud AI service - everything runs on your own machine against a local Ollama server.
2. **`clean_dataset.py`** - validates and cleans the labeled output (invalid labels, duplicates,
   empty rows, length/class-imbalance diagnostics), producing the file `ml_pipeline/train_model.py`
   actually trains on.

## Requirements

- Python 3.10+ (deps in the project's root `requirements.txt`)
- [Ollama](https://ollama.com) installed and running locally (only needed for step 1)
- Model: `qwen3:8b` (fits an RTX 4060 8 GB with Ollama's default quantization)

```bash
ollama pull qwen3:8b
ollama serve
```

The labeling script checks Ollama is reachable and the model is installed before it starts; it
does not download the model for you.

**Important:** step 1 needs your GPU and a running local Ollama server, so it must be run
directly on this PC (not through a network-isolated remote bridge, which cannot reach
`127.0.0.1:11434`).

## Run it (from the project root)

```bash
pip install -r requirements.txt

# Step 1 - label (already run for real; produced data/labeled/labeled_comments.csv, 862/862, 0 failures)
python labeling_cleaning/label_comments.py --input data/raw/raw_financial_comments.json --output data/labeled/labeled_comments.csv

# Step 2 - clean (already run for real; produced data/processed/cleaned_labeled_comments.csv, 862/862 kept)
python -m labeling_cleaning.clean_dataset
```

At roughly 1-3 seconds per comment on an RTX 4060, a full labeling run of ~860 comments takes
around 20-40 minutes. Progress prints to the terminal as it goes:

```text
Processing comment 1/862
Label: Positive
```

### Try a small batch first (recommended, for future re-runs)

```bash
python labeling_cleaning/label_comments.py --input data/raw/raw_financial_comments.json --output data/labeled/labeled_comments_sample.csv --limit 20
```

### Resuming

Progress is saved to the output CSV every 5 comments (configurable via `SAVE_EVERY` in `.env`),
plus a `<output>.checkpoint.json` sidecar. If Ollama crashes, your PC restarts, or you just close
the terminal, **re-run the exact same command** - already-labeled comments are not re-sent to the
model.

### If some comments fail

Comments that get an invalid response after 3 retries are recorded in `failed_comments.json` at
the project root (not discarded silently) and labeling continues with the rest.

## How the model is prompted

`config.py` (project root) holds the exact system prompt: classify strictly as `Positive`,
`Negative`, or `Neutral`, based on the commenter's overall dominant opinion (not just individual
charged words), with Qwen3's "thinking" mode explicitly turned off (`/no_think`) so it returns a
plain label quickly instead of a reasoning trace. `label_comments.py` strips any stray
`<think>...</think>` block just in case, then validates the output is exactly one of the three
allowed labels before accepting it.

## What `clean_dataset.py` does

Validates labels against the allowed set, drops empty/null comments, removes exact duplicates,
normalizes text, flags (but doesn't auto-remove) very short comments, and reports the label
distribution and class-imbalance ratio. Writes:

- `data/processed/cleaned_labeled_comments.csv` - what `ml_pipeline/train_model.py` trains on
- `reports/dataset_quality_report.json` / `.txt`
- `reports/manual_review_sample.csv` - 50 random rows with a blank `human_reviewed_label` column,
  for an optional human spot-check

## Known limitation

Labels are generated automatically by a local LLM and have **not been exhaustively manually
verified** - this is disclosed in the main project README as well. Use
`reports/manual_review_sample.csv` as an optional spot-check workflow before trusting the numbers
reported to an evaluator.
