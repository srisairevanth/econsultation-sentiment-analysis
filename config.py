"""
Central configuration for the Sentiment Analysis of E-Consultation Comments project.

All paths are resolved relative to the project root (this file's directory) so the
code works regardless of the current working directory it is invoked from, and
regardless of what the project root folder itself is named.

This single file is the shared source of truth for all four pipeline stages
(extraction/, labeling_cleaning/, ml_pipeline/, frontend/). Environment variables
(see .env.example) can override the Gemini API key, the Regulations.gov API key
(used by extraction/), and the local Ollama labeling settings (used by
labeling_cleaning/label_comments.py).
"""
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Allowed sentiment labels (strict, exactly three)
# ---------------------------------------------------------------------------
ALLOWED_LABELS = ["Positive", "Negative", "Neutral"]

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LABELED_DIR = DATA_DIR / "labeled"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
REGULATORY_CONTEXT_DIR = DATA_DIR / "regulatory_context"

# Candidate locations/filenames for the raw comments produced by extraction/.
RAW_DATA_CANDIDATES = [
    RAW_DIR / "raw_financial_comments.json",
    PROJECT_ROOT / "raw_financial_comments.json",
]

# Optional explicit override for where the labeled dataset lives - set this in
# .env only if it's ever produced somewhere other than this project's own
# data/labeled/ folder.
_env_labeled_path = os.getenv("LABELED_DATASET_PATH", "").strip()

LABELED_DATA_CANDIDATES = [
    *([Path(_env_labeled_path)] if _env_labeled_path else []),
    LABELED_DIR / "labeled_comments.csv",
]

CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_labeled_comments.csv"
SAMPLE_DATASET_PATH = SAMPLE_DIR / "sample_labeled_comments.csv"

# ---------------------------------------------------------------------------
# Cleaning configuration (Stage 2 - labeling_cleaning/clean_dataset.py)
# ---------------------------------------------------------------------------
MIN_COMMENT_LENGTH_CHARS = 10          # below this, comment is flagged (not auto-removed)
MIN_COMMENT_LENGTH_WORDS = 3
CLASS_IMBALANCE_RATIO_THRESHOLD = 3.0  # majority:minority ratio above this => flagged
INVALID_LABEL_ACTION = "remove"        # "remove" or "flag" - see clean_dataset.py


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


# ---------------------------------------------------------------------------
# Local LLM labeling configuration (Stage 2 - labeling_cleaning/label_comments.py)
# Offline, one-time sentiment labeling via a local Ollama server. Nothing here
# is sent to any cloud AI service.
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_OUTPUT_CSV = LABELED_DIR / "labeled_comments.csv"
DEFAULT_FAILED_PATH = PROJECT_ROOT / "failed_comments.json"

# Nested JSON paths checked in order (Regulations.gov uses attributes.comment).
JSON_COMMENT_PATHS = (
    ("attributes", "comment"),
    ("attributes", "commentText"),
    ("comment",),
    ("commentText",),
    ("comment_text",),
    ("text",),
    ("body",),
    ("content",),
)

OLLAMA_HOST = env_str("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = env_str("OLLAMA_MODEL", DEFAULT_MODEL)
OLLAMA_TEMPERATURE = env_float("OLLAMA_TEMPERATURE", 0.0)
OLLAMA_SEED = env_int("OLLAMA_SEED", 42)
SAVE_EVERY = env_int("SAVE_EVERY", 5)
MAX_RETRIES = env_int("MAX_RETRIES", 3)
COMMENT_COLUMN = env_str("COMMENT_COLUMN", "comment")

SYSTEM_PROMPT = """You are a sentiment classification system for public comments submitted regarding U.S. government financial regulations.

Your task is to classify the overall sentiment expressed in the given public comment.

Allowed labels are strictly:

Positive
Negative
Neutral

Definitions:

* Positive: The comment expresses support, approval, satisfaction, favorable opinion, or positive expectations regarding the regulation or policy.
* Negative: The comment expresses opposition, criticism, dissatisfaction, concern, complaint, or unfavorable opinion regarding the regulation or policy.
* Neutral: The comment is primarily informational, factual, mixed without a clear dominant sentiment, asks for clarification, or does not express a clear positive or negative opinion.

Rules:

1. Analyze the meaning and context of the complete comment.
2. Do not classify based only on individual positive or negative words.
3. Consider the overall opinion of the commenter.
4. If a comment contains both positive and negative opinions, select the overall dominant sentiment. If there is no clearly dominant positive or negative sentiment, use Neutral.
5. Return exactly one of: Positive, Negative, or Neutral.
6. Do not provide an explanation.
7. Do not return reasoning.
8. Do not return Markdown.
9. The output must contain only the sentiment label.
"""

# ---------------------------------------------------------------------------
# Model paths (Stage 3 - ml_pipeline/)
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_sentiment_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Train/validation/test split proportions
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
# Datasets smaller than this fall back to a simpler 80/20 train/test split
# (with cross-validation used for tuning instead of a held-out validation set)
MIN_ROWS_FOR_3WAY_SPLIT = 150

# ---------------------------------------------------------------------------
# Reports (Stages 2-3)
# ---------------------------------------------------------------------------
REPORTS_DIR = PROJECT_ROOT / "reports"
VISUALIZATIONS_DIR = REPORTS_DIR / "visualizations"
DATASET_QUALITY_REPORT_JSON = REPORTS_DIR / "dataset_quality_report.json"
DATASET_QUALITY_REPORT_TXT = REPORTS_DIR / "dataset_quality_report.txt"
BASELINE_EVALUATION_REPORT = REPORTS_DIR / "baseline_evaluation.json"
MODEL_COMPARISON_REPORT = REPORTS_DIR / "model_comparison.json"
MANUAL_REVIEW_SAMPLE_PATH = REPORTS_DIR / "manual_review_sample.csv"

# ---------------------------------------------------------------------------
# Chatbot / Regulatory Assistant (Stage 4 - frontend/, additional feature)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Tried in order when the primary model is overloaded (503) or unavailable. Verified live: when
# gemini-3.8-flash / gemini-flash-latest returned "high demand", gemini-flash-lite-latest still
# answered in ~2s.
GEMINI_FALLBACK_MODELS = [
    m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-flash-lite-latest,gemini-flash-latest").split(",")
    if m.strip()
]
GEMINI_ATTEMPT_TIMEOUT_S = env_float("GEMINI_ATTEMPT_TIMEOUT_S", 12.0)   # one API call (normal answers take 2-9s)
GEMINI_TOTAL_BUDGET_S = env_float("GEMINI_TOTAL_BUDGET_S", 50.0)         # whole question, all retries
# Extra same-model tries on 5xx/timeout. 0 on purpose: an overloaded model stays overloaded for
# minutes, so waiting on it again just delays the fallback model that would answer in ~2s.
GEMINI_RETRIES_PER_MODEL = env_int("GEMINI_RETRIES_PER_MODEL", 0)
MAX_CHAT_HISTORY_MESSAGES = 6  # trimmed to keep the context window bounded

for _d in [DATA_DIR, RAW_DIR, LABELED_DIR, PROCESSED_DIR, SAMPLE_DIR, REGULATORY_CONTEXT_DIR,
           MODELS_DIR, REPORTS_DIR, VISUALIZATIONS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
