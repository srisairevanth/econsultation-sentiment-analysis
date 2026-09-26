"""
Part 6.1: Regulatory context management.

Deliberately simple, transparent design (no embeddings/vector DB, as agreed):
each regulation/topic is one plain text file under data/regulatory_context/.
The user (or app) selects a regulation by id/filename; a lightweight keyword
scorer can also suggest the best-matching document for a free-text question.
"""
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


def list_available_regulations() -> List[Dict[str, str]]:
    """Returns [{'id': filename_stem, 'title': first_line, 'path': str}, ...]."""
    if not config.REGULATORY_CONTEXT_DIR.exists():
        return []
    items = []
    for path in sorted(config.REGULATORY_CONTEXT_DIR.glob("*.txt")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip().lstrip("#").strip()
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
            first_line = path.stem
        items.append({"id": path.stem, "title": first_line or path.stem, "path": str(path)})
    return items


def get_context(regulation_id: str) -> Optional[str]:
    """Load the full text of a regulation/topic by its id (filename stem)."""
    path = config.REGULATORY_CONTEXT_DIR / f"{regulation_id}.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def suggest_regulation_for_question(question: str) -> Optional[str]:
    """Very simple keyword-overlap scorer to suggest the most relevant regulation id."""
    regulations = list_available_regulations()
    if not regulations:
        return None
    question_words = set(re.findall(r"[a-zA-Z]{3,}", question.lower()))
    best_id, best_score = None, 0
    for reg in regulations:
        text = get_context(reg["id"]) or ""
        text_words = set(re.findall(r"[a-zA-Z]{3,}", text.lower()))
        score = len(question_words & text_words)
        if score > best_score:
            best_id, best_score = reg["id"], score
    return best_id
