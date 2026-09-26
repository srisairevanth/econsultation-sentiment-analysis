"""
"Most distinctive words per sentiment" word clouds.

Uses a document-frequency RATIO (how much more often a word appears in this
class's comments vs. the rest of the dataset), not raw term counts. Raw
frequency was tried first and just surfaced generic regulatory boilerplate
("comment", "rule", "ftc") and whatever a single large cluster of near-
duplicate comments happened to repeat - not useful or distinctive. The ratio
approach, verified against the real 862-comment dataset, surfaces genuinely
class-specific vocabulary instead.

Frequency-based only (not tied to any specific model's internals), so this
keeps working regardless of which model type ml_pipeline currently selects.
"""
import re
from collections import Counter
from typing import Dict

import pandas as pd
from wordcloud import STOPWORDS, WordCloud

EXTRA_STOPWORDS = {
    "ftc", "commission", "comment", "comments", "rule", "regulation",
    "proposed", "regulations", "federal", "trade", "will", "would",
    "also", "please", "one", "us", "government", "com", "https", "http",
}
ALL_STOPWORDS = STOPWORDS.union(EXTRA_STOPWORDS)

_TOKEN_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> set:
    words = _TOKEN_RE.findall(text.lower())
    return {w for w in words if len(w) > 2 and w not in ALL_STOPWORDS}


def distinctive_word_scores(df: pd.DataFrame, label: str, min_docs: int = 3) -> Dict[str, float]:
    """Score = (word's document-frequency rate inside `label`) / (its rate outside `label`)."""
    class_docs = df.loc[df["label"] == label, "comment"].astype(str)
    other_docs = df.loc[df["label"] != label, "comment"].astype(str)
    n_class, n_other = len(class_docs), len(other_docs)
    if n_class == 0:
        return {}

    class_counts: Counter = Counter()
    for text in class_docs:
        class_counts.update(_tokenize(text))
    other_counts: Counter = Counter()
    for text in other_docs:
        other_counts.update(_tokenize(text))

    scores = {}
    for word, count in class_counts.items():
        if count < min_docs:
            continue
        class_rate = count / n_class
        other_rate = (other_counts.get(word, 0) + 1) / (n_other + 1)
        scores[word] = class_rate / other_rate
    return scores


def generate_wordcloud_image(df: pd.DataFrame, label: str, colormap: str, max_words: int = 40):
    """Returns a PIL Image of the word cloud, or None if there's not enough data yet."""
    scores = distinctive_word_scores(df, label)
    if not scores:
        return None
    wc = WordCloud(
        width=600, height=350, background_color="white",
        colormap=colormap, max_words=max_words, collocations=False,
        prefer_horizontal=0.9,
    )
    wc.generate_from_frequencies(scores)
    return wc.to_image()
