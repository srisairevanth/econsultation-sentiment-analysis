# Viva / Panel Prep — Preprocessing & Model Questions

Quick-reference answers based on exactly what the code does (Parts 3 & 4).
Speak from this in your own words — don't read it verbatim.

---

## Q: "Walk us through your preprocessing steps."

Say this as a numbered story — it maps 1:1 to `preprocessing/clean_dataset.py`:

1. **Load the LLM-labeled data.** The dataset comes from the earlier stage
   (Ollama + Qwen3 8B labeling the Regulations.gov comments as Positive /
   Negative / Neutral). The cleaning script auto-detects the file regardless
   of exact column names (comment/text/body, label/sentiment/class).

2. **Label validation.** Strip whitespace, keep only rows whose label is
   exactly one of the three allowed classes. Anything else (typos, blank
   labels, stray LLM output) is removed and counted in the report.

3. **Remove empty/null comments.** Catches None, NaN, empty strings,
   whitespace-only, and strings made of only punctuation/symbols.

4. **Duplicate removal.** Exact duplicate comment text is dropped (keep the
   first occurrence). Duplicate (comment, label) pairs are counted
   separately in the report for transparency.

5. **Text normalization** (`preprocessing/text_utils.py`) — deliberately
   *conservative*:
   - Unicode NFKC normalization (fixes encoding artifacts / mojibake)
   - Strip control characters
   - Collapse repeated whitespace, trim
   - **Does NOT** lowercase, strip punctuation, remove stopwords, or touch
     negation words (not/no/never). Reason: TF-IDF's vectorizer lowercases
     for you, and punctuation like "!" "?" plus negation words carry real
     sentiment signal — stripping them would hurt accuracy, not help it.

6. **Re-check for empties after normalization** — a comment that was only
   symbols could become empty after cleaning; those are removed too.

7. **Length analysis.** Compute char/word length stats (min/max/mean/median/
   percentiles). Comments under 10 characters or 3 words are **flagged**,
   not deleted — a short comment like "Terrible service" is still valid
   signal, so it's kept but marked for optional manual review.

8. **Label distribution & class imbalance check.** Count/percentage per
   class; compute majority:minority ratio. If ≥ 3.0, it's flagged as
   imbalanced with a recommendation to use `class_weight='balanced'` and
   Macro-F1 as the primary metric — no synthetic oversampling (e.g. SMOTE)
   was applied automatically, that's a conscious choice.

9. **Outputs:** a cleaned CSV, a machine-readable JSON quality report, a
   human-readable TXT report, and an optional random sample (default 50
   rows) exported with a blank column for manual spot-checking of the LLM's
   labels — since Part 3's own note says LLM labels "may contain labeling
   noise" and haven't been exhaustively human-verified.

**If asked "why didn't you do stemming/lemmatization/stopword removal?"**
→ Because this is a TF-IDF + classical ML pipeline, not a bag-of-words
frequency-only model. Stopwords like "not" and punctuation like "!" are
sentiment-bearing. Aggressive NLP preprocessing is a classic mistake that
quietly deletes the signal you're trying to classify.

---

## Q: "Which models did you use, and why?"

Say: "I compared three classical ML models on the same TF-IDF features,
rather than committing to one blindly."

**Feature extraction (same for all 3, for a fair comparison):**
TF-IDF vectorizer — unigrams + bigrams (`ngram_range=(1,2)`), `min_df=2`,
`max_df=0.95`, `sublinear_tf=True`, capped at 20,000 features.

**Three candidates (`training/model_utils.py`):**
1. **Logistic Regression** — `max_iter=1000`, `class_weight='balanced'`
2. **Linear SVM** (`LinearSVC`, `class_weight='balanced'`) — wrapped in
   `CalibratedClassifierCV` (3-fold Platt scaling) because a raw linear SVM
   doesn't output probabilities, and the project needs a confidence score
   for every prediction (Part 5 requirement).
3. **Multinomial Naive Bayes** — the classic, fast text-classification
   baseline.

**If asked "why not BERT / a transformer / deep learning?"**
→ This is literally addressed in slide 6 (Research Gaps) of the
presentation: transformer models are accurate but need heavy compute that's
impractical on a typical laptop for this project's scope, and lexicon tools
like VADER are too crude for formal regulatory language. TF-IDF + classical
ML is the practical, interpretable middle ground — fast to train, easy to
explain to a panel, and strong on this kind of formal short-text data.

**Data splitting & leakage prevention:**
- Stratified 70/15/15 train/validation/test split (class proportions
  preserved in every split). Falls back to an 80/20 split + 3-fold
  cross-validation if the dataset is small (< 150 rows or a class has
  fewer than 10 examples).
- All three models are trained on the **exact same training split** and
  compared on the **exact same validation split** — never the test set.
- TF-IDF is fit only inside each model's own pipeline on the training fold
  (`fit` on train, `transform` only on val/test) — so vocabulary/IDF
  weights never see validation or test text. That's the "no data leakage"
  guarantee.
- `random_state=42` fixed everywhere for reproducibility.

**Model selection:**
Ranked by **Macro F1** first (treats all 3 classes equally regardless of
how imbalanced the dataset is), then Weighted F1, then Accuracy as
tie-breakers. Whichever wins on validation is retrained/finalized and then
evaluated **once** on the held-out test set — a number that was never used
to pick the model, so it's an honest, unbiased final result.

**If asked "why Macro F1 and not just accuracy?"**
→ Accuracy can look great on an imbalanced dataset just by always
predicting the majority class. Macro F1 forces the model to do well on
Positive, Negative, *and* Neutral individually — which matters here because
public comments are rarely evenly split across sentiments.

---

## One-line summary if they want it fast

"I cleaned the LLM-labeled comments — validating labels, removing empties
and duplicates, conservatively normalizing text without destroying
sentiment cues, and flagging short comments and class imbalance in a
quality report. Then I compared Logistic Regression, Linear SVM, and Naive
Bayes on identical TF-IDF features using a stratified 70/15/15 split, with
no leakage since TF-IDF is fit only on the training fold, and picked the
best one by Macro F1 on validation before a single honest test-set check."
