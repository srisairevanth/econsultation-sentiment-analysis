"""
Generates a SMALL, CLEARLY SYNTHETIC sample dataset purely to smoke-test the
Stage 2-4 pipeline (cleaning -> training -> prediction -> analytics) end to end
before the real Regulations.gov + Local LLM labeled dataset is available.

This is NOT real regulatory data and must not be presented as such. It is
saved separately under data/sample/ and is never picked up automatically by
labeling_cleaning.clean_dataset (which only looks at config.LABELED_DATA_CANDIDATES).

Run:
    python scripts/make_sample_dataset.py
"""
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

random.seed(config.RANDOM_STATE)

POSITIVE_TEMPLATES = [
    "This proposed rule will provide much needed protection for {group}.",
    "I strongly support this regulation because it improves transparency for {group}.",
    "This is a well-designed rule that benefits {group} without excessive burden.",
    "The proposed changes will make {topic} safer and more accessible for {group}.",
    "I am glad the agency is taking action to protect {group} from unfair practices.",
    "This regulation strikes a fair balance and should be finalized as written.",
    "Thank you for proposing clearer disclosure requirements around {topic}.",
    "These reforms will increase trust in {topic} and help {group} make better decisions.",
]
NEGATIVE_TEMPLATES = [
    "This regulation will create unnecessary compliance costs for {group}.",
    "The proposed rule is overly burdensome and will hurt small {group}.",
    "I strongly oppose this rule because it will reduce access to {topic} for {group}.",
    "This proposal is poorly designed and will cause more harm than good to {group}.",
    "The compliance timeline is unrealistic and will force {group} out of the market.",
    "This rule adds red tape without meaningfully improving {topic}.",
    "I am concerned this regulation will raise costs for {group} with little benefit.",
    "The agency has not adequately considered the negative impact on {group}.",
]
NEUTRAL_TEMPLATES = [
    "Please provide additional clarification regarding the implementation of {topic}.",
    "It is unclear how this rule will apply to {group} in practice.",
    "We request more time to review the proposed changes to {topic}.",
    "Could the agency clarify the effective date for {group}?",
    "The comment period should be extended to allow further analysis of {topic}.",
    "We are submitting this comment to request additional data on {topic}.",
    "More examples would help {group} understand how to comply with {topic}.",
    "This comment requests technical clarification and does not take a position on {topic}.",
]
GROUPS = ["consumers", "small banks", "credit unions", "borrowers", "community lenders",
          "fintech startups", "investors", "low-income households"]
TOPICS = ["mortgage disclosures", "credit card fees", "data privacy rules", "derivatives clearing",
          "consumer lending standards", "capital requirements", "overdraft practices", "loan servicing"]


def make_rows(templates, label, n):
    rows = []
    for _ in range(n):
        t = random.choice(templates)
        text = t.format(group=random.choice(GROUPS), topic=random.choice(TOPICS))
        rows.append({"comment": text, "label": label})
    return rows


def main():
    rows = []
    rows += make_rows(POSITIVE_TEMPLATES, "Positive", 110)
    rows += make_rows(NEGATIVE_TEMPLATES, "Negative", 100)
    rows += make_rows(NEUTRAL_TEMPLATES, "Neutral", 90)

    # Inject data-quality issues on purpose, to exercise the cleaning module:
    rows.append({"comment": "", "label": "Positive"})                       # empty
    rows.append({"comment": "   ", "label": "Negative"})                    # whitespace only
    rows.append({"comment": None, "label": "Neutral"})                      # null
    rows.append({"comment": "This proposal is beneficial for consumers.", "label": "Mixed"})  # invalid label
    rows.append({"comment": "...", "label": "Positive"})                    # meaningless
    dup = rows[5]
    rows.append(dict(dup))  # exact duplicate

    random.shuffle(rows)
    df = pd.DataFrame(rows)
    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.SAMPLE_DATASET_PATH, index=False)
    print(f"Wrote {len(df)} synthetic sample rows to {config.SAMPLE_DATASET_PATH}")


if __name__ == "__main__":
    main()
