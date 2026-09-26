# Part 1 - Regulations.gov Extraction (+ Part 1.5 Raw Cleaning)

Pulls real public comments from the U.S. **Regulations.gov API v4** and prepares them for
LLM labeling. This is a one-time data-collection step; it does not run every time the app
is demoed.

## What's in this folder

```
extraction/
├── comments_extractor.py         Pulls raw comments from the Regulations.gov API
├── analyze_extracted_comments.py Quality report on a raw JSON dump (text vs attachment-only vs empty)
├── clean_raw_comments.py         Merges batches, drops unusable rows, writes the final raw file
├── raw_output/                   Untouched extraction batches (kept for provenance/audit trail)
│   ├── batch1_ftc_111.json       First extraction run - 111 FTC comments
│   └── batch2_ftc_1000.json      Second extraction run - 1000 FTC comments
└── raw_cleaning_report.txt       Output of the last clean_raw_comments.py run (counts, see below)
```

Dependencies (`requests`, `python-dotenv`) and the `REGULATIONS_API_KEY` setting live in the
project's root `requirements.txt` / `.env.example` - there is no separate copy in this folder.

## Part 1 - what was pulled

Both batches were pulled from the **Federal Trade Commission (FTC)** docket comments via
`comments_extractor.py`, which fetches comment metadata then the full comment body text for
each one, paginating around the API's 20-page cap and retrying on rate limits automatically.
1,111 unique comments were collected across the two runs (no overlapping IDs between them).

To pull more data later (e.g. from SEC, CFPB, FDIC, OCC, or the Federal Reserve), run e.g.:

```bash
python comments_extractor.py --agency-id CFPB --limit 2000 --output raw_output/batch3_cfpb.json
```

then add the new file to `INPUT_BATCHES` in `clean_raw_comments.py` and re-run it.

## Part 1.5 - why raw cleaning exists

Regulations.gov comments include a meaningful fraction that carry **no usable sentiment at
all** - most commonly a submitter who wrote nothing but "See attached file(s)" because their
real opinion is in an uploaded PDF the API doesn't expose as text. Feeding those into the LLM
labeling step would waste time and adds label noise (an LLM asked to judge sentiment on "See
attached" has nothing to go on). `clean_raw_comments.py`:

1. Merges every file listed in `INPUT_BATCHES`, de-duplicating by Regulations.gov comment ID.
2. Drops null/empty comment bodies.
3. Decodes leftover HTML entities/tags from the API response (`&rsquo;`, `<br/>`, etc.) into
   clean plain text.
4. Drops comments that are **short and purely an attachment pointer** (e.g. "See attached",
   "Please see attached comment.") - but keeps longer comments that merely *mention* an
   attachment while also containing real inline argument/opinion text (e.g. cover letters
   quoting the submitter's actual position).
5. Drops exact-duplicate text (after normalization) - a handful of comments were submitted
   twice.

Run it whenever `raw_output/` changes:

```bash
python clean_raw_comments.py
```

It writes the final, ready-to-label file to `../data/raw/raw_financial_comments.json` and a
plain-text count report to `raw_cleaning_report.txt`.

### Latest run

See `raw_cleaning_report.txt` in this folder for the exact counts from the most recent run.
