# Stage 4 - Frontend (Streamlit App)

The demo/evaluator-facing app. Reads only artifacts already produced by stages 1-3 (raw data,
trained model, reports) - running the app never re-runs extraction, labeling, or training.

```
frontend/
├── app.py                 Streamlit entrypoint - 6 tabs (see below)
├── theme.py                Shared color palette + "modern SaaS dashboard" CSS (dark sidebar,
│                           gradient header, card-style stats, restyled tabs/buttons/tables)
├── predictor.py            Single-comment prediction (loads models/best_sentiment_model.joblib)
├── batch_predict.py        Batch (CSV/JSON) prediction
├── analytics.py            Sentiment summary stats + descriptive insights
├── visualizations.py       Interactive Plotly charts: bar, donut, trend, confusion-matrix
│                           heatmap, model-comparison bar chart
├── wordclouds.py           "Most distinctive words per sentiment" word clouds
├── model_insights.py       Loads models/model_metadata.json for the Model Performance tab
├── chatbot_service.py      Gemini-backed regulatory Q&A (additional feature)
├── context_manager.py      Loads data/regulatory_context/*.txt, picks the best-matching topic
└── prompts.py               System prompt / disclaimer text for the chatbot
```

## Run it

```bash
streamlit run frontend/app.py
```

## Tabs

0. **Overview** - hero stat cards (comment count, accuracy, class count, AI assistant) and a
   4-stage pipeline visual, so the whole project is legible in five seconds before touching
   anything.
1. **Single Comment Analysis** - paste a comment, one-click a fixed example, or hit "Surprise me"
   for a random real comment from the dataset; get Positive/Negative/Neutral on a colored result
   card plus an interactive confidence gauge.
2. **Batch Analysis** - upload a CSV, get predictions for every row, download the results, see an
   inline interactive sentiment breakdown chart. `data/processed/test_split.csv` (130 real,
   held-out comments) works as a ready-made demo file.
3. **Sentiment Analytics** - dashboard over the last batch run (or the full cleaned dataset if no
   batch has been run yet): counts, percentages, interactive bar/donut/trend charts, descriptive
   insights, and a "most distinctive words per sentiment" word-cloud section.
4. **Model Performance** - the numbers from `models/model_metadata.json` made visible: best model,
   test accuracy/macro-F1/weighted-F1, per-class precision/recall/F1, an interactive confusion
   matrix heatmap, an interactive model-comparison chart + table, and class distribution.
5. **Regulatory AI Assistant** (additional feature) - ask a free-text question about a selected
   U.S. financial regulation; answered via the Gemini API grounded in the matching text file under
   `data/regulatory_context/`. Requires `GEMINI_API_KEY` in `.env` (or, when deployed, in the
   host's secrets) - shows a clear in-app message instead of failing silently if it's not set.

The sidebar shows a live pipeline-status summary (extraction/labeling/cleaning/model) read
straight from the artifacts on disk.

## New dependencies

`plotly` (interactive charts) and `wordcloud` (word-cloud images) - both in the root
`requirements.txt`, both install cleanly with prebuilt wheels, no compiler needed.
