# Stage 4 - Frontend (Streamlit App)

The demo/evaluator-facing app. Reads only artifacts already produced by stages 1-3 (raw data,
trained model, reports) - running the app never re-runs extraction, labeling, or training.

```
frontend/
├── app.py                 Streamlit entrypoint - 5 tabs (see below)
├── predictor.py           Single-comment prediction (loads models/best_sentiment_model.joblib)
├── batch_predict.py       Batch (CSV/JSON) prediction
├── analytics.py           Sentiment summary stats + descriptive insights
├── visualizations.py      Bar / donut / trend charts (matplotlib, saved under reports/visualizations/)
├── model_insights.py      Loads models/model_metadata.json for the Model Performance tab
├── chatbot_service.py     Gemini-backed regulatory Q&A (additional feature)
├── context_manager.py     Loads data/regulatory_context/*.txt, picks the best-matching topic
└── prompts.py             System prompt / disclaimer text for the chatbot
```

## Run it

```bash
streamlit run frontend/app.py
```

## Tabs

1. **Single Comment Analysis** - paste (or one-click a "Try an example") a comment, get
   Positive/Negative/Neutral + confidence.
2. **Batch Analysis** - upload a CSV, get predictions for every row, download the results, see an
   inline sentiment breakdown chart.
3. **Sentiment Analytics** - dashboard over the last batch run (or the full cleaned dataset if no
   batch has been run yet): counts, percentages, bar/donut/trend charts, descriptive insights.
4. **Model Performance** - the numbers from `models/model_metadata.json` made visible: best model,
   test accuracy/macro-F1/weighted-F1, per-class precision/recall/F1, the confusion matrix image,
   class distribution, and the model-comparison table.
5. **Regulatory AI Assistant** (additional feature) - ask a free-text question about a selected
   U.S. financial regulation; answered via the Gemini API grounded in the matching text file under
   `data/regulatory_context/`. Requires `GEMINI_API_KEY` in `.env` (or, when deployed, in the
   host's secrets) - shows a clear in-app message instead of failing silently if it's not set.

The sidebar shows a live pipeline-status summary (extraction/labeling/cleaning/model) read
straight from the artifacts on disk.
