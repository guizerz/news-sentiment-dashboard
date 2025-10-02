# news-sentiment-dashboard
# 🗞️ News Sentiment Trading Dashboard (FinBERT + Streamlit)

Quant-style dashboard that:
- Fetches finance headlines (Yahoo Finance RSS)
- Scores sentiment with FinBERT (finance domain) or VADER fallback
- Visualizes distributions and runs a naive sentiment backtest

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # create venv
pip install -r requirements.txt                     # install deps
streamlit run app.py                                # run dashboard
