from __future__ import annotations
import streamlit as st
import pandas as pd
import altair as alt
from sentiment import finbert_scores, label_from_scores, score_scalar
from data_sources import fetch_rss_headlines, load_prices
from strategy import daily_sentiment_aggregate, backtest

st.set_page_config(page_title="News Sentiment Trading", layout="wide")

st.title("🗞️ News Sentiment Trading Dashboard")
st.caption("FinBERT-scored headlines + simple signal backtest. No API keys.")

with st.sidebar:
    tickers = st.text_input("Tickers (comma-separated)", value="AAPL, TSLA, MSFT").upper()
    lookback = st.slider("Lookback (days)", 1, 30, 7)
    start = st.date_input("Backtest start", value=pd.Timestamp.utcnow().date() - pd.Timedelta(days=90))
    run = st.button("Fetch & Score")

if run:
    tlist, all_news = [t.strip() for t in tickers.split(',') if t.strip()], []
    with st.spinner("Fetching RSS headlines and scoring sentiment…"):
        for t in tlist:
            news = fetch_rss_headlines(t, lookback_days=lookback)
            if news.empty: continue
            scores = news['title'].apply(finbert_scores)
            news['label'], news['sentiment_score'] = scores.apply(label_from_scores), scores.apply(score_scalar)
            all_news.append(news)
    if not all_news:
        st.warning("No news found for the selected tickers / dates.")
        st.stop()

    df_news = pd.concat(all_news, ignore_index=True)
    st.subheader("Headlines")
    st.dataframe(df_news.sort_values('published', ascending=False)[['published','ticker','label','sentiment_score','title','link']], use_container_width=True)

    st.subheader("Sentiment Distribution")
    dist = df_news.groupby(['ticker','label']).size().reset_index(name='count')
    chart = alt.Chart(dist).mark_bar().encode(x='label:N', y='count:Q', color='label:N', column='ticker:N')
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Price vs Sentiment Backtest")
    for t in tlist:
        st.markdown(f"### {t}")
        sub, daily, px = df_news[df_news['ticker'] == t], daily_sentiment_aggregate(df_news[df_news['ticker'] == t]), load_prices(t, start=str(start))
        if px.empty or daily.empty:
            st.info("Insufficient data for backtest plot.")
            continue
        bt = backtest(px, daily)
        left, right = st.columns(2)
        with left: st.line_chart(px['Close'], height=260)
        with right: st.line_chart(bt['equity'], height=260)
    st.success("Done.")
else:
    st.info("Set tickers and click **Fetch & Score** to begin.")
