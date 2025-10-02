from __future__ import annotations
import feedparser
import pandas as pd
import yfinance as yf

def yahoo_finance_rss_url(ticker: str) -> str:
    return f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker.upper()}&lang=en-US"

def fetch_rss_headlines(ticker: str, lookback_days: int = 7) -> pd.DataFrame:
    url, feed = yahoo_finance_rss_url(ticker), feedparser.parse(yahoo_finance_rss_url(ticker))
    rows, cutoff = [], pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
    for e in feed.entries:
        title, link = getattr(e, 'title', ''), getattr(e, 'link', '')
        published = getattr(e, 'published', None) or getattr(e, 'updated', None)
        try: ts = pd.to_datetime(published)
        except: ts = pd.Timestamp.utcnow()
        ts = ts.tz_localize('UTC') if ts.tzinfo is None else ts
        if ts < cutoff: continue
        rows.append({'ticker': ticker.upper(),'title': title,'link': link,'published': ts.tz_convert('UTC')})
    return pd.DataFrame(rows)

def load_prices(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    end = end or pd.Timestamp.utcnow().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    df = df.rename(columns=str.title)
    df.index = pd.to_datetime(df.index, utc=True)
    return df
