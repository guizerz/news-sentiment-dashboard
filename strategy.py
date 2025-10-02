from __future__ import annotations
import pandas as pd

SENTI_COL = 'sentiment_score'

def daily_sentiment_aggregate(df_news: pd.DataFrame) -> pd.DataFrame:
    if df_news.empty:
        return pd.DataFrame(columns=['date', SENTI_COL])
    g = (df_news.assign(date=pd.to_datetime(df_news['published']).dt.floor('D'))
         .groupby('date', as_index=False)['sentiment_score'].mean())
    return g.rename(columns={'sentiment_score': SENTI_COL})

def naive_signal(df_daily: pd.DataFrame, pos=0.15, neg=-0.15):
    s = pd.Series(0, index=df_daily.index)
    s[df_daily[SENTI_COL] > pos] = 1
    s[df_daily[SENTI_COL] < neg] = -1
    return s.rename('position')

def backtest(prices: pd.DataFrame, daily_senti: pd.DataFrame) -> pd.DataFrame:
    if prices.empty or daily_senti.empty:
        return pd.DataFrame(columns=['equity'])
    df = (prices[['Close']].rename(columns={'Close': 'close'})
          .join(daily_senti.set_index('date'), how='left').ffill())
    df['ret'] = df['close'].pct_change().fillna(0.0)
    df['position'] = naive_signal(df).shift(1).fillna(0.0)
    df['strategy_ret'] = df['position'] * df['ret']
    df['equity'] = (1 + df['strategy_ret']).cumprod()
    return df
