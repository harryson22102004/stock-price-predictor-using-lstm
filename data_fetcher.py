import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(
    ticker: str,
    period: str = "5y",
    interval: str = "1d"
) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    df.index = pd.to_datetime(df.index)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df
