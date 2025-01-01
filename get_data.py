import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import os


def get_stock_data(tickers, start, end):
    os.makedirs('data', exist_ok=True)
    data = {}

    for ticker in tickers:
        if not os.path.exists(f"data/{ticker}_{start}_{end}.csv"):
            df = yf.download(ticker, start=start, end=end)
            df.to_csv(f"data/{ticker}_{start}_{end}.csv")

        df = pd.read_csv(f"data/{ticker}_{start}_{end}.csv",
                         skiprows=2,
                         names=['Date', 'Open', 'High',
                                'Low', 'Close', 'Volume'],
                         parse_dates=['Date'],
                         date_format='%Y-%m-%d',
                         dtype={
            'Open': float,
            'High': float,
            'Low': float,
            'Close': float,
            'Volume': float
        })
        df.set_index('Date', inplace=True)
        data[ticker] = df

    return data


def plot(data, ticker):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"], name="Price"
    ))

    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(rangeslider=dict(visible=False))
    )

    fig.show()


if __name__ == "__main__":

    tickers = ["TSLA", "AAPL", "AMZN", "GOOG"]
    start = "2023-01-01"
    end = "2024-11-01"

    stock_data = get_stock_data(tickers=tickers, start=start, end=end)
    for ticker, data, in stock_data.items():
        plot(data, ticker)
