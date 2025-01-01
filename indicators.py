import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from get_data import get_stock_data


def sma(data, short_window=20, long_window=50):
    """Helper function for strategy(), to implements simple moving average strategy."""

    data["Short_MA"] = data["Close"].rolling(window=short_window).mean()
    data["Long_MA"] = data["Close"].rolling(window=long_window).mean()

    return data


def ema(data, short_window=20, long_window=50):
    """Helper function for strategy(), to implement exponential moving average strategy.
    This strategy gives more weight to recent prices.
    Note: span = (2 * window - 1) for comparable results with SMA"""

    short_span = 2 * short_window - 1
    long_span = 2 * long_window - 1

    data["Short_MA"] = data["Close"].ewm(
        span=short_span, adjust=False).mean()
    data["Long_MA"] = data["Close"].ewm(
        span=long_span, adjust=False).mean()

    return data


def moving_avg_strategy(data, short_window=20, long_window=50, choice="sma"):
    """Implements either simple moving average strategy or exponential moving average strategy."""
    if choice == "sma":
        data = sma(data, short_window, long_window)
    elif choice == "ema":
        data = ema(data, short_window, long_window)
    else:
        raise ValueError(
            f"Strategy '{choice}' not supported. Use 'sma' or 'ema'.")

    data = data.dropna()
    print(data.tail())
    data["Signal"] = np.where(
        np.array(data["Short_MA"] > data["Long_MA"]) & np.array(
            data["Short_MA"].shift(1) <= data["Long_MA"].shift(1)), 1,
        np.where(
            np.array(data["Short_MA"] < data["Long_MA"]) & np.array(
                data["Short_MA"].shift(1) >= data["Long_MA"].shift(1)), -1, 0)
    )

    return data


def calculate_indicators(data):

    delta = data["Close"].diff()
    avg_gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    avg_loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    rs = avg_gain / avg_loss  # relative strength
    # rsi measures price velocity and magnitude
    # 0 < rsi < 100
    # rsi > 70 => potentially overbought
    # rsi < 30 => potentially oversold
    rsi = 100 - (100 / (1 + rs))
    data["RSI"] = rsi

    data["MA_20"] = data["Close"].rolling(window=20).mean()

    data["Upper_Band"] = data["MA_20"] + 2 * \
        data["Close"].rolling(window=20).std()  # upper Bollinger band

    data["Lower_Band"] = data["MA_20"] - 2 * \
        data["Close"].rolling(window=20).std()  # lower Bollinger band

    # Bollinger bands help identify overbought and oversold assets

    data = data.dropna()
    return data


def plot_with_strategy(data, ticker):

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.7, 0.3],
        subplot_titles=("Candlestick Chart", "RSI")
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'], name='Price'
    ), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Short_MA'], mode='lines', name='Short MA'), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Long_MA'], mode='lines', name='Long MA'), row=1, col=1)

    # Strategy
    buy_signals = data[data['Signal'] == 1]
    sell_signals = data[data['Signal'] == -1]

    fig.add_trace(go.Scatter(
        x=buy_signals.index, y=buy_signals['Close'],
        mode='markers', marker=dict(color='green', size=10), name='Buy'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sell_signals.index, y=sell_signals['Close'],
        mode='markers', marker=dict(color='red', size=10), name='Sell'
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=data.index, y=data['RSI'], mode='lines', name='RSI'), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=data.index, y=[70] * len(data), mode='lines',
        line=dict(dash='dash', color='red'), name='Overbought'
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=data.index, y=[30] * len(data), mode='lines',
        line=dict(dash='dash', color='green'), name='Oversold'
    ), row=2, col=1)

    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        xaxis=dict(rangeslider=dict(visible=False)),
        xaxis2_title="Date",
        yaxis1_title="Price",
        yaxis2_title="RSI",
        height=800,
        showlegend=True
    )

    fig.show()


if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2024-01-01'
    end_date = '2024-12-31'

    stock_data = get_stock_data(tickers, start=start_date, end=end_date)

    for ticker, data in stock_data.items():
        data = moving_avg_strategy(data)
        data = calculate_indicators(data)
        plot_with_strategy(data, ticker)
