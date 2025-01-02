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

    data = data.copy()

    if choice == "sma":
        data = sma(data, short_window, long_window)
    elif choice == "ema":
        data = ema(data, short_window, long_window)
    else:
        raise ValueError(
            f"Strategy '{choice}' not supported. Use 'sma' or 'ema'.")

    data = data.dropna()
    data["Signal_MA"] = np.where(
        np.array(data["Short_MA"] > data["Long_MA"]) & np.array(
            data["Short_MA"].shift(1) <= data["Long_MA"].shift(1)), 1,
        np.where(
            np.array(data["Short_MA"] < data["Long_MA"]) & np.array(
                data["Short_MA"].shift(1) >= data["Long_MA"].shift(1)), -1, 0)
    )

    return data


def rsi_strategy(data, overbought=70, oversold=30):
    """Implements RSI-based trading strategy."""

    data = data.copy()

    if 'RSI' not in data.columns:
        data = calculate_indicators(data)

    data["Signal_RSI"] = np.where(
        np.array(data["RSI"] > overbought) & np.array(
            data["RSI"].shift(1) <= overbought), -1,
        np.where(
            np.array(data["RSI"] < oversold) & np.array(data["RSI"].shift(1) >= oversold), 1, 0)
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

    # ma strategy
    buy_ma_signals = data[data['Signal_MA'] == 1]
    sell_ma_signals = data[data['Signal_MA'] == -1]

    # rsi strategy
    buy_rsi_signals = data[data['Signal_RSI'] == 1]
    sell_rsi_signals = data[data['Signal_RSI'] == -1]

    fig.add_trace(go.Scatter(
        x=buy_ma_signals.index, y=buy_ma_signals['Close'],
        mode='markers', marker=dict(color='green', size=10), name='Buy MA'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sell_ma_signals.index, y=sell_ma_signals['Close'],
        mode='markers', marker=dict(color='red', size=10), name='Sell MA'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=buy_rsi_signals.index, y=buy_rsi_signals['Close'],
        mode='markers', marker=dict(color='green', size=10), name='Buy RSI'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sell_rsi_signals.index, y=sell_rsi_signals['Close'],
        mode='markers', marker=dict(color='red', size=10), name='Sell RSI'
    ), row=1, col=1)

    # calculate y-axis ranges for normalization for plotting lines between rsi and purchase/sell points
    price_range = data['Close'].max() - data['Close'].min()
    price_min = data['Close'].min()

    for idx in buy_rsi_signals.index:
        # Convert to relative coordinates (0-1)
        y0 = (data.loc[idx, 'RSI'] / 100) * 0.3  # RSI in bottom 30% of plot
        y1 = ((data.loc[idx, 'Close'] - price_min) /
              price_range) * 0.7 + 0.3  # price in top 70%

        fig.add_shape(
            type="line",
            x0=idx,
            x1=idx,
            y0=y0,
            y1=y1,
            line=dict(color="green", width=1, dash="dash"),
            xref="x",
            yref="paper"
        )

    for idx in sell_rsi_signals.index:
        y0 = (data.loc[idx, 'RSI'] / 100) * 0.3
        y1 = ((data.loc[idx, 'Close'] - price_min) / price_range) * 0.7 + 0.3

        fig.add_shape(
            type="line",
            x0=idx,
            x1=idx,
            y0=y0,
            y1=y1,
            line=dict(color="red", width=1, dash="dash"),
            xref="x",
            yref="paper"
        )

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


class Portfolio:
    def __init__(self, initial_cash=100000):
        self.cash = initial_cash
        self.positions = {}  # {ticker: quantity}
        self.trades = []
        self.value_history = []

    def execute_trade(self, ticker, price, signal, date):
        if signal == 0:
            return

        quantity = int(self.cash * 0.05 / price)  # Use 10% of cash per trade
        cost = quantity * price
        commission = max(1.0, cost * 0.001)  # $1 min or 0.1%

        if signal == 1 and self.cash >= (cost + commission):  # Buy
            self.cash -= (cost + commission)
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
            self.trades.append({
                'date': date,
                'ticker': ticker,
                'action': 'BUY',
                'quantity': quantity,
                'price': price,
                'commission': commission
            })

        elif signal == -1 and self.positions.get(ticker, 0) > 0:  # Sell
            quantity = self.positions[ticker]
            self.cash += (cost - commission)
            self.positions[ticker] = 0
            self.trades.append({
                'date': date,
                'ticker': ticker,
                'action': 'SELL',
                'quantity': quantity,
                'price': price,
                'commission': commission
            })

    def update_value(self, data, date):
        value = self.cash
        for ticker, quantity in self.positions.items():
            value += quantity * data.loc[date, 'Close']
        self.value_history.append({'date': date, 'value': value})

    def get_value_history(self):
        return self.value_history


if __name__ == "__main__":
    tickers = ['TSLA']
    start_date = '2023-8-01'
    end_date = '2024-12-31'

    stock_data = get_stock_data(tickers, start=start_date, end=end_date)

    portfolio = Portfolio()

    for ticker, data in stock_data.items():
        data = moving_avg_strategy(data)
        data = calculate_indicators(data)
        data = rsi_strategy(data)

        for date, row in data.iterrows():
            portfolio.execute_trade(
                ticker, row['Close'], row['Signal_MA'], date)
            portfolio.execute_trade(
                ticker, row['Close'], row['Signal_RSI'], date)
            portfolio.update_value(data, date)

        plot_with_strategy(data, ticker)

    # performance metrics
    initial_value = 100000
    final_value = portfolio.value_history[-1]['value']
    total_return = final_value - initial_value
    percent_return = (total_return / initial_value) * 100

    print(f"\nPortfolio Performance Summary:")
    print(f"Initial Value: ${initial_value:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Total Return: ${total_return:,.2f}")
    print(f"Percent Return: {percent_return:.2f}%")
    print(f"\nFinal Positions:")
    for ticker, quantity in portfolio.positions.items():
        print(f"{ticker}: {quantity} shares")
