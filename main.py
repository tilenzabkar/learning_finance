import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from get_data import get_stock_data

def create_sequences(data, seq_len):
    """Returns X and y where X is a sequence of stock prices and y is the next stock price.
    X is the data we train on and y is the desired prediction."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

class LinearRegressionGD:
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            # predictions
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
        
def main():
    tickers = ["QQQ", "GOOG"]
    start_date = '2021-01-01'
    end_date = '2024-01-01'
    stock_data = get_stock_data(tickers, start_date, end_date)
        
    for ticker, df in stock_data.items():
        df = df.copy()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()
        
        log_returns = df['log_returns'].values 
        X, y = create_sequences(log_returns, seq_len=10)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # reshape data so that it is in the right format for the model (2d format)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        scaler = StandardScaler() # we need to scale the data for the model to perform well, mean=0, std=1
        X_train = scaler.fit_transform(X_train) # fit and transform the training set
        X_test = scaler.transform(X_test) # transform only using the mean and std from the training set
              
        # run the model
        model = LinearRegressionGD(learning_rate=0.01, epochs=1000)
        model.fit(X_train, y_train)

        # make predictions
        y_pred = model.predict(X_test)

        # plot results
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=y_test,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            y=y_pred,
            mode='lines',
            name='Predicted',
            line=dict(color='orange', width=2)
        ))

        fig.update_layout(
            title=f"Linear Regression Predictions vs Actual for {ticker}",
            xaxis_title="Index",
            yaxis_title="Log Returns",
            legend=dict(x=0, y=1),
            template="plotly_dark"
        )

        fig.show()

if __name__ == "__main__":
    main()