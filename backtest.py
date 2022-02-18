import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from data_preparation import DataPreparation
from plotly.subplots import make_subplots


class Backtester():
    """
    Class provides functionality to test different trading strategies against historical data
    """
    model = load_model("models/NextDay+MA/lstm50x50x50.h5")

    def __init__(self):
        (self.datasets, self.symbols) = DataPreparation().dfs   # Contains list of stock dataframes
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaled_datasets = []
        self._scaling()

        self.start_balance = 10000
        self.balance = self.start_balance
        self.performance = []
        self.total_invested = 0
        self.currently_invested = 0
        self.open_trades = dict()
        self.closed_trades = dict()
        self.current_index = 0


    def _scaling(self):
        """
        Partially fits and scales available datasets
        """
        _ = [self.scaler.partial_fit(df) for df in self.datasets]
        self.scaled_datasets = [pd.DataFrame(data=self.scaler.transform(df), 
                                             columns=["close","MA_30","MA_100","MA_200","MACD","MACD_h","MACD_s","RSI"]) \
                                             for df in self.datasets]

    def buy(self, position_size:int, symbol:str):
        """
        Buy Stock at the current index.

        Keyword arguments:
        position_size -- Amount of money to be used for buying a selected stock
        symbol -- identifier of the stock to invest in
        """

    def visualize_stock(self, symbol:str):
        """
        Visualizes historical data for given symbol

        Keyword arguments:
        symbol --  stock identifier
        """
        symbol_index = self.symbols.index(symbol) # Index of corresponding data in datasets
        symbol_timeseries = self.datasets[symbol_index]
        trace1 = go.Scatter(
            x = symbol_timeseries.index,
            y = symbol_timeseries["close"],
            name = "Close"
        )
        trace2 = go.Scatter(
            x = symbol_timeseries.index,
            y = symbol_timeseries["MA_30"],
            name = "MA 30"
        )
        trace3 = go.Scatter(
            x = symbol_timeseries.index,
            y = symbol_timeseries["MA_100"],
            name = "MA 100"
        )
        trace4 = go.Scatter(
            x = symbol_timeseries.index,
            y = symbol_timeseries["MA_200"],
            name = "MA 200"
        )
        fig = make_subplots()
        fig.add_trace(trace1)
        fig.add_trace(trace2)
        fig.add_trace(trace3)
        fig.add_trace(trace4)
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(
            title= f"Historical Data of {symbol}",
            xaxis_title="Trading Day Index",
            yaxis_title="Closing Price in $"
        )
        fig.show()

    def visualize_all(self, num_datapoints:int= 1000):
        """
        Visualizes all available stocks in one plot

        Keyword arguments:
        num_datapoints -- number of past datapoints to plot (default: 1000, max: len(dataset))
        """
        fig = make_subplots()
        for index, symbol in enumerate(self.symbols):
            symbol_timeseries = self.datasets[index]
            first = symbol_timeseries.iloc[0]["close"]
            last = symbol_timeseries.iloc[-1]["close"]
            print(f'{symbol} => {first}, {last}')
            trace = go.Scatter(
                x = symbol_timeseries.index[-num_datapoints:],
                y = (symbol_timeseries[-num_datapoints:]["close"]/\
                     symbol_timeseries.iloc[-num_datapoints]["close"]-1)*100,
                name = symbol
            )
            fig.add_trace(trace)
        fig.update_xaxes(rangeslider_visible=True)
        fig.update_layout(
            title = "Comparison of every available stock",
            xaxis_title = "Trading Day Index",
            yaxis_title = "Return in % (since 1st datapoint)",
        )
        fig.show()





t = Backtester()
t.visualize_all(num_datapoints=300)
