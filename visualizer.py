import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualizer():
    """
    Provides different functionalities to visualize available data
    """
    def __init__(self, datasets:list, symbols:list):
        self.datasets = datasets
        self.symbols = symbols

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
