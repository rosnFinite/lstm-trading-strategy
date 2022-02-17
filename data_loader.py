"""
Provides functionality to load new timeseries data from TwelveAPI and adds additional technical
indicators
"""
# pylint: disable=E0401
import io
import pathlib
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from api import API_KEY

class DataLoader():
    """
    Allows loading/saving of historical timeseries data from TwelveAPI for a given symbol
    and corresponding exchange. In addition, technical indicators (RSI, SMA and MACD) will
    be calculated and added to the timeseries.
    """
    def __init__(self) -> None:
        self.data = pd.DataFrame()

    def load_historical_data(self, symbol:str,
                             exchange:str="NYSE", folder:str=None, visualize:bool=False):
        """
        Requests timeseries data from the API, adds additional technical indicators
        (SMA,MACD,RSI) and saves it under /data/<symbol>_1day_prep.csv

        Keyword arguments:
        symbol -- Stock Symbol which identifies the company in the stock market
        exchange -- Stock-Exchange from which to collect historical data (default: NYSE)
        folder -- Allows to save data into specific directory inside
                  /data -> /data/<name>/<symbol>_1day_prep.csv
                  (default: /data/<symbol>_1day_prep.csv)
        visualize -- If True, will plot the timeseries and technical indicators
                     via plotly (default: False)
        """
        url = "https://twelve-data1.p.rapidapi.com/time_series"

        querystring = {"exchange": exchange,
                       "symbol":symbol,
                       "interval":"1day",
                       "outputsize":2520,
                       "format":"csv"}

        headers = {
            'x-rapidapi-host': "twelve-data1.p.rapidapi.com",
            'x-rapidapi-key': API_KEY
            }

        # Get Daily data from API as CSV
        response = requests.request("GET", url, headers=headers, params=querystring)
        self.data = pd.read_csv(io.StringIO(response.content.decode("utf-8")), sep=";")

        # Reverse data (first recorded datapoint = first datapoint in dataframe)
        self.data = self.data[::-1].reset_index()
        self.data.drop(columns=["index"], inplace= True)
        # Combining Open, Low, High and Close into a new column
        try:
            self.data["OHLC_avg"] = self.data[["open", "high", "low", "close"]].mean(axis=1)
        except KeyError as error:
            o_error = "Orignal error message: "+ str(error)
            sep = "-"*len(o_error)
            print(sep)
            print(o_error)
            print("KeyError: Couldn't load historical data for given "
                   f"symbol {symbol} from {exchange}")
            print("\tCheck if symbol existes and wheter it is listed on the given "
                    "exchange.")
            print("\tIf still no success, the API(twelveAPI) might not support the "
                    "requested symbol or exchange.")
            print(sep)
            return
        # Add additional technical indicators to timeseries
        self.__add_macd()
        self.__add_moving_average()
        self.__add_rsi()

        if visualize:
            self.__visualize_data()

        # Write data to csv file
        if folder is None:
            self.data.to_csv("data/" + symbol + "_1day_prep.csv")
        else:
            pathlib.Path("data/"+folder).mkdir(exist_ok=True)
            self.data.to_csv("data/" + folder + "/"+ symbol + "_1day_prep.csv")

    def __add_moving_average(self, window_sizes:list=None):
        if window_sizes is None:
            window_sizes = [30,100,200]
        for window in window_sizes:
            self.data["MA_"+str(window)] = self.data["OHLC_avg"].rolling(window=window).mean()

    def __add_macd(self):
        ema_26 = self.data["OHLC_avg"].ewm(span=26, adjust=False, min_periods=26).mean()
        ema_12 = self.data["OHLC_avg"].ewm(span=12, adjust=False, min_periods=12).mean()
        macd = ema_12 - ema_26
        macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        macd_h = macd - macd_s

        self.data["MACD"] = self.data.index.map(macd)
        self.data["MACD_h"] = self.data.index.map(macd_h)
        self.data["MACD_s"] = self.data.index.map(macd_s)


    def __add_rsi(self):
        close_delta = self.data["OHLC_avg"].diff()

        higher_close = close_delta.clip(lower=0)
        lower_close = -1 * close_delta.clip(upper=0)

        # Using exponential moving average
        ma_up = higher_close.ewm(com= 14-1, adjust=True, min_periods=14).mean()
        ma_down = lower_close.ewm(com=14-1, adjust=True, min_periods=14).mean()

        rsi = ma_up / ma_down
        rsi = 100 -(100/(1+rsi))
        self.data["RSI"] = self.data.index.map(rsi)

    def __visualize_data(self):
        # Plot MACD
        fig = make_subplots(rows=3, cols=1)
        # price Line
        fig.append_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['open'],
                line=dict(color='#ff9900', width=1),
                name='open',
                # showlegend=False,
                legendgroup='1',
            ), row=1, col=1
        )

        # Candlestick chart for pricing
        fig.append_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['open'],
                high=self.data['high'],
                low=self.data['low'],
                close=self.data['close'],
                increasing_line_color='#ff9900',
                decreasing_line_color='black',
                showlegend=False
            ), row=1, col=1
        )

        # Fast Signal (%k)
        fig.append_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['MACD'],
                line=dict(color='#ff9900', width=2),
                name='macd',
                # showlegend=False,
                legendgroup='2',
            ), row=2, col=1
        )

        # Slow signal (%d)
        fig.append_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['MACD_s'],
                line=dict(color='#000000', width=2),
                # showlegend=False,
                legendgroup='2',
                name='signal'
            ), row=2, col=1
        )
        # Colorize the histogram values
        colors = np.where(self.data['MACD_h'] < 0, '#000', '#ff9900')
        # Plot the histogram
        fig.append_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['MACD_h'],
                name='histogram',
                marker_color=colors,
            ), row=2, col=1
        )

        fig.append_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data["RSI"],
                line=dict(color='#295AB7', width=2),
                name="rsi"
            ), row=3, col=1
        )
        # Make it pretty
        layout = go.Layout(
            plot_bgcolor='#efefef',
            # Font Families
            font_family='Monospace',
            font_color='#000000',
            font_size=20,
            xaxis=dict(
                rangeslider=dict(
                    visible=False
                )
            )
        )
        # Update options and show plot
        fig.update_layout(layout)
        fig.show()
