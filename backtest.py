import os
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from data_preparation import DataPreparation
from plotly.subplots import make_subplots

from visualizer import Visualizer


class Backtester():
    """
    Class provides functionality to test different trading strategies against historical data
    """
    model = load_model("models/NextDay+MA/lstm50x50x50.h5")

    def __init__(self, start_index:int=None):
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
        if start_index is not None:
            self.start_index = start_index
            self.current_index = start_index
        else:
            self.start_index = 0
            self.current_index = 0


    def _scaling(self):
        """
        Partially fits and scales available datasets
        """
        _ = [self.scaler.partial_fit(df) for df in self.datasets]
        self.scaled_datasets = [pd.DataFrame(data=self.scaler.transform(df), 
                                             columns=["close","MA_30","MA_100","MA_200","MACD","MACD_h","MACD_s","RSI"]) \
                                             for df in self.datasets]

    def start_simulation(self, step_time:int=None):
        """
        Starts the backtesting process
        """
        def run_simulation(strategy):
            """

            """
            for _ in range(len(self.datasets[0])-self.start_index):
                strategy()

                self.update_perf()
                self.current_index += 1
                if step_time is not None:
                    time.sleep(step_time)

        run_simulation(self.lstm_strategy)

    def lstm_strategy(self, num_past=30, theta_threshold = 1.02):
        """
        Uses the predicted next day price and predicted SMA for the next 30 days of
        an univariate LSTM to find good entry points for a long position.

        If ratio of predicted_ma/current_price bigger than a given threshold (theta_threshold)
        and prediction of next_day_price > current_price => Open Long Position

        If ratio < 1 and next_day_price < current_price => Close open positions

        ! Only one active trade allowed !
        """
        if self.current_index < num_past:
            return

        # Check every stock for buy signals
        for index, symbol in enumerate(self.symbols):
            x_past = np.array([self.scaled_datasets[index].values[self.current_index-num_past:self.current_index, 3]])
            x_past = np.reshape(x_past, (x_past.shape[0], x_past.shape[1], 1))

            pred = self.model.predict(x_past)
            next_day = np.repeat([[pred[0][0]]], self.datasets[0].shape[1], axis=-1)
            pred_ma = np.repeat([[pred[0][1]]], self.datasets[0].shape[1], axis=-1)
            next_day = self.scaler.inverse_transform(next_day)[:,0]
            pred_ma = self.scaler.inverse_transform(pred_ma)[:,0]

            current_price = self.datasets[index].iloc[self.current_index]["close"]
            theta = pred_ma / current_price

            if symbol not in self.open_trades.keys():
                if theta >= theta_threshold and next_day > current_price:
                    self.buy(position_type="long", position_size=self.balance*0.1, symbol=symbol)
            else:
                if theta < 1 and next_day < current_price:
                    self.sell(symbol=symbol)


    def buy(self, position_type:str, position_size:int, symbol:str):
        """
        Buys stock at the current index.
        Currently only one open trade per stock allowed.

        Keyword arguments:
        position_type -- Wheter it is a short or long position ("short" / "long")
        position_size -- Amount of money to be used for buying a selected stock
        symbol -- identifier of the stock to invest in
        """
        if position_type not in ["short", "long"]:
            print("Invalid position_type (method: buy)")
            return
        if symbol not in self.symbols:
            print("Invalid symbol (method: buy)")
            return
        if self.balance - position_size < 0:
            print("Not enought balance available for trade (method: buy)")
            return

        # Get data for given symbol
        symbol_index = self.symbols.index(symbol)
        symbol_timeseries = self.datasets[symbol_index]

        current_price = symbol_timeseries.iloc[self.current_index]["close"]
        shares = position_size/current_price
        print(f'{self.current_index}: Bought {shares} shares of {symbol} at {current_price} (total: {position_size}) ')
        self.open_trades[symbol] = {"type": position_type,
                                    "bought_for": current_price,
                                    "invested": position_size,
                                    "amount": shares}

        self.balance -= position_size
        self.total_invested += position_size
        self.currently_invested += position_size

    def sell(self, symbol:str):
        """
        Sells stock at the current index.

        Keyword arguments:
        symbol -- identifier of the stock to sell
        """
        if symbol not in self.symbols:
            print("Invalid symbol (method: sell)")
            return

        symbol_index = self.symbols.index(symbol)
        symbol_timeseries = self.datasets[symbol_index]

        current_price = symbol_timeseries.iloc[self.current_index]["close"]

        value = self.open_trades[symbol]["amount"] * current_price
        print(f'{self.current_index}: SOLD {self.open_trades[symbol]["type"]} position for {symbol} at '
             f'{self.current_index}(Currently at {current_price:.2f}$) for W/L of '
             f'{value - self.open_trades[symbol]["invested"]:.2f}$')
        self.closed_trades[len(self.closed_trades)] = {"type": self.open_trades[symbol]["type"],
                        "bought_for": self.open_trades[symbol]["bought_for"],
                        "amount": self.open_trades[symbol]["amount"],
                        "sold_for": current_price,
                        "sold_at": self.current_index,
                        "invested": self.open_trades[symbol]["invested"],
                        "return": value - self.open_trades[symbol]["invested"],
                        "roi": value * 100 / self.open_trades[symbol]["invested"]}
        self.balance += value
        self.currently_invested -= self.open_trades[symbol]["invested"]
        del self.open_trades[symbol]

    def update_perf(self):
        """
        Logs the balance value over time
        """
        active_value = 0
        for symbol in self.open_trades:
            symbol_index = self.symbols.index(symbol)
            symbol_timeseries = self.datasets[symbol_index]
            current_price = symbol_timeseries.iloc[self.current_index]["close"]
            active_value += self.open_trades[symbol]["amount"] * current_price
        print("==============================================================")
        print(f"Index = {self.current_index}")
        print(f"Current Account Value = {active_value+ self.balance:.2f}")
        print(f"Available Balance = {self.balance}")
        print(f"Invested = {active_value}")
        print("==============================================================")
        self.performance.append(self.balance + active_value)

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

    def visualize_all(self, num_datapoints:int= 1000, sector:str=None):
        """
        Visualizes all available stocks in one plot

        Keyword arguments:
        num_datapoints -- number of past datapoints to plot (default: 1000)
        sector -- select sector (directories in data/) and only plot related stocks (default:None)
        """
        sector_stocks = []
        for subdir, _, files in os.walk("data/"+sector):
            for file in files:
                filepath = subdir + "/" + file
                if filepath.endswith("prep.csv"):
                    stock_symbol = filepath.split("/")[2].split("_")[0]
                    sector_stocks.append(stock_symbol)

        fig = make_subplots()
        if num_datapoints >= len(self.datasets[0]):
            num_datapoints = len(self.datasets[0])
        for index, symbol in enumerate(self.symbols):
            if sector is not None and symbol not in sector_stocks:
                continue
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
    
    def _close_all_trades(self):
        """
        Closes all currently open trades
        """
        for symbol in list(self.open_trades):
            symbol_index = self.symbols.index(symbol)
            symbol_timeseries = self.datasets[symbol_index]
            current_price = symbol_timeseries.iloc[-1]["close"]

            value = self.open_trades[symbol]["amount"] * current_price
            self.closed_trades[len(self.closed_trades)] = {"type": self.open_trades[symbol]["type"],
                        "bought_for": self.open_trades[symbol]["bought_for"],
                        "amount": self.open_trades[symbol]["amount"],
                        "sold_for": current_price,
                        "sold_at": self.current_index,
                        "invested": self.open_trades[symbol]["invested"],
                        "return": value - self.open_trades[symbol]["invested"],
                        "roi": value * 100 / self.open_trades[symbol]["invested"]}
            self.balance += value
            self.currently_invested -= self.open_trades[symbol]["invested"]
            del self.open_trades[symbol]

    def print_stats(self):
        """
        Displays statistics for tested strategy
        """
        self._close_all_trades()
        num_profitable = 0
        for trade in self.closed_trades:
            if self.closed_trades[trade]["return"] > 0:
                num_profitable += 1
        
        try:
            num_trades = len(self.closed_trades)
            print("============Strategy Stats============")
            print(f'Total invested = {self.total_invested:.2f}$')
            print(f'Number of total trades = {num_trades}')
            print(f'Number of profitable trades = {num_profitable}')
            print(f'% of profitable trades = {num_profitable/num_trades*100:3.2f}%')
            print('Average return per trade = '
                  f'{(self.balance - self.start_balance)/num_trades:.2f}$')
            print(f'Starting balance = {self.start_balance}$')
            print(f'Final balance = {self.balance:.2f}$')
            print(f'Profit = {self.balance - self.start_balance:.2f}$')
            # roc = (self.balance - 10000)*100 / self.total_invested
            roc = (self.balance / self.start_balance) * 100 - 100
            print(f'Strategy yield = {roc:.2f}%')
            # stock_yield = (self.timeseries.iloc[-1]["close"]/ self.timeseries.iloc[0]["close"] - 1) \
            #               * 100
            # print(f'Stock yield = {stock_yield:.2f}%')
            print("======================================")

            #Plot performance

        except ZeroDivisionError:
            print("No Trades in History")
        







t = Backtester(start_index=2200)
t.start_simulation()
t.print_stats()
