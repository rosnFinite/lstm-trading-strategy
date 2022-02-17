"""
Placeholder docstring
"""
import matplotlib.pyplot as plt 
import pandas as pd  
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model



# Moving Average crossover strategy
# Whenever MA100 crosses over MA200 -> buy
# Moving Average perform poorly on choppy data => e.g. intel    good on trends => e.g. nvidia

# MACD crossover strategy
# When MACD crosses over zero and price above MA200 -> buy

# RSI + MACD strategy (reverse strategy for shorting)
# MACD_s crosses MACD from below -> Buy signal
# RSI crosses 70 from above-> Sell signal for bought stocks


class TestBroker():
    """
    PlaceholderDosctring
    """
    def __init__(self, balance:int = 10000,invest_per_trade:float = 0.2,
                 data_path:str = None, model_path:str = None) -> None:
        # Trading balance to be used by the broker
        self.start_balance = balance
        self.balance = balance
        # Performance of account balance
        self.performance = []
        # Total amount of money that has been invested
        self.total_invested = 0
        # Amount of money currently invested in active positions
        self.currently_invested = 0
        # Currently active positions
        self.active_trades = dict()
        # Closed positions
        self.closed_trades = dict()
        # Stop-Loss [default -> 25%]
        self.stop_loss = 0.03
        self.take_profit = 0.50
        # Percentage of balance to invest per trade [default -> 10%]
        self.invest_per_trade = invest_per_trade
        # load timeseries data and drop entries with NaN
        self.timeseries = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.timeseries.drop(columns=["datetime","OHLC_avg","volume"], inplace=True)
        self.timeseries = self.timeseries \
                          .iloc[self.timeseries.apply(pd.Series.first_valid_index).max():] \
                          .reset_index(drop=True)
        print(self.timeseries)
        # Index denotes the current time
        self.index = 0


        if model_path is not None:
            # Need to handle possible exception
            self.scaler = MinMaxScaler(feature_range=(0,1))
            self.scaled_timeseries = self.scaler.fit_transform(self.timeseries)
            self.model = load_model(model_path)
            # Check model architecture
            self.model.summary()


    def run_simulation(self):
        """Moves through timeseries one step at a time"""
        for _ in range(len(self.timeseries) - self.index):
            # Sells stock if stop_loss was hit or updates stop_loss if not
            # stop_loss default on 25% loss
            #self.check_sl_tp()
            self.sell_all()

            # Check for new buy signal (MA crossover)
            #self.check_ma_buy_signal()

            # Check for new buy signal (MACD crossover)
            #self.check_macd_buy_signal()

            # Check for RSI + MACD signals (Long and Short)
            #self.check_rsi_signals()

            # Check for Buy Signals multivariate LSTM
            #technical hatte num_past=24
            # self.check_uni_lstm_signals()
            # self.check_multi_lstm_signals(num_past=24)
            self.check_uni_lstm_ma_day_signals()

            #self.check_persitent_model_signals()

            # Update total value of investments and available balance
            self.update_performance()
            self.index += 1

    def buy(self, current_index, position_type:str, verbose=True):
        """
        Buys stock at current_index. Amount of invested money is
        calculated via defined percentage of available balance.
        Trade will be saved in active_trades.

        Keyword arguments:
        current_index -- index of current row in timeseries
        position_type -- wheter it is a short or long position
        """
        if verbose:
            print(f'BUY at {current_index}')
        if self.balance == 0:
            return
        current_price = self.timeseries.iloc[current_index]["close"]
        investment = self.balance*self.invest_per_trade
        amount = investment / current_price
        if position_type == "long":
            stop_loss = current_price - current_price*self.stop_loss
            take_profit = current_price + current_price*self.take_profit
        else:
            stop_loss = current_price + current_price*self.stop_loss
            take_profit = current_price - current_price*self.take_profit
        self.active_trades[current_index] = {"type": position_type,
                                    "bought_for": current_price,
                                    "stop_loss": stop_loss,
                                    "take_profit": take_profit,
                                    "invested": investment,
                                    "amount": amount}
        self.balance -= investment
        self.total_invested += investment
        self.currently_invested += investment

    def sell(self, current_index, trade):
        """
        Sells owned stock bought at index denoted as trade for the price at
        current_index. Updates balance and deletes trade from active_trades,
        adding it to closed_trades.

        Keyword arguments:
        current_index -- index of current rown in timeseries
        trade -- key for the trade to delete in active_trade
        """
        current_price = self.timeseries.iloc[current_index]["close"]
        # Check type of position and calculate value of trade accordingly
        if self.active_trades[trade]["type"] == "long":
            value = self.active_trades[trade]["amount"] * current_price
        if self.active_trades[trade]["type"] == "short":
            value = (current_price - self.active_trades[trade]["bought_for"] + current_price) \
                    * self.active_trades[trade]["amount"]
        print(f'SOLD {self.active_trades[trade]["type"]} position {trade} at '
             f'{current_index}(Currently at {current_price:.2f}$) for W/L of '
             f'{value - self.active_trades[trade]["invested"]:.2f}$')
        self.balance += value
        # Delete trade from active trades
        self.closed_trades[trade] = {"type": self.active_trades[trade]["type"],
                        "bought_for": self.active_trades[trade]["bought_for"],
                        "amount": self.active_trades[trade]["amount"],
                        "sold_for": current_price,
                        "sold_at": current_index,
                        "invested": self.active_trades[trade]["invested"],
                        "return": value - self.active_trades[trade]["invested"],
                        "roi": value * 100 / self.active_trades[trade]["invested"]}
        del self.active_trades[trade]

    def sell_all(self):
        """Sells all currently active trades
        -- Meant to be used with LSTM Buy signals, to sell a day later (needs to be optimized)
        """
        current_price = self.timeseries.iloc[self.index]["close"]
        for trade in list(self.active_trades):
            if self.active_trades[trade]["type"] == "long":
                value = self.active_trades[trade]["amount"] * current_price
            if self.active_trades[trade]["type"] == "short":
                value = (current_price - self.active_trades[trade]["bought_for"] + current_price) \
                        * self.active_trades[trade]["amount"]
            self.balance += value
            print(f'SOLD {self.active_trades[trade]["type"]} position {trade} at '
                  f'{self.index}(Currently at {current_price:.2f}$) for W/L of '
                  f'{value - self.active_trades[trade]["invested"]:.2f}$  '
                  f'// BALANCE = {self.balance:.2f}')
            # Delete trade from active trades
            self.closed_trades[trade] = {"type": self.active_trades[trade]["type"],
                        "bought_for": self.active_trades[trade]["bought_for"],
                        "amount": self.active_trades[trade]["amount"],
                        "sold_for": current_price,
                        "sold_at": self.index,
                        "invested": self.active_trades[trade]["invested"],
                        "return": value - self.active_trades[trade]["invested"],
                        "roi": value * 100 / self.active_trades[trade]["invested"]}
            del self.active_trades[trade]

    def check_ma_buy_signal(self):
        """
        Checks for buy/sell signals via Simple Moving Average (SMA).
        If MA_100 crosses MA_200 from below => Open Long Position
        If MA_100 crosses MA_200 from above => Open Short Position
        """
        current_ma200 = self.timeseries.iloc[self.index]["MA_200"]
        current_ma100 = self.timeseries.iloc[self.index]["MA_100"]

        prev_ma200 = self.timeseries.iloc[self.index-1]["MA_200"]
        prev_ma100 = self.timeseries.iloc[self.index-1]["MA_100"]
        # Buy signal if MA_100 crosses MA_200
        if prev_ma100 < prev_ma200 and current_ma100 > current_ma200:
            self.buy(current_index=self.index, position_type="long")

    def check_macd_buy_signal(self):
        """
        Checks for buy signal via MACD and SMA.
        If MACD crosses 0 and price > MA_200 => Open Long Position
        """
        current_macd = self.timeseries.iloc[self.index]["MACD"]
        prev_macd = self.timeseries.iloc[self.index-1]["MACD"]
        current_ma200 = self.timeseries.iloc[self.index]["MA_200"]
        current_price = self.timeseries.iloc[self.index]["close"]

        # If macd crosses over 0 -> NACHSCHAUEN
        if (current_macd < 0 and prev_macd > 0) or (current_macd > 0 and prev_macd < 0):
            if current_price > current_ma200:
                self.buy(current_index=self.index, position_type="long")

    def check_rsi_signals(self):
        """
        Checks for buy/sell signals via MACD and determine optimal sellingpoint
        with relative strength index (RSI)
        If MACD_s crosses MACD from below => Open Long Position
        If MACD_s crosses MACD from above => Open Short Position

        If RSI drops below 70 => Sell Long Positions
        IF RSI raises above 30 => Sell Short Positions
        """
        current_macd = self.timeseries.iloc[self.index]["MACD"]
        current_macds = self.timeseries.iloc[self.index]["MACD_s"]
        prev_macd = self.timeseries.iloc[self.index-1]["MACD"]
        prev_macds = self.timeseries.iloc[self.index-1]["MACD_s"]

        # If macd_s crosses macd from below -> Long
        if prev_macds < prev_macd and current_macds > current_macd:
            self.buy(current_index=self.index, position_type="long")

        # If macd_s crosses macd from above -> Short

        if prev_macds > prev_macd and current_macds < current_macd:
            self.buy(current_index=self.index, position_type="short")

        # Check for sellingpoint with RSI
        current_rsi = self.timeseries.iloc[self.index]["RSI"]
        prev_rsi = self.timeseries.iloc[self.index-1]["RSI"]
        # If rsi drops below 70 -> Sell Long
        if prev_rsi > 70 and current_rsi < 70:
            for trade in list(self.active_trades):
                if self.active_trades[trade]["type"] == "long":
                    self.sell(current_index=self.index, trade=trade)

        # If rsi esceeds 30 -> Sell Short
        if prev_rsi < 30 and current_rsi > 30:
            for trade in list(self.active_trades):
                if self.active_trades[trade]["type"] == "short":
                    self.sell(current_index=self.index, trade=trade)

    def check_multi_lstm_signals(self, num_past):
        """
        Checks for buy/sell signals by comparing predictions of a multivariate LSTM
        with current stock price.
        If prediction > current price => Open Long Position
        If prediction < current price => Open Short Position

        ! Only one active trade allowed !
        """
        # Only predict if enough past values exist for prediction
        if self.index < num_past or len(self.active_trades)>0:
            return
        # Create Feature list of last num_past closing prices
        x_past = np.array([self.scaled_timeseries[self.index-num_past:self.index, \
                                                  0:self.scaled_timeseries.shape[1]]])

        prediction = self.model.predict(x_past)
        prediction_copies = np.repeat(prediction, self.timeseries.shape[1], axis=-1)
        y_pred = self.scaler.inverse_transform(prediction_copies)[:,0]

        current_price = self.timeseries.iloc[self.index]["close"]

        if y_pred > current_price + current_price*0.03:
            self.buy(current_index=self.index, position_type="long", verbose=False)
        elif y_pred < current_price - current_price*0.03:
            self.buy(current_index=self.index, position_type="short", verbose=False)

    def check_uni_lstm_signals(self, num_past=30):
        """
        Checks for buy/sell signals by comparing predictions of a univariate LSTM
        with current stock price.
        If prediction > current price => Open Long Position
        If prediction < current price => Open Short Position

        ! Only one active trade allowed !
        """
        if self.index < num_past or len(self.active_trades)>0:
            return

        # Create Feature list of last num_past closing prices
        x_past = np.array([self.scaled_timeseries[self.index-num_past:self.index, 3]])
        x_past = np.reshape(x_past, (x_past.shape[0], x_past.shape[1], 1))

        prediction = self.model.predict(x_past)
        prediction_copies = np.repeat(prediction, self.timeseries.shape[1], axis=-1)
        y_pred = self.scaler.inverse_transform(prediction_copies)[:,0]

        current_price = self.timeseries.iloc[self.index]["close"]

        if y_pred > current_price + current_price*0.03:
            self.buy(current_index=self.index, position_type="long", verbose=False)
        elif y_pred < current_price - current_price*0.03:
            self.buy(current_index=self.index, position_type="short", verbose=False)

    def check_uni_lstm_ma_day_signals(self, num_past=30, theta_threshold=1.01):
        """
        Uses the predicted next day price and predicted SMA for the next 30 days of
        an univariate LSTM to find good entry points for a long position.

        If ratio of predicted_ma/current_price bigger than a given threshold (theta_threshold)
        and prediction of next_day_price > current_price => Open Long Position

        If ratio < 1 and next_day_price < current_price => Close open positions

        ! Only one active trade allowed !
        """
        if self.index < num_past:
            return
        # Create Feature list of last num_past closing prices
        x_past = np.array([self.scaled_timeseries[self.index-num_past:self.index, 3]])
        x_past = np.reshape(x_past, (x_past.shape[0], x_past.shape[1], 1))

        prediction = self.model.predict(x_past)
        next_day = np.repeat([[prediction[0][0]]], self.timeseries.shape[1], axis=-1)
        predicted_ma = np.repeat([[prediction[0][1]]], self.timeseries.shape[1], axis=-1)
        next_day = self.scaler.inverse_transform(next_day)[:,0]
        predicted_ma = self.scaler.inverse_transform(predicted_ma)[:,0]

        current_price = self.timeseries.iloc[self.index]["close"]

        # Ratio between medium term prediction and current price
        # (theta > 1 = Price increse || theta < 1 = Price decrease)
        theta = predicted_ma / current_price

        if theta >= theta_threshold and next_day > current_price:
            self.buy(current_index=self.index, position_type="long", verbose=False)
        if theta < 1 and next_day < current_price:
            self.sell_all()


    def check_persitent_model_signals(self, lag=2):
        """
        Needs to be updated
        """
        if self.index < 1:
            return
        current_price =self.timeseries.iloc[self.index]["close"]
        prediction = self.timeseries.iloc[self.index-(lag-1)]["close"]

        if current_price < prediction:
            self.buy(current_index=self.index, position_type="long", verbose=False)
        else:
            self.buy(current_index=self.index, position_type="short", verbose=False)


    def check_sl_tp(self):
        """
        Checks if any active trade reached its defined stop/loss value and closes the trade.
        Otherwise if price increased, it will update stop/loss to a new value.
        """
        current_price = self.timeseries.iloc[self.index]["close"]
        for trade in list(self.active_trades):
            if self.active_trades[trade]["type"] == "long":
                # Update stop_loss / take_profit stays the same
                new_sl = current_price - current_price*self.stop_loss
                # If stop_loss was hit
                if current_price <= self.active_trades[trade]["stop_loss"]:
                    # Sell corresponding stock and update account balance
                    self.sell(current_index=self.index, trade=trade)
                # update stop_loss only if newly calculated stop_loss is higher than last
                elif new_sl > self.active_trades[trade]["stop_loss"]:
                    self.active_trades[trade]["stop_loss"] = new_sl
                # check if take_profit was hit
                elif current_price >= self.active_trades[trade]["take_profit"]:
                    self.sell(current_index=self.index, trade=trade)

                if self.model is not None and self.index+30 == trade:
                    self.sell(current_index=self.index, trade=trade)
            else:
                new_sl = current_price + current_price*self.stop_loss
                if current_price >= self.active_trades[trade]["stop_loss"]:
                    self.sell(current_index=self.index, trade=trade)
                elif new_sl < self.active_trades[trade]["stop_loss"]:
                    self.active_trades[trade]["stop_loss"] = new_sl
                elif current_price <= self.active_trades[trade]["take_profit"]:
                    self.sell(current_index=self.index, trade=trade)

                if self.model is not None and self.index+30 == trade:
                    self.sell(current_index=self.index, trade=trade)


    def update_performance(self):
        """
        Logs the change in yield of all active positions over time
        """
        active_value = 0
        current_price = self.timeseries.iloc[self.index]["close"]
        for trade in self.active_trades:
            if self.active_trades[trade]["type"] == "long":
                active_value += self.active_trades[trade]["amount"] * current_price
            else:
                active_value += \
                    (current_price - self.active_trades[trade]["bought_for"] + current_price) \
                    * self.active_trades[trade]["amount"]

        self.performance.append(self.balance + active_value)

    def plot_performance(self):
        """
        Plot comparison of stock performance to performance of trading strategy.
        """
        df_perf = pd.DataFrame(self.performance, columns=["value"])
        df_perf["return"] = (df_perf["value"]/df_perf.iloc[0]["value"] -1) * 100
        perf_trace = go.Scatter(
            x=df_perf.index,
            y=df_perf["return"],
            name="Account Performance",
        )
        self.timeseries["close"] = pd.to_numeric(self.timeseries["close"], errors="coerce")
        stock_trace = go.Scatter(
            x=self.timeseries.index,
            y=(self.timeseries["close"]/self.timeseries.iloc[0]["close"] - 1)*100,
            name="Stock Performance",
        )
        fig = make_subplots()
        fig.add_trace(perf_trace)
        fig.add_trace(stock_trace)
        # fig = px.line(df_perf, x=df_perf.index, y="value", title="Performance")
        fig.update_layout(
            title="Stock Performance vs Strategy Performance",
            xaxis_title="Trading Day Index",
            yaxis_title="Return in %"
        )
        fig.show()


    # Calculate return of investment
    def print_statistics(self):
        """
        Displays the underlying statistics of a performed backtest
        """
        current_price = self.timeseries.iloc[-1]["close"]
        # Close still active trades
        for trade in list(self.active_trades):
            value = self.active_trades[trade]["amount"] * current_price
            self.closed_trades[trade] ={"type": self.active_trades[trade]["type"],
                                        "bought_for": self.active_trades[trade]["bought_for"],
                                        "amount": self.active_trades[trade]["amount"],
                                        "sold_for": current_price,
                                        "sold_at": self.index,
                                        "invested": self.active_trades[trade]["invested"],
                                        "return": value,
                                        "roi": value * 100 / self.active_trades[trade]["invested"]}
            self.balance += value
            self.currently_invested -= self.active_trades[trade]["invested"]
            del self.active_trades[trade]

        num_profitable_trades = 0
        for trade in list(self.closed_trades):
            if self.closed_trades[trade]["return"] > 0:
                num_profitable_trades += 1

        try:
            num_trades = len(self.closed_trades)
            print("============Strategy Stats============")
            print(f'Last close = {current_price:.2f}$')
            print(f'Total invested = {self.total_invested:.2f}$')
            print(f'Number of total trades = {num_trades}')
            print(f'Number of profitable trades = {num_profitable_trades}')
            print(f'% of profitable trades = {num_profitable_trades/num_trades*100:3.2f}%')
            print('Average return per trade = '
                  f'{(self.balance - self.start_balance)/num_trades:.2f}$')
            print(f'Starting balance = {self.start_balance}$')
            print(f'Final balance = {self.balance:.2f}$')
            print(f'Profit = {self.balance - self.start_balance:.2f}$')
            # roc = (self.balance - 10000)*100 / self.total_invested
            roc = (self.balance / self.start_balance) * 100 - 100
            print(f'Strategy yield = {roc:.2f}%')
            stock_yield = (self.timeseries.iloc[-1]["close"]/self.timeseries.iloc[0]["close"] - 1) \
                          * 100
            print(f'Stock yield = {stock_yield:.2f}%')
            print("======================================")

            #Plot performance
            self.plot_performance()

        except ZeroDivisionError:
            print("No Trades in History")


    def plot_timeseries(self):
        """
        Plots the timeseries and corresponding simple moving average.
        (MA_30, MA_100, MA_200)
        """
        # Plot Moving Averages above OHLC_avg
        axis = self.timeseries["OHLC_avg"].plot(x="Index", y="OHLC_avg", kind="line")
        self.timeseries["MA_200"].plot(x="Index", y="OHLC_avg", kind="line", ax=axis)
        self.timeseries["MA_100"].plot(x="Index", y="OHLC_avg", kind="line", ax=axis)
        self.timeseries["MA_30"].plot(x="Index", y="OHLC_avg", kind="line", ax=axis)
        plt.show()

# Mehrere Aktien
# Add Risk/Reward
# Problem bei invest_per_trade=1 => divide by zero wenn nicht nach jedem Tag verkauft wird

# model_path="models/multivariate_lstm128x64_technical.h5"
# model_path="models/univariate_lstm50x50x50.h5"
broker = TestBroker(balance=1000,
                    invest_per_trade=0.1,
                    data_path="data/Energie/GE_1day_prep.csv",
                    model_path="models/NextDay+Ma/lstm50x50x50.h5")
broker.run_simulation()
broker.print_statistics()
# broker.plot_timeseries()
