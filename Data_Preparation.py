"""
Provides a class responsible for preparing available timeseries data
for training the LSTM Network
"""
# pylint: disable=E1101
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import Config

class DataPreparation():
    """
    Prepares all available timeseries data in /data for training.
    This includes combining the datasets and transforming them into the needed format for training
    """

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0,1))
        if Config.TRAIN_SPLIT >= 1 and Config.TRAIN_SPLIT <= 0:
            raise ValueError("train split >= 1 or <= 0 are invalid !")
        self.train_split = Config.TRAIN_SPLIT
        self.stocks = []
        self.df_combined = pd.DataFrame()

        self.x_train = []
        self.y_train =  []
        self.x_test = []
        self.y_test = []


    def __collect_timeseries(self):
        for subdir, _, files in os.walk("data/"):
            for file in files:
                filepath = subdir + "/" + file

                if filepath.endswith("prep.csv"):
                    stock_symbol = filepath.split("/")[2].split("_")[0]
                    self.stocks.append(stock_symbol)
                    df_ts = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    df_ts = df_ts.iloc[df_ts.apply(pd.Series.first_valid_index).max():] \
                            .reset_index(drop=True)
                    self.df_combined[stock_symbol] = df_ts["close"]
                    self.df_combined[stock_symbol+"_ma"] = df_ts["MA_30"]

    def __scaling_combined_timeseries(self):
        self.df_combined[self.df_combined.columns] = \
            self.scaler.fit_transform(self.df_combined[self.df_combined.columns])

    def __collect_and_scale_data(self):
        self.__collect_timeseries()
        self.__scaling_combined_timeseries()

    def prepare_data(self):
        """
        Formats data into train and test data.
        For every single stock inside the combined timeseries it will create
        x_train => sliding windows of last 30 days
        y_train => the expected value of the next day and MA for next 30 days
        (same for x_test and y_test)
        and recombines them to one train and test dataset
        """

        print("Start preparing data for all available stocks in /data")
        self.__collect_and_scale_data()

        num_past = 30
        split_index = int(len(self.df_combined)*self.train_split)

        for stock in self.stocks:
            print(f'\tStart preparing data for {stock}')
            values = self.df_combined.filter([stock]).values
            ma_values = self.df_combined.filter([stock+"_ma"]).values

            x_train = []
            y_train = []
            for i in range(num_past, split_index-30):
                x_train.append(values[i-num_past:i, 0])
                y_train.append([values[i, 0], ma_values[i+30, 0]])
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            self.x_train.append(x_train)
            self.y_train.append(y_train)
            print(f"\t\tFinished preparing training data -> X {x_train.shape}, Y {y_train.shape}")

            x_test = []
            y_test = []
            for i in range(split_index + num_past, len(self.df_combined)-30):
                x_test.append(values[i-num_past:i, 0])
                y_test.append([values[i, 0], ma_values[i+30, 0]])
            x_test, y_test = np.array(x_test), np.array(y_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            self.x_test.append(x_test)
            self.y_test.append(y_test)
            print(f"\t\tFinished preparing test data -> X {x_test.shape}, Y {y_train.shape}")

        self.x_train = np.array(self.x_train)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[1]*self.x_train.shape[0], 30,1))
        self.y_train = np.array(self.y_train)
        self.y_train = np.reshape(self.y_train, (self.y_train.shape[1]*self.y_train.shape[0], 2))

        self.x_test = np.array(self.x_test)
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[1]*self.x_test.shape[0], 30, 1))
        self.y_test = np.array(self.y_test)
        self.y_test = np.reshape(self.y_test, (self.y_test.shape[1]*self.y_test.shape[0], 2))
