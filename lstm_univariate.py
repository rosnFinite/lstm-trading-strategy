from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class UnivariateLSTM():
    def __init__(self, data) -> None:
        self.original_data = data
        self.original_data_diff = None
        self.interpolated_data = None
        # instance of scaler
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.scaler_diff = MinMaxScaler(feature_range=(0,1))
        # Single feature from dataframe
        self.dataset = None
        self.dataset_diff = None
        self.scaled_dataset = None
        self.scaled_dataset_diff = None
        self.train = None
        self.train_diff = None
        self.test = None
        self.test_diff = None
        self.x_train = []
        self.x_train_diff = []
        self.y_train = []
        self.y_train_diff = []
        self.x_test = []
        self.x_test_diff = []
        # Model
        self.model = None

    def preprocess_data(self):
        print("========================Preprocessing Data========================")
        # Calculate number of missing dates in original dataset
        missing_dates = pd.date_range(start=str(self.original_data.index[0]), 
                        end=str(self.original_data.index[-1])).difference(self.original_data.index)
        print(f'Number of missing dates in original dataset -> {len(missing_dates)}')

        # Insert missing dates
        self.interpolated_data = self.original_data.resample("D").interpolate().reset_index()
        self.interpolated_data = self.interpolated_data.set_index("datetime")

        # Select one column to create a prediction on
        data = self.interpolated_data.filter(["close"])
        self.dataset = data.values
        # Scale selected data in range (0,1) -> MinMaxScaler
        self.scaled_dataset = self.scaler.fit_transform(self.dataset)

        # Eliminate Trend in data
        data = self.interpolated_data.diff().filter(["close"])
        self.dataset_diff = data.values[1:]
        self.scaled_dataset_diff = self.scaler_diff.fit_transform(self.dataset_diff)

        # Split data into train a test set
        split_index = int(len(self.dataset)*0.7)
        self.train = self.scaled_dataset[:split_index,:]
        self.test = self.scaled_dataset[split_index:,:]

        # Prepare train data
        # Last 60 datapoints are relevant for next 1 datapoint 
        for i in range(60, split_index):
            self.x_train.append(self.scaled_dataset[i-60:i, 0])
            self.y_train.append(self.scaled_dataset[i, 0])
        self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))
        print(f'Finished preparing training data -> X {self.x_train.shape}, Y {self.y_train.shape}')

        # Prepare test data
        for i in range(split_index+60, len(self.dataset)):
            self.x_test.append(self.scaled_dataset[i-60:i, 0])
        self.x_test = np.array(self.x_test)
        self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1],1))
        print(f'Finished preparing test data -> X {self.x_test.shape}')

        # Split trend eliminated data in train and test
        split_index = int(len(self.dataset_diff)*0.7)
        self.train_diff = self.scaled_dataset_diff[:split_index,:]
        self.test_diff = self.scaled_dataset_diff[split_index:,:]

        # Prepare train set on trend eliminated data
        for i in range(60, split_index):
            self.x_train_diff.append(self.scaled_dataset_diff[i-60:i,0])
            self.y_train_diff.append(self.scaled_dataset_diff[i,0])
        self.x_train_diff, self.y_train_diff = np.array(self.x_train_diff), np.array(self.y_train_diff)
        self.x_train_diff = np.reshape(self.x_train_diff, 
                            (self.x_train_diff.shape[0], self.x_train_diff.shape[1], 1))
        print('Finished preparing trend eliminated training data -> X '
              f'{self.x_train_diff.shape}, Y {self.y_train_diff.shape}')

        # Prepare test set on trend eliminated data
        for i in range(split_index+60, len(self.dataset_diff)):
            self.x_test_diff.append(self.scaled_dataset_diff[i-60:i,0])
        self.x_test_diff = np.array(self.x_test_diff)
        self.x_test_diff = np.reshape(self.x_test_diff, (self.x_test_diff.shape[0], self.x_test_diff.shape[1],1))
        print(f'Finished preparing trend eliminated test data -> X {self.x_test_diff.shape}')
        print("========================================================================")


    def build_and_fit_model(self, on_diff=False, num_epochs=100):
        print("=======================Building and Fitting Model=======================")
        self.model = Sequential()
        if on_diff:
            self.model.add(LSTM(units=50,
                                return_sequences=True,
                                input_shape=(self.x_train_diff.shape[1], 1)))
        else:
            self.model.add(LSTM(units=50,
                                return_sequences=True,
                                input_shape=(self.x_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer="adam", loss="mean_squared_error")
        if on_diff:
            self.model.fit(self.x_train_diff, self.y_train_diff,
                           epochs=num_epochs,
                           batch_size=32)
        else:
            self.model.fit(self.x_train, self.y_train,
                           epochs=num_epochs,
                           batch_size=32)

    def validate_model(self, on_diff=False):
        if self.model is None:
            print("!!!!! No model built to be validated")
            return

        test = self.x_test
        scaler = self.scaler
        dataset = self.dataset
        if on_diff:
            test = self.x_test_diff
            scaler = self.scaler_diff
            dataset = self.dataset_diff

        # Do predictions on test data
        predictions = self.model.predict(test)
        # scale back predictions
        predictions = scaler.inverse_transform(predictions)

        plt.plot(dataset[-len(predictions):], color="red")
        plt.plot(predictions, color="blue")
        plt.title("Groundtruth vs Prediction")
        plt.legend()
        plt.show()



df_ts = pd.read_csv("data/INTC_1day_prep.csv", index_col=0, parse_dates=True)
# df_ts.drop(columns=["Unnamed: 0"], inplace=True)
df_ts = df_ts.iloc[df_ts.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)
df_ts.set_index("datetime", inplace=True)
df_ts.index = pd.to_datetime(df_ts.index)
print(df_ts)

Model = UnivariateLSTM(data=df_ts)
Model.preprocess_data()
Model.build_and_fit_model(on_diff=True, num_epochs=100)
Model.validate_model(on_diff=True)
