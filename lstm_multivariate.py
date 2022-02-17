import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

df_intel = pd.read_csv("data/INTC_1day_prep.csv", index_col=0, parse_dates=True)
df_intel = df_intel.iloc[df_intel.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)
df_nvidia = pd.read_csv("data/NVDA_1day_prep.csv", index_col=0, parse_dates=True)
df_nvidia = df_nvidia.iloc[df_nvidia.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)
df_amazon = pd.read_csv("data/AMZN_1day_prep.csv", index_col=0, parse_dates=True)
df_amazon = df_amazon.iloc[df_amazon.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)
df_microsoft = pd.read_csv("data/MSFT_1day_prep.csv", index_col=0, parse_dates=True)
df_microsoft = df_microsoft.iloc[df_microsoft.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)
df_apple = pd.read_csv("data/AAPL_1day_prep.csv", index_col=0, parse_dates=True)
df_apple = df_apple.iloc[df_apple.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)
df_google = pd.read_csv("data/GOOGL_1day_prep.csv", index_col=0, parse_dates=True)
df_google = df_google.iloc[df_google.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)
df_qcom = pd.read_csv("data/QCOM_1day_prep.csv", index_col=0, parse_dates=True)
df_qcom = df_qcom.iloc[df_qcom.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)


df_combined_close = pd.DataFrame()
df_combined_close["intc"] = df_intel["close"]
df_combined_close["open"] = df_intel["open"]
df_combined_close["high"] = df_intel["high"]
df_combined_close["low"] = df_intel["low"]
df_combined_close["MA_30"] = df_intel["MA_30"]
df_combined_close["MA_100"] = df_intel["MA_100"]
df_combined_close["MA_200"] = df_intel["MA_200"]
df_combined_close["RSI"] = df_intel["RSI"]
df_combined_close["MACD"] = df_intel["MACD"]
df_combined_close["MACD_h"] = df_intel["MACD_h"]
df_combined_close["MACD_s"] = df_intel["MACD_s"]
#df_combined_close["nvda"] = df_nvidia["close"]
#df_combined_close["amzn"] = df_amazon["close"]
#df_combined_close["msft"] = df_microsoft["close"]
#df_combined_close["aapl"] = df_apple["close"]
#df_combined_close["googl"] = df_google["close"]
#df_combined_close["qcom"] = df_qcom["close"]
print(df_combined_close)

"""
intc_scaler = MinMaxScaler(feature_range=(0,1))
nvda_scaler = MinMaxScaler(feature_range=(0,1))
amzn_scaler = MinMaxScaler(feature_range=(0,1))
msft_scaler = MinMaxScaler(feature_range=(0,1))
aapl_scaler = MinMaxScaler(feature_range=(0,1))
googl_scaler = MinMaxScaler(feature_range=(0,1))
qcom_scaler = MinMaxScaler(feature_range=(0,1))

df_combined_close["intc"] = intc_scaler.fit_transform(df_combined_close["intc"].to_numpy().reshape(-1,1))
df_combined_close["nvda"] = nvda_scaler.fit_transform(df_combined_close["nvda"].to_numpy().reshape(-1,1))
df_combined_close["amzn"] = amzn_scaler.fit_transform(df_combined_close["amzn"].to_numpy().reshape(-1,1))
df_combined_close["msft"] = msft_scaler.fit_transform(df_combined_close["msft"].to_numpy().reshape(-1,1))
df_combined_close["aapl"] = aapl_scaler.fit_transform(df_combined_close["aapl"].to_numpy().reshape(-1,1))
df_combined_close["googl"] = googl_scaler.fit_transform(df_combined_close["googl"].to_numpy().reshape(-1,1))
df_combined_close["qcom"] = qcom_scaler.fit_transform(df_combined_close["qcom"].to_numpy().reshape(-1,1))
"""
print(df_combined_close.corr())

df_combined_close.plot()
plt.show()
#df_combined_close[["intc","nvda","amzn","msft","aapl","googl","qcom"]] = scaler.fit_transform(df_combined_close[["intc","nvda","amzn","msft","aapl","googl","qcom"]])

scaler = MinMaxScaler(feature_range=(0,1))
df_combined_close = scaler.fit_transform(df_combined_close)

x = []
y = []

n_future = 1 # Number of days to predict
n_past = 28 # Number of past days for prediction
""" Funktioniert
for i in range(28, len(df_combined_close)-1):
    x.append(df_combined_close[i - n_past:i, 0:df_combined_close.shape[1]])
    y.append(df_combined_close[i + n_future - 1:i + n_future, 0])
"""
for i in range(28, len(df_combined_close)-31):
    x.append(df_combined_close[i-n_past:i, 0:df_combined_close.shape[1]])
    y.append(df_combined_close[i+29:i+30, 0])
x, y = np.array(x), np.array(y)

train_x = x[:int(len(df_combined_close)*0.9)]
train_y = y[:int(len(df_combined_close)*0.9)]

test_x = x[int(len(df_combined_close)*0.9):]
test_y = y[int(len(df_combined_close)*0.9):]

print(f'train_x shape == {train_x.shape}')
print(f'train_y shape == {train_y.shape}')
print("=================================")
print(f'test_x shape == {test_x.shape}')
print(f'test_y shape == {test_y.shape}')


model = Sequential()
model.add(LSTM(128, activation="relu", input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation="relu", return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(train_y.shape[1]))

model.compile(optimizer="adam", loss="mse")
model.summary()

callback = EarlyStopping(monitor="loss", patience=10)
model.fit(train_x, train_y, epochs=100, batch_size=16, validation_split=0.1, callbacks=[callback], verbose=1)

#Save the entire model
model.save("models/multivariate_lstm128x64_technical_monthly.h5")

prediction = model.predict(test_x)
prediction_copies = np.repeat(prediction, df_combined_close.shape[1], axis=-1)
y_pred = scaler.inverse_transform(prediction_copies)[:,0]
y_actual = df_intel["close"][-len(y_pred):].values

rmse = mean_squared_error(y_actual, y_pred, squared=False)
print(f'RMSE: {rmse}')

plt.plot(y_pred)
plt.plot(y_actual)
plt.show()

#Pattern Matching





