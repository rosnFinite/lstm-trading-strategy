import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

stocks = ["intel","nvidia","amazon","microsoft","apple","google"]
df_intel = pd.read_csv("data/Technologie/INTC_1day_prep.csv", index_col=0, parse_dates=True)
df_intel = df_intel.iloc[df_intel.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)
df_nvidia = pd.read_csv("data/Technologie/NVDA_1day_prep.csv", index_col=0, parse_dates=True)
df_nvidia = df_nvidia.iloc[df_nvidia.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)
df_amazon = pd.read_csv("data/Technologie/AMZN_1day_prep.csv", index_col=0, parse_dates=True)
df_amazon = df_amazon.iloc[df_amazon.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)
df_microsoft = pd.read_csv("data/Technologie/MSFT_1day_prep.csv", index_col=0, parse_dates=True)
df_microsoft = df_microsoft.iloc[df_microsoft.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)
df_apple = pd.read_csv("data/Technologie/AAPL_1day_prep.csv", index_col=0, parse_dates=True)
df_apple = df_apple.iloc[df_apple.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)
df_google = pd.read_csv("data/Technologie/GOOGL_1day_prep.csv", index_col=0, parse_dates=True)
df_google = df_google.iloc[df_google.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)

df_combined = pd.DataFrame()
df_combined["intel"] = df_intel["close"]
df_combined["intel_ma"] = df_intel["MA_30"]
df_combined["nvidia"] = df_nvidia["close"]
df_combined["nvidia_ma"] = df_nvidia["MA_30"]
df_combined["amazon"] = df_amazon["close"]
df_combined["amazon_ma"] = df_amazon["MA_30"]
df_combined["microsoft"] = df_microsoft["close"]
df_combined["microsoft_ma"] = df_microsoft["MA_30"]
df_combined["apple"] = df_apple["close"]
df_combined["apple_ma"] = df_apple["MA_30"]
df_combined["google"] = df_google["close"]
df_combined["google_ma"] = df_google["MA_30"]

scaler = MinMaxScaler(feature_range=(0,1))

df_combined[["intel","intel_ma","nvidia","nvidia_ma","amazon","amazon_ma","microsoft","microsoft_ma","apple","apple_ma","google","google_ma"]] = scaler.fit_transform(df_combined[["intel",
                                                                                                                                                                                    "intel_ma",                            
                                                                                                                                                                                    "nvidia",
                                                                                                                                                                                    "nvidia_ma",
                                                                                                                                                                                    "amazon",
                                                                                                                                                                                    "amazon_ma",
                                                                                                                                                                                    "microsoft",
                                                                                                                                                                                    "microsoft_ma",
                                                                                                                                                                                    "apple",
                                                                                                                                                                                    "apple_ma",
                                                                                                                                                                                    "google",
                                                                                                                                                                                    "google_ma"]])

num_past = 30
split_index = int(len(df_combined)*0.8)
combined_x_train = []
combined_y_train = []
combined_x_test = []
combined_y_test = []
x_tests = []
y_tests = []
for stock in stocks:
    print(f'Start preparing data for {stock}')
    values = df_combined.filter([stock]).values
    ma_values = df_combined.filter([stock+"_ma"]).values
    train = values[:split_index,:]
    test = values[split_index:,:]

    x_train = []
    y_train = []
    for i in range(num_past, split_index-30):
        x_train.append(values[i-num_past:i, 0])
        y_train.append([values[i, 0],ma_values[i+30, 0]])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    combined_x_train.append(x_train)
    combined_y_train.append(y_train)
    print(f'Finished preparing training data -> X {x_train.shape}, Y {y_train.shape}')

    x_test = []
    y_test = []
    for i in range(split_index+num_past, len(df_combined)-30):
        x_test.append(values[i-num_past:i, 0])
        #print(ma_values[i+30])
        y_test.append([values[i, 0],ma_values[i+30, 0]])
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    x_tests.append(x_test)
    y_tests.append(y_test)
    combined_x_test.append(x_test)
    combined_y_test.append(y_test)
    print(f'Finished preparing test data -> X {x_test.shape}, Y {y_train.shape}')

combined_x_train = np.array(combined_x_train)
combined_x_train = np.reshape(combined_x_train, (combined_x_train.shape[1]*combined_x_train.shape[0], 30, 1))
combined_y_train = np.array(combined_y_train)
# combined_y_train = np.reshape(combined_y_train, (-1, 1)) #single output
combined_y_train = np.reshape(combined_y_train, (combined_y_train.shape[1]*combined_y_train.shape[0], 2)) #multi output
combined_x_test = np.array(combined_x_test)
combined_x_test = np.reshape(combined_x_test, (combined_x_test.shape[1]*combined_x_test.shape[0], 30, 1))
combined_y_test = np.array(combined_y_test)
#combined_y_test = np.reshape(combined_y_test, (-1, 1)) #single output
print(combined_y_test)
combined_y_test = np.reshape(combined_y_test, (combined_y_test.shape[1]*combined_y_test.shape[0], 2)) #multi output


"""Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(num_past,1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2))

model.compile(optimizer="adam", loss="mean_squared_error")

callback = EarlyStopping(monitor="loss", patience=10)
model.fit(combined_x_train, combined_y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[callback])
model.save("models/univariate_lstm50x50x50_ma+day.h5")
"""

model = load_model("models/univariate_lstm50x50x50_ma+day.h5")

x = x_tests[0]
#prediction = model.predict(x)
prediction = model.predict(x)[:,1]
prediction = np.reshape(prediction, (-1,1)) # nicht für single output 
prediction_copies = np.repeat(prediction, df_combined.shape[1], axis=-1)
y_pred = scaler.inverse_transform(prediction_copies)[:,0]
y_actual = df_intel["MA_30"][-len(y_pred):].values

rmse = mean_squared_error(y_actual, y_pred, squared=False)
print(f'RMSE: {rmse}')

plt.plot(y_pred)
plt.plot(y_actual)
plt.show()
