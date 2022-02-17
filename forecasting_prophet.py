import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

df_ts = pd.read_csv("data/NVDA_1day_prep.csv", index_col=0)
df_ts = df_ts.iloc[df_ts.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)

df_prophet = df_ts.drop(columns=["open","high","low","close","volume","MA_30","MA_100","MA_200","MACD","MACD_h","MACD_s","RSI"])
df_prophet = df_prophet.rename(columns={"datetime":"ds", "OHLC_avg":"y"})
"""
df_prophet["volume"] = df_ts["volume"]
df_prophet["rsi"] = df_ts["RSI"]
df_prophet["macd"] = df_ts["MACD"]
df_prophet["macd_h"] = df_ts["MACD_h"]
df_prophet["macd_s"] = df_ts["MACD_s"]
df_prophet["ma_30"] = df_ts["MA_30"]
df_prophet["ma_100"] = df_ts["MA_100"]
df_prophet["ma_200"] = df_ts["MA_200"]
"""

m = Prophet()
"""
m.add_regressor("volume")
m.add_regressor("rsi")
m.add_regressor("macd")
m.add_regressor("macd_h")
m.add_regressor("macd_s")
m.add_regressor("ma_30")
m.add_regressor("ma_100")
m.add_regressor("ma_200")
"""
metrics = m.fit(df_prophet)

df_future = m.make_future_dataframe(periods=365)
future_no_weekends = df_future[df_future['ds'].dt.dayofweek < 5]

forecast = m.predict(df_prophet.drop(columns="y"))
m.plot(forecast)
plt.show()