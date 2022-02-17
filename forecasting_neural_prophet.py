import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

df_ts = pd.read_csv("data/INTC_1day_prep.csv", index_col=0)
df_ts = df_ts.iloc[df_ts.apply(pd.Series.first_valid_index).max():].reset_index(drop=True)

df_prophet = df_ts.drop(columns=["open","high","low","close","volume","MA_30","MA_100","MA_200","MACD","MACD_h","MACD_s","RSI"])
df_prophet = df_prophet.rename(columns={"datetime":"ds", "OHLC_avg":"y"})

m = NeuralProphet()
metrics = m.fit(df_prophet, freq="D")

df_future = m.make_future_dataframe(df_prophet,periods=365)
future_no_weekends = df_future[df_future['ds'].dt.dayofweek < 5]

forecast = m.predict(future_no_weekends)
m.plot(forecast)
plt.show()