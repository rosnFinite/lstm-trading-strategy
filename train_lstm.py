import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from data_preparation import Data_Preparation

prep_data = Data_Preparation()
prep_data.prepare_data()


"""Model"""
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(30,1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2))

model.compile(optimizer="adam", loss="mean_squared_error")

callback = EarlyStopping(monitor="loss", patience=10)
model.fit(prep_data.x_train, prep_data.y_train, epochs=250, batch_size=32, validation_split=0.1, callbacks=[callback])
model.save("models/NextDay+MA/lstm50x50x50.h5")