"""
File contains editable settings for training and backtesting process
"""
import pandas as pd

class Config():
    df_ts = pd.read_csv("data/dummy.csv", index_col=0, parse_dates=True)

    # Train Split
    TRAIN_SPLIT = 0.65

    # Backtesting should only be run on unseen data
    # Therefor is the start_index set to the index of
    # the first datapoint of the test set
    SIMULATION_START_INDEX = int(len(df_ts)*TRAIN_SPLIT)


