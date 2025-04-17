"""Utility functions for data processing and visualization."""

import pandas as pd

def transform_df(df: pd.DataFrame) -> pd.DataFrame:
    """transform the df for datetime and spot price"""
    df[['start_timestamp', 'end_timestamp']] = df['MTU (CET/CEST)'].str.split(' - ', expand=True)
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], format='%d.%m.%Y %H:%M')
    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'], format='%d.%m.%Y %H:%M')
    df = df.rename(columns={'Day-ahead Price [EUR/MWh]': 'spot_price'})
    return df

def split_60_mins_data_into_15_mins(data_60: pd.DataFrame) -> pd.DataFrame:
    """split 60 mins data into 15 mins amd keep the spot price within the hour the same."""
    # Ensure that 'Start Timestamp' is the index and is a datetime# Ensure that 'Start Timestamp' is the index and is a datetime
    data_60.set_index('start_timestamp', inplace=True)
    data_60.index = pd.to_datetime(data_60.index)

    # Resample to 15-minutes intervals
    data_15 = data_60.resample('15T').asfreq()

    # Forward fill the 'Day-ahead Price [EUR/MWh]' column
    data_15['spot_price'] = data_15['spot_price'].ffill()
    if 'y_pred' in data_15.columns:
        data_15['y_pred'] = data_15['y_pred'].ffill()

    # Reset the index
    data_15.reset_index(inplace=True)

    return data_15