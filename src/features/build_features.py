import pandas as pd
import numpy as np

BASE_LAGS = [1,2,3,7,14,28]
ROLL_WINDOWS = [7,14,28]

def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df['dow'] = df['date'].dt.dayofweek
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    return df

def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['store_id','sku_id','date'])
    for lag in BASE_LAGS:
        df[f'sales_lag_{lag}'] = df.groupby(['store_id','sku_id'])['sales'].shift(lag)
    for w in ROLL_WINDOWS:
        df[f'sales_rollmean_{w}'] = df.groupby(['store_id','sku_id'])['sales'].shift(1).rolling(w).mean()
        df[f'sales_rollstd_{w}'] = df.groupby(['store_id','sku_id'])['sales'].shift(1).rolling(w).std()
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_calendar(df.copy())
    df = add_lags(df)
    # Interactions
    if 'on_promo' in df.columns:
        df['promo_last7'] = df.groupby(['store_id','sku_id'])['on_promo'].shift(1).rolling(7).mean()
    # Drop early rows with NaNs due to lags
    return df
