import os
import pandas as pd
from typing import Optional

def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date'])
    df['store_id'] = df['store_id'].astype(str)
    df['sku_id'] = df['sku_id'].astype(str)
    # Basic sanitization
    df = df.sort_values(['store_id','sku_id','date']).reset_index(drop=True)
    return df

def write_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
