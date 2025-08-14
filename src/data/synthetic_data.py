import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
import os

def make_sine_trend(n_days, base, amplitude, noise, promo_prob=0.1, holiday_prob=0.05, price=10.0):
    days = np.arange(n_days)
    seasonal = amplitude * (1 + np.sin(2*np.pi*days/7) * 0.3 + np.sin(2*np.pi*days/365) * 0.7)
    baseline = base + 0.01*days  # slight upward drift
    demand = np.maximum(0, baseline + seasonal + np.random.normal(0, noise, size=n_days)).astype(float)
    on_promo = (np.random.rand(n_days) < promo_prob).astype(int)
    holiday = (np.random.rand(n_days) < holiday_prob).astype(int)
    # promotion uplift
    demand = demand * (1 + 0.25*on_promo + 0.15*holiday)
    price_series = price * (1 - 0.10*on_promo)  # discount on promo
    return demand, on_promo, holiday, price_series

def generate(n_days=365*2, n_stores=3, n_skus=5, start_date="2023-01-01", seed=7):
    np.random.seed(seed)
    start = pd.to_datetime(start_date)
    rows = []
    for s in range(n_stores):
        for k in range(n_skus):
            base = np.random.uniform(20, 120)
            amplitude = np.random.uniform(2, 25)
            noise = np.random.uniform(2, 12)
            price = np.random.uniform(8, 20)
            demand, promo, holiday, price_series = make_sine_trend(n_days, base, amplitude, noise,
                                                                   promo_prob=0.12, holiday_prob=0.06, price=price)
            dates = pd.date_range(start, periods=n_days, freq='D')
            df = pd.DataFrame({
                'date': dates,
                'store_id': str(s+1),
                'sku_id': str(k+1),
                'sales': np.round(demand, 0).astype(int),
                'on_promo': promo,
                'holiday': holiday,
                'price': np.round(price_series, 2)
            })
            rows.append(df)
    return pd.concat(rows, ignore_index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='data/raw/sales.csv')
    parser.add_argument('--n_days', type=int, default=540)
    parser.add_argument('--n_stores', type=int, default=3)
    parser.add_argument('--n_skus', type=int, default=5)
    args = parser.parse_args()
    df = generate(n_days=args.n_days, n_stores=args.n_stores, n_skus=args.n_skus)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote synthetic dataset to {args.out} with {len(df):,} rows")
