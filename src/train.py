import os
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.utils.io import read_data, write_df
from src.features.build_features import prepare_features

FEATURE_COLS = [
    # calendar
    'dow','week','month','year','is_weekend',
    # drivers
    'price','on_promo','holiday',
    # lags/rolls added dynamically
]

def get_feature_columns(df: pd.DataFrame):
    cols = FEATURE_COLS.copy()
    cols += [c for c in df.columns if c.startswith('sales_lag_') or c.startswith('sales_roll') or c.startswith('promo_last')]
    # Keep only existing
    return [c for c in cols if c in df.columns]

def train_xgb(df: pd.DataFrame, artifacts_dir: str, test_days: int = 28):
    df_feat = prepare_features(df)
    df_feat = df_feat.dropna().reset_index(drop=True)
    df_feat = df_feat.sort_values(['store_id','sku_id','date'])

    feature_cols = get_feature_columns(df_feat)
    results = []
    os.makedirs(artifacts_dir, exist_ok=True)

    for (store, sku), g in df_feat.groupby(['store_id','sku_id']):
        # time-based split: last test_days as test
        g = g.sort_values('date')
        X = g[feature_cols]
        y = g['sales']

        if len(g) <= test_days + 30:
            # skip very short series
            continue

        X_train, X_test = X.iloc[:-test_days], X.iloc[-test_days:]
        y_train, y_test = y.iloc[:-test_days], y.iloc[-test_days:]

        scaler = StandardScaler(with_mean=False)  # sparse-friendly
        X_train_scaled = scaler.fit_transform(X_train.fillna(0))
        X_test_scaled = scaler.transform(X_test.fillna(0))

        model = XGBRegressor(
            max_depth=6,
            n_estimators=400,
            learning_rate=0.06,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)

        # Persist artifacts per (store, sku)
        key = f"store={store}__sku={sku}"
        joblib.dump({
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols
        }, os.path.join(artifacts_dir, f"{key}.joblib"))

        results.append({'store_id': store, 'sku_id': sku, 'mae': mae, 'rmse': rmse, 'n_test': len(y_test)})

    metrics = pd.DataFrame(results)
    return metrics

def forecast_next(df: pd.DataFrame, artifacts_dir: str, horizon: int = 14):
    # For each SKU, iteratively forecast horizon days using last known features
    df = df.sort_values(['store_id','sku_id','date']).copy()
    max_date = df['date'].max()
    rows = []
    # Precompute rolling stats efficiently by reusing prepare_features on an extended frame
    for (store, sku), g in df.groupby(['store_id','sku_id']):
        g2 = g.copy()
        # Load artifacts
        key = f"store={store}__sku={sku}"
        path = os.path.join(artifacts_dir, f"{key}.joblib")
        if not os.path.exists(path):
            continue
        art = joblib.load(path)
        model, scaler, feature_cols = art['model'], art['scaler'], art['feature_cols']

        last_date = g2['date'].max()
        for h in range(1, horizon+1):
            pred_date = last_date + pd.Timedelta(days=1)
            # create a one-row future feature frame using latest info
            tmp = pd.DataFrame({
                'date':[pred_date],
                'store_id':[store],
                'sku_id':[sku],
                'sales':[np.nan],
                'on_promo':[0],
                'holiday':[0],
                'price':[g2['price'].iloc[-1] if 'price' in g2.columns else 0]
            })
            g2 = pd.concat([g2, tmp], ignore_index=True)
            g2_feat = prepare_features(g2).iloc[[-1]]  # last row
            X = g2_feat[feature_cols].fillna(0)
            Xs = scaler.transform(X)
            yhat = float(model.predict(Xs)[0])
            g2.loc[g2.index[-1], 'sales'] = max(0.0, yhat)
            rows.append({'date': pred_date, 'store_id': store, 'sku_id': sku, 'forecast': yhat})
            last_date = pred_date
    return pd.DataFrame(rows)

if __name__ == '__main__':
    import argparse, yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--artifacts', type=str, default='models/artifacts')
    parser.add_argument('--horizon', type=int, default=14)
    parser.add_argument('--test_days', type=int, default=28)
    args = parser.parse_args()
    df = read_data(args.data)
    metrics = train_xgb(df, args.artifacts, test_days=args.test_days)
    os.makedirs('data/processed', exist_ok=True)
    metrics.to_csv('data/processed/metrics.csv', index=False)
    print(metrics.head())
    fcst = forecast_next(df, args.artifacts, horizon=args.horizon)
    fcst.to_csv('data/processed/forecast.csv', index=False)
    print('Wrote forecast to data/processed/forecast.csv')
