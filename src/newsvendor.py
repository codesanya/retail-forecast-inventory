import numpy as np
import pandas as pd
from scipy.stats import norm

def compute_service_level(unit_cost: float, unit_price: float, holding_cost: float, stockout_cost: float) -> float:
    # Underage cost: min(stockout_cost, unit_price - unit_cost) as a proxy
    cu = max(unit_price - unit_cost, stockout_cost)
    co = holding_cost
    alpha = cu / (cu + co) if (cu + co) > 0 else 0.5
    return min(max(alpha, 0.01), 0.99)

def recommend_orders(forecast_df: pd.DataFrame, history_df: pd.DataFrame,
                     unit_cost: float, unit_price: float, holding_cost: float, stockout_cost: float):
    """
    Inputs:
      forecast_df: columns [date, store_id, sku_id, forecast]
      history_df: original actuals with columns [date, store_id, sku_id, sales]
    Output:
      DataFrame with recommended order quantities per row.
    """
    # Estimate residual std per SKU-store from recent window
    his = history_df.copy()
    # fallback sigma if errors unavailable
    default_sigma = 3.0

    # Compute sigma per key using rolling residuals if prior forecasts exist; here use sales rolling std as proxy
    recent = his.groupby(['store_id','sku_id'])['sales'].rolling(28).std().reset_index()
    his = his.merge(recent, on=['store_id','sku_id','level_1'], how='left')
    his.rename(columns={'sales_x':'sales','sales_y':'sales_rollstd_28'}, inplace=True)

    alphas = {}
    alpha = compute_service_level(unit_cost, unit_price, holding_cost, stockout_cost)
    z = norm.ppf(alpha)

    # Determine sigma per key
    sigma_map = his.groupby(['store_id','sku_id'])['sales_rollstd_28'].agg(lambda s: s.dropna().iloc[-1] if s.dropna().size>0 else np.nan).to_dict()

    recs = []
    for i, r in forecast_df.iterrows():
        key = (r['store_id'], r['sku_id'])
        sigma = sigma_map.get(key, np.nan)
        if np.isnan(sigma) or sigma <= 0:
            sigma = default_sigma
        q = max(0.0, r['forecast'] + z * sigma)
        recs.append({**r, 'recommended_order_qty': float(np.round(q, 0))})
    return pd.DataFrame(recs)
