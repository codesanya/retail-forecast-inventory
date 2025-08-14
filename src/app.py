import os
import pandas as pd
import streamlit as st
import yaml
import joblib
from src.utils.io import read_data, write_df
from src.models.train import train_xgb, forecast_next
from src.optimization.newsvendor import recommend_orders

st.set_page_config(page_title="Retail Forecast & Inventory Optimizer", layout="wide")
st.title("Retail Sales Forecasting and Inventory Optimization")
st.caption("Upload data, train models, forecast demand, and get order recommendations.")

with open('config/config.yaml','r') as f:
    cfg = yaml.safe_load(f)

data_file = st.file_uploader("Upload CSV (date, store_id, sku_id, sales, [price, on_promo, holiday])", type=['csv'])
use_synth = st.checkbox("Use synthetic dataset", value=True, help="Generate a demo dataset if no file is uploaded.")
horizon = st.number_input("Forecast horizon (days)", min_value=7, max_value=60, value=cfg.get('horizon',14), step=1)

if st.button("Run Pipeline"):
    if data_file is not None:
        raw_path = 'data/raw/upload.csv'
        os.makedirs('data/raw', exist_ok=True)
        with open(raw_path, 'wb') as f:
            f.write(data_file.getvalue())
    elif use_synth:
        from src.data.synthetic_data import generate
        df_synth = generate(n_days=540, n_stores=3, n_skus=5)
        raw_path = 'data/raw/sales.csv'
        df_synth.to_csv(raw_path, index=False)
    else:
        st.error("Please upload a CSV or enable synthetic dataset.")
        st.stop()

    df = read_data(raw_path)
    st.write("Data sample:", df.head())

    st.info("Training models per store/SKU...")
    metrics = train_xgb(df, artifacts_dir='models/artifacts', test_days=cfg.get('test_days',28))
    st.success("Training complete.")
    st.dataframe(metrics)

    st.info("Forecasting...")
    fcst = forecast_next(df, artifacts_dir='models/artifacts', horizon=int(horizon))
    st.dataframe(fcst.head())

    econ = cfg['economics']
    st.info("Optimizing inventory (Newsvendor)...")
    recs = recommend_orders(fcst, df, **econ)
    st.dataframe(recs.head(20))

    out_fcst = 'data/processed/forecast.csv'
    out_recs = 'data/processed/recommendations.csv'
    os.makedirs('data/processed', exist_ok=True)
    fcst.to_csv(out_fcst, index=False)
    recs.to_csv(out_recs, index=False)
    st.success("Artifacts written to data/processed and models/artifacts.")
    st.download_button("Download Forecast CSV", data=open(out_fcst,'rb').read(), file_name='forecast.csv')
    st.download_button("Download Recommendations CSV", data=open(out_recs,'rb').read(), file_name='recommendations.csv')
