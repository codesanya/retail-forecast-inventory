# Retail Sales Forecasting and Inventory Optimization

Production-ready, end-to-end analytics project demonstrating time-series demand forecasting and inventory optimization with a Streamlit dashboard.

## Value Proposition
- Forecast daily product demand at store/SKU granularity.
- Recommend inventory order quantities to minimize stockouts and overstock (newsvendor model).
- Modular ETL → Feature Engineering → Modeling → Optimization → Dashboard.

## Tech Stack
- **Python**: pandas, numpy, scikit-learn, xgboost, statsmodels
- **Optimization**: Newsvendor policy via cost-based service level
- **App**: Streamlit
- **Packaging**: joblib artifacts, YAML config

## Repo Structure
```
.
├── config/
│   └── config.yaml
├── data/
│   ├── raw/                 # place your raw csv here (optional: use synthetic provided)
│   └── processed/
├── models/
│   └── artifacts/           # trained models & scalers
├── notebooks/               # optional EDA
├── scripts/
│   └── run_pipeline.py      # one-click training + forecasting + optimization
├── src/
│   ├── data/synthetic_data.py
│   ├── features/build_features.py
│   ├── models/train.py
│   ├── models/forecast.py
│   ├── optimization/newsvendor.py
│   ├── utils/io.py
│   └── visualization/app.py # Streamlit UI
├── tests/
├── requirements.txt
└── README.md
```

## Data Schema
Input CSV required columns (synthetic file provided):
- `date` (YYYY-MM-DD)
- `store_id` (string/int)
- `sku_id` (string/int)
- `sales` (non-negative numeric units)
- Optional drivers: `price`, `on_promo` (0/1), `holiday` (0/1)

## Quickstart
1. Create a virtual environment and install dependencies:
```
pip install -r requirements.txt
```
2. (Optional) Use provided synthetic dataset:
```
python -m src.data.synthetic_data --out data/raw/sales.csv
```
3. Run the end-to-end pipeline (train → forecast → optimize):
```
python scripts/run_pipeline.py --data data/raw/sales.csv --horizon 14
```
Artifacts and outputs will be written under `models/artifacts` and `data/processed`.

4. Launch the Streamlit app:
```
streamlit run src/visualization/app.py
```

## Configuration
See `config/config.yaml` to adjust horizons, algorithms, and inventory costs:
- `holding_cost`, `stockout_cost`, `unit_cost`, `unit_price`

## Inventory Optimization (Newsvendor)
We compute a service level target: \( \alpha = \frac{c_u}{c_u + c_o} \) where
- underage cost \( c_u \) ≈ `stockout_cost` or `unit_price - unit_cost`
- overage cost \( c_o \) ≈ `holding_cost`

Assuming forecast errors are approximately normal with std `sigma`, the optimal order is:
\( Q^* = \hat{\mu} + z_\alpha \cdot \sigma \).

## License
MIT
