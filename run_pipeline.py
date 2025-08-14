import argparse, os, yaml
from src.utils.io import read_data, write_df
from src.models.train import train_xgb, forecast_next
from src.optimization.newsvendor import recommend_orders

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--horizon', type=int, default=14)
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print("Reading data...")
    df = read_data(args.data)
    print(df.head())

    print("Training models...")
    metrics = train_xgb(df, artifacts_dir='models/artifacts', test_days=cfg.get('test_days',28))
    write_df(metrics, 'data/processed/metrics.csv')
    print("Metrics written to data/processed/metrics.csv")

    print("Forecasting next {} days...".format(args.horizon))
    fcst = forecast_next(df, artifacts_dir='models/artifacts', horizon=args.horizon)
    write_df(fcst, 'data/processed/forecast.csv')

    econ = cfg['economics']
    print("Optimizing inventory via newsvendor policy...")
    recs = recommend_orders(fcst, df, **econ)
    write_df(recs, 'data/processed/recommendations.csv')
    print("Done. Outputs in data/processed and models/artifacts.")
