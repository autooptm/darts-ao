"""darts benchmark: one N-BEATS model per series, trained and used to forecast.

    python ao_bench.py                  # 24 series x 300 points, 3 epochs each
    python ao_bench.py --series 6 --epochs 2

For each synthetic monthly series: scale it, fit an NBEATSModel for a few
epochs, forecast 12 steps, inverse the scaling and score MAPE against the
held-out tail. The loop over series is the unit of work. Writes
out/summary.json and out/forecasts.csv.
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings("ignore")


def make_series(i, length, rng):
    from darts import TimeSeries
    t = np.arange(length)
    y = (100 + rng.uniform(-0.3, 0.8) * t + rng.uniform(10, 40) * np.sin(2 * np.pi * t / 12)
         + rng.standard_normal(length) * rng.uniform(2, 8))
    idx = pd.date_range("1990-01-01", periods=length, freq="MS")
    return TimeSeries.from_times_and_values(idx, y.astype(np.float32), columns=[f"s{i:03d}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=int, default=24)
    ap.add_argument("--length", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from darts.dataprocessing.transformers import Scaler
    from darts.metrics import mape
    from darts.models import NBEATSModel

    rng = np.random.default_rng(0)
    series = [make_series(i, args.length, rng) for i in range(args.series)]
    print(f"series: {len(series)} x {args.length} months", flush=True)

    rows, walls, preds = [], [], []
    for i, s in enumerate(series):
        t0 = time.perf_counter()
        train, test = s[:-args.horizon], s[-args.horizon:]
        scaler = Scaler()
        train_s = scaler.fit_transform(train)
        model = NBEATSModel(input_chunk_length=36, output_chunk_length=args.horizon,
                            n_epochs=args.epochs, batch_size=32, random_state=i,
                            pl_trainer_kwargs={"enable_progress_bar": False,
                                               "enable_model_summary": False,
                                               "logger": False})
        model.fit(train_s, verbose=False)
        pred = scaler.inverse_transform(model.predict(n=args.horizon, verbose=False))
        err = float(mape(test, pred))
        dt = time.perf_counter() - t0
        walls.append(dt)
        preds.append(pred.to_dataframe())
        rows.append({"series": i, "s": round(dt, 4), "mape": round(err, 3)})
        print(f"  series {i:2d} {dt:.3f}s mape {err:.2f}%", flush=True)

    pd.concat(preds, axis=1).to_csv(os.path.join(args.out, "forecasts.csv"))
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"series": rows, "total_s": sum(walls),
                   "median_s": float(np.median(walls)),
                   "mean_mape": float(np.mean([r["mape"] for r in rows]))}, fh, indent=1)
    print(f"done: {len(walls)} series, median {np.median(walls):.3f}s, total {sum(walls):.2f}s")


if __name__ == "__main__":
    main()
