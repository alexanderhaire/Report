#!/usr/bin/env python3
"""
chem_futures_linkage.py — Correlation tests and continuously updating ML between
chemical prices and exchange‑traded futures.

Features
- Ingest tidy CSVs (date, symbol, price) for chemicals and futures.
- Align to a common calendar; forward-fill gaps; build returns.
- Correlation suite: Pearson, Spearman, rolling corr, cross‑correlations (lead/lag),
  Granger causality, cointegration (if statsmodels is available).
- ML model: SGDRegressor with partial_fit for incremental/online learning.
- Saves artifacts: processed dataset, correlation CSVs, plots (matplotlib), model.pkl, metrics.json.

CSV expectations
  chemicals.csv: date,symbol,price
  futures.csv:   date,symbol,price    # (use settle/close as price)
Dates can be YYYY‑MM‑DD or ISO 8601; script will parse.

Quick start
  # 1) Prep & feature build
  python chem_futures_linkage.py prep --chem chemicals.csv --fut futures.csv --outdir out

  # 2) Correlation & causality analysis (plots + CSVs)
  python chem_futures_linkage.py corr --dataset out/dataset.parquet --outdir out

  # 3) Train or keep training a streaming model for a given chemical symbol
  python chem_futures_linkage.py train --dataset out/dataset.parquet --symbol KTS \
      --horizon 1D --lags 5 --outdir out --model out/model_KTS.pkl

  # 4) Predict the next return given latest data
  python chem_futures_linkage.py predict --dataset out/dataset.parquet --symbol KTS --model out/model_KTS.pkl

Schedule retrains
  Windows Task Scheduler or cron can run the same 'train' command daily/hourly.
"""

import argparse, os, json, math, warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import matplotlib.pyplot as plt

# Optional imports
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint, grangercausalitytests
    HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

plt.rcParams.update({"figure.dpi": 110})

# ---------------------------- IO utils ----------------------------

def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def _parse_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


# ---------------------------- 1) PREP ----------------------------

@dataclass
class PrepArgs:
    chem: str
    fut: str
    outdir: str
    freq: str = "B"  # business daily
    method: str = "ffill"


def prep_dataset(args: PrepArgs) -> str:
    _ensure_outdir(args.outdir)

    chem = pd.read_csv(args.chem)
    fut = pd.read_csv(args.fut)
    for df, name in [(chem, "chemicals"), (fut, "futures")]:
        if not set(["date", "symbol", "price"]).issubset(df.columns):
            raise ValueError(f"{name} CSV must have columns: date,symbol,price")
        df["date"] = _parse_date(df["date"])
    chem = chem.dropna(subset=["date", "price"]).copy()
    fut = fut.dropna(subset=["date", "price"]).copy()

    # Pivot to wide panels
    chem_w = chem.pivot_table(index="date", columns="symbol", values="price", aggfunc="last")
    fut_w = fut.pivot_table(index="date", columns="symbol", values="price", aggfunc="last")

    # Reindex to common calendar
    start = min(chem_w.index.min(), fut_w.index.min())
    end = max(chem_w.index.max(), fut_w.index.max())
    idx = pd.date_range(start, end, freq=args.freq)
    chem_w = chem_w.reindex(idx).sort_index().astype(float)
    fut_w = fut_w.reindex(idx).sort_index().astype(float)

    # Fill gaps sensibly (carry forward; do not create future info)
    if args.method == "ffill":
        chem_w = chem_w.ffill()
        fut_w = fut_w.ffill()
    elif args.method == "bfill":
        chem_w = chem_w.bfill()
        fut_w = fut_w.bfill()

    # Compute log returns (safer for stationarity)
    chem_r = np.log(chem_w).diff()
    fut_r = np.log(fut_w).diff()
    chem_r.columns = [f"chem_{c}" for c in chem_r.columns]
    fut_r.columns = [f"fut_{c}" for c in fut_r.columns]

    # Merge into single frame
    data = pd.concat([chem_w.add_prefix("lvl_chem_"), fut_w.add_prefix("lvl_fut_"), chem_r, fut_r], axis=1)
    data.index.name = "date"

    outpath = os.path.join(args.outdir, "dataset.parquet")
    data.to_parquet(outpath)

    # Save metadata
    meta = {
        "chem_symbols": sorted(chem.pivot_table(index="symbol").index.unique().tolist()),
        "fut_symbols": sorted(fut.pivot_table(index="symbol").index.unique().tolist()),
        "freq": args.freq,
        "rows": int(len(data)),
        "start": data.index.min().strftime("%Y-%m-%d"),
        "end": data.index.max().strftime("%Y-%m-%d"),
    }
    with open(os.path.join(args.outdir, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved dataset to {outpath} with {meta['rows']} rows.")
    return outpath


# ---------------------------- 2) CORRELATION ----------------------------

@dataclass
class CorrArgs:
    dataset: str
    outdir: str
    window: int = 60  # rolling days
    max_lag: int = 10 # cross-corr +/- lags in days


def corr_and_causality(args: CorrArgs):
    _ensure_outdir(args.outdir)
    df = pd.read_parquet(args.dataset)

    chem_cols = [c for c in df.columns if c.startswith("chem_") and not c.startswith("lvl_")]
    fut_cols = [c for c in df.columns if c.startswith("fut_") and not c.startswith("lvl_")]

    # 2.1 Static Pearson & Spearman
    pearson = df[chem_cols + fut_cols].corr(method="pearson").loc[chem_cols, fut_cols]
    spearman = df[chem_cols + fut_cols].corr(method="spearman").loc[chem_cols, fut_cols]

    pearson.to_csv(os.path.join(args.outdir, "corr_pearson.csv"))
    spearman.to_csv(os.path.join(args.outdir, "corr_spearman.csv"))

    # 2.2 Rolling correlation (save per chem/future pair top‑N by absolute corr)
    top_pairs = (
        pearson.abs()
        .stack()
        .sort_values(ascending=False)
        .head(6)
        .index.tolist()
    )

    for chem_col, fut_col in top_pairs:
        rc = df[chem_col].rolling(args.window).corr(df[fut_col])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(df.index, rc)
        ax.set_title(f"Rolling {args.window}D corr: {chem_col} vs {fut_col}")
        ax.set_xlabel("Date"); ax.set_ylabel("Correlation")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"rollingcorr_{chem_col}_vs_{fut_col}.png"))
        plt.close(fig)

    # 2.3 Cross‑correlations: lead/lag heatmaps (chem vs each future)
    cc_rows = []
    for chem_col in chem_cols:
        for fut_col in fut_cols:
            for k in range(-args.max_lag, args.max_lag + 1):
                if k < 0:
                    corr = df[chem_col].corr(df[fut_col].shift(-k))
                else:
                    corr = df[chem_col].shift(k).corr(df[fut_col])
                cc_rows.append({"chem": chem_col, "future": fut_col, "lag": k, "corr": corr})
    cc = pd.DataFrame(cc_rows)
    cc.to_csv(os.path.join(args.outdir, "crosscorr.csv"), index=False)

    # 2.4 Granger & Cointegration (if available)
    results = {"granger": {}, "coint": {}}
    if HAVE_STATSMODELS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for chem_col in chem_cols:
                for fut_col in fut_cols:
                    # Granger: does future help predict chem? (up to 5 lags)
                    try:
                        pair = df[[chem_col, fut_col]].dropna()
                        g = grangercausalitytests(pair[[chem_col, fut_col]], maxlag=5, verbose=False)
                        # Store smallest p‑value over lags for F‑test
                        best_p = min(g[i][0]["ssr_ftest"][1] for i in g.keys())
                        results["granger"].setdefault(chem_col, {})[fut_col] = best_p
                    except Exception:
                        continue
                    # Engle‑Granger cointegration
                    try:
                        score, pval, _ = coint(pair[chem_col], pair[fut_col])
                        results["coint"].setdefault(chem_col, {})[fut_col] = float(pval)
                    except Exception:
                        continue
        # Save as CSV heatmaps
        if results["granger"]:
            gmat = pd.DataFrame(results["granger"]).T
            gmat.to_csv(os.path.join(args.outdir, "granger_minp.csv"))
        if results["coint"]:
            cmat = pd.DataFrame(results["coint"]).T
            cmat.to_csv(os.path.join(args.outdir, "cointegration_p.csv"))

    # Save summary JSON
    summary = {
        "pearson_top_pairs": [(a,b,float(pearson.loc[a,b])) for a,b in top_pairs],
        "have_statsmodels": HAVE_STATSMODELS,
    }
    with open(os.path.join(args.outdir, "corr_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Correlation + causality artifacts written to:", args.outdir)


# ---------------------------- 3) ML (online) ----------------------------

@dataclass
class TrainArgs:
    dataset: str
    symbol: str  # chemical symbol to predict (return)
    horizon: str # e.g., "1D" (predict next‑day return)
    lags: int
    outdir: str
    model: str
    step: int = 5  # partial_fit every N rows


def _build_features(df: pd.DataFrame, chem_symbol: str, lags: int, horizon: str) -> Tuple[pd.DataFrame, pd.Series]:
    y_col = f"chem_{chem_symbol}"
    if y_col not in df.columns:
        raise ValueError(f"Target column {y_col} not found in dataset. Available: {[c for c in df.columns if c.startswith('chem_')]}")

    # Target: future return aligned to requested horizon
    horizon_n = pd.Timedelta(horizon)
    if horizon_n <= pd.Timedelta(0):
        raise ValueError("horizon must be positive, e.g., '1D'")

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    idx = pd.DatetimeIndex(df.index)

    if len(idx) < 2:
        raise ValueError("Dataset must contain at least two timestamps to infer frequency")

    idx_sorted = idx.sort_values()
    freq = pd.infer_freq(idx_sorted)
    if freq:
        offset = to_offset(freq)
        try:
            base_delta = pd.Timedelta(offset)
        except (TypeError, ValueError):
            if getattr(offset, "delta", None) is not None:
                base_delta = pd.Timedelta(offset.delta)
            elif getattr(offset, "nanos", None):
                base_delta = pd.Timedelta(offset.nanos, unit="ns")
            else:
                raise ValueError(f"Unable to convert inferred frequency {freq!r} to a timedelta") from None
    else:
        diffs = idx_sorted[1:] - idx_sorted[:-1]
        diffs = diffs[diffs > pd.Timedelta(0)]
        if len(diffs) == 0:
            raise ValueError(
                "Unable to infer dataset frequency from index; ensure timestamps are monotonic and non-duplicated"
            )
        base_delta = diffs.median()

    if base_delta <= pd.Timedelta(0):
        raise ValueError("Unable to infer a positive base frequency from dataset index")

    steps_ratio = horizon_n / base_delta
    steps = int(round(steps_ratio))
    if steps <= 0 or not math.isclose(steps_ratio, steps, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(
            f"Horizon {horizon} is not an integer multiple of inferred base frequency ({base_delta})"
        )

    y = df[y_col].shift(-steps)

    # Feature set: lagged chem + lagged futures + rolling stats
    X = pd.DataFrame(index=df.index)

    # Lags for chem target and all futures returns
    for L in range(1, lags + 1):
        X[f"lag{L}_{y_col}"] = df[y_col].shift(L)
    fut_cols = [c for c in df.columns if c.startswith("fut_") and not c.startswith("lvl_")]
    for fut in fut_cols:
        for L in range(0, lags + 1):
            X[f"lag{L}_{fut}"] = df[fut].shift(L)
        # rolling mean/vol
        X[f"roll_mean_{fut}"] = df[fut].rolling(lags).mean()
        X[f"roll_std_{fut}"] = df[fut].rolling(lags).std()

    # Drop rows with NA from lags/rolls and future shift
    XY = pd.concat([X, y.rename("y")], axis=1).dropna()

    return XY.drop(columns=["y"]), XY["y"]


def train_streaming(args: TrainArgs):
    _ensure_outdir(args.outdir)
    df = pd.read_parquet(args.dataset)

    X, y = _build_features(df, args.symbol, args.lags, args.horizon)

    # If a prior model exists, load it and continue training; else build fresh pipeline
    if os.path.exists(args.model):
        pipe: Pipeline = joblib.load(args.model)
        print("↻ Loaded existing model; continuing training…")
    else:
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("sgd", SGDRegressor(loss="huber", penalty="l2", alpha=1e-4, learning_rate="invscaling", eta0=0.01, random_state=42))
        ])

    # Warm start on first chunk, then partial_fit in a streaming fashion
    n = len(X)
    chunk = max(args.step, 200)
    i0 = 0
    if hasattr(pipe.named_steps["sgd"], "partial_fit"):
        # Initial fit on first chunk
        pipe.fit(X.iloc[i0:i0+chunk], y.iloc[i0:i0+chunk])
        i0 += chunk
        while i0 < n:
            i1 = min(i0 + args.step, n)
            pipe.partial_fit(X.iloc[i0:i1], y.iloc[i0:i1])
            i0 = i1
    else:
        pipe.fit(X, y)

    # Evaluate out‑of‑sample with last 20% as test
    split = int(0.8 * n)
    yhat = pipe.predict(X.iloc[split:])
    metrics = {
        "r2": float(r2_score(y.iloc[split:], yhat)),
        "mae": float(mean_absolute_error(y.iloc[split:], yhat)),
        "train_rows": int(split),
        "test_rows": int(n - split)
    }
    with open(os.path.join(args.outdir, f"metrics_{args.symbol}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(pipe, args.model)
    print(f"✅ Trained model saved to {args.model}")
    print("📊 Metrics:", metrics)


# ---------------------------- 4) PREDICT ----------------------------

@dataclass
class PredictArgs:
    dataset: str
    symbol: str
    model: str


def predict_next(args: PredictArgs):
    df = pd.read_parquet(args.dataset)
    pipe: Pipeline = joblib.load(args.model)
    X, y = _build_features(df, args.symbol, lags=5, horizon="1D")  # lags/horizon must match training
    latest_x = X.iloc[[-1]]
    pred = float(pipe.predict(latest_x)[0])
    print(json.dumps({"symbol": args.symbol, "predicted_next_return": pred}))


# ---------------------------- CLI ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Chemical ↔ Futures correlation + online ML")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prep", help="Build aligned dataset from CSVs")
    sp.add_argument("--chem", required=True)
    sp.add_argument("--fut", required=True)
    sp.add_argument("--outdir", required=True)
    sp.add_argument("--freq", default="B")
    sp.add_argument("--method", choices=["ffill","bfill"], default="ffill")

    sc = sub.add_parser("corr", help="Run correlation/causality suite")
    sc.add_argument("--dataset", required=True)
    sc.add_argument("--outdir", required=True)
    sc.add_argument("--window", type=int, default=60)
    sc.add_argument("--max-lag", type=int, default=10)

    st = sub.add_parser("train", help="Train or continue training an online model for a chemical symbol")
    st.add_argument("--dataset", required=True)
    st.add_argument("--symbol", required=True)
    st.add_argument("--horizon", default="1D")
    st.add_argument("--lags", type=int, default=5)
    st.add_argument("--outdir", required=True)
    st.add_argument("--model", required=True)
    st.add_argument("--step", type=int, default=5)

    spred = sub.add_parser("predict", help="Predict the next return for a symbol")
    spred.add_argument("--dataset", required=True)
    spred.add_argument("--symbol", required=True)
    spred.add_argument("--model", required=True)

    args = ap.parse_args()

    if args.cmd == "prep":
        prep_dataset(PrepArgs(args.chem, args.fut, args.outdir, args.freq, args.method))
    elif args.cmd == "corr":
        corr_and_causality(CorrArgs(args.dataset, args.outdir, args.window, args.max_lag))
    elif args.cmd == "train":
        train_streaming(TrainArgs(args.dataset, args.symbol, args.horizon, args.lags, args.outdir, args.model, args.step))
    elif args.cmd == "predict":
        predict_next(PredictArgs(args.dataset, args.symbol, args.model))

if __name__ == "__main__":
    main()
