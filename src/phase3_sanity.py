"""
Phase 3 sanity check  --  scripted (NON-agentic) bias correction
================================================================

This is your DAY-ONE INSURANCE POLICY. It proves, before any agent or any real
data exists, that the pipeline can actually reduce forecast error on a strictly
held-out, time-ordered TEST block. If this beats the baselines, a complete,
presentable project already exists.

It deliberately uses only a handful of hand-written features and two model
classes (ridge, gradient boosting). No LLM. The agent (Phases 5-6) will later
PROPOSE features/models that flow through this exact train->predict->score path.

Run:  python src/phase3_sanity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import Config
from src.data_layer import load_aligned_table
from src.splits_baselines import assign_blocks, all_baselines, mae, rmse, skill_vs


def _make_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Hand-written, FORECAST-TIME-SAFE features.

    Every feature is a pure function of info available at/before valid_time.
    The lag features look strictly BACKWARD (groupby location, shift > 0).
    These same operations are what the feature builder will later expose to the
    agent as composable primitives.
    """
    var = cfg.target_variable
    d = df.sort_values(["location", "valid_time"]).copy()

    # diurnal & seasonal encodings (known at forecast time -- pure calendar)
    hod = d["valid_time"].dt.hour
    doy = d["valid_time"].dt.dayofyear
    d["hod_sin"] = np.sin(2 * np.pi * hod / 24)
    d["hod_cos"] = np.cos(2 * np.pi * hod / 24)
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # the raw forecast itself is the most important feature
    d["fcst"] = d[f"fcst_{var}"]

    # backward lag of the forecast (1 step = 6h) -- safe, available at fcst time
    d["fcst_lag1"] = d.groupby("location")[f"fcst_{var}"].shift(1)

    # interaction that should help recover the synthetic winter-morning bias
    d["winter_morning"] = d["doy_cos"] * d["hod_cos"]

    feature_cols = ["fcst", "fcst_lag1", "hod_sin", "hod_cos",
                    "doy_sin", "doy_cos", "winter_morning"]
    d = d.dropna(subset=feature_cols)
    return d, feature_cols


def run():
    cfg = Config()
    df = load_aligned_table(cfg)
    df = assign_blocks(df, cfg)
    var = cfg.target_variable

    feat, cols = _make_features(df, cfg)
    tr = feat[feat["block"] == "train"]
    te = feat[feat["block"] == "test"]

    Xtr, ytr = tr[cols].values, tr[f"obs_{var}"].values
    Xte, yte = te[cols].values, te[f"obs_{var}"].values

    print("=== baselines on TEST ===")
    base = {b["name"]: b for b in all_baselines(te, cfg)}
    for name, b in base.items():
        print(f"  {name:>22}  MAE={b['mae']:.3f}  RMSE={b['rmse']:.3f}")
    raw_mae = base["raw_forecast"]["mae"]

    print("\n=== scripted correction models on TEST ===")
    for label, model in [
        ("ridge", Ridge(alpha=1.0)),
        ("hist_gbr", HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=cfg.seed)),
    ]:
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        m, r = mae(pred, yte), rmse(pred, yte)
        sk = skill_vs(m, raw_mae)
        verdict = "BEATS raw" if m < raw_mae else "worse than raw"
        print(f"  {label:>10}  MAE={m:.3f}  RMSE={r:.3f}  "
              f"skill_vs_raw={sk:+.1f}%   [{verdict}]")

    print("\nIf a model beats raw_forecast here, the pipeline works end-to-end "
          "and you have a presentable result before building the agent.")


if __name__ == "__main__":
    run()
