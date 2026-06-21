"""
Temporal splitting + baselines  (Phase 2)
=========================================

This module is the SCIENTIFIC FOUNDATION of the project. If the split logic is
wrong, every downstream number is invalid, so the leakage rules live here and
the verifier will lean on them.

Two things are provided:

  1. Time-ordered block split: train < validation < test, by calendar date.
     The agent optimizes against VALIDATION. TEST is touched exactly once, at
     the very end, by you -- never inside the loop.

  2. Purged + embargoed walk-forward CV *within* training, for the executor to
     estimate a correction model's skill honestly during iteration.

  3. The three baselines every corrected forecast must beat:
        - raw forecast      (did correction add anything?)
        - persistence       (did we beat 'value N hours ago'?)
        - classical NWP     (placeholder until you wire IFS from WeatherBench2)

------------------------------------------------------------------------------
WHY PURGE + EMBARGO
------------------------------------------------------------------------------
A trailing-window feature (e.g. precip summed over the last 48h) computed for a
row at time T draws on data in [T-48h, T]. If a CV validation fold starts right
where the training fold ends, those windows straddle the boundary and the model
effectively sees validation-adjacent information during training -> optimistic,
leaked skill. We drop an EMBARGO gap (>= max feature lookback) between folds so
no window can straddle. This is the standard purge/embargo from financial ML,
applied to weather.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Iterator

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Block split
# ---------------------------------------------------------------------------
def assign_blocks(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add a 'block' column in {'train','val','test',''} by calendar date.

    Rows outside all three ranges get '' and should be ignored. Returns a copy.
    """
    def _ts(d: date, end: bool = False) -> pd.Timestamp:
        h = 23 if end else 0
        return pd.Timestamp(datetime(d.year, d.month, d.day, h,
                                     tzinfo=timezone.utc))

    vt = df["valid_time"]
    block = np.full(len(df), "", dtype=object)
    block[(vt >= _ts(cfg.train_start)) & (vt <= _ts(cfg.train_end, True))] = "train"
    block[(vt >= _ts(cfg.val_start)) & (vt <= _ts(cfg.val_end, True))] = "val"
    block[(vt >= _ts(cfg.test_start)) & (vt <= _ts(cfg.test_end, True))] = "test"
    out = df.copy()
    out["block"] = block
    # sanity: blocks must be calendar-ordered and non-overlapping
    assert cfg.train_end < cfg.val_start < cfg.val_end < cfg.test_start, \
        "block date ranges must be strictly time-ordered and non-overlapping"
    return out


def purged_walk_forward(
    train_df: pd.DataFrame, cfg, n_folds: int = 4
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) positional indices over the TRAIN block only.

    Expanding-window walk-forward: fold k trains on the earliest k slices and
    validates on the (k+1)-th, with an embargo gap removed from the END of each
    training window. Indances are positional into a TIME-SORTED train_df.

    Use this inside the executor to score a candidate correction during the
    agent loop. Never let it see val/test blocks.
    """
    d = train_df.sort_values("valid_time").reset_index(drop=True)
    times = d["valid_time"].values
    n = len(d)
    # unique ordered timestamps -> split the TIME axis, not the row axis, so
    # multiple stations sharing a timestamp stay together.
    uniq = np.array(sorted(pd.unique(times)))
    bounds = np.linspace(0, len(uniq), n_folds + 2, dtype=int)
    embargo = np.timedelta64(cfg.embargo_hours, "h")

    for k in range(1, n_folds + 1):
        tr_end_time = uniq[bounds[k] - 1]
        val_lo_time = uniq[bounds[k]]
        val_hi_time = uniq[bounds[k + 1] - 1]
        # embargo: training rows must end at least `embargo` before val starts
        tr_mask = times <= (val_lo_time - embargo)
        val_mask = (times >= val_lo_time) & (times <= val_hi_time)
        tr_idx = np.where(tr_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(tr_idx) == 0 or len(val_idx) == 0:
            continue
        yield tr_idx, val_idx


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mae(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - obs)))


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def skill_vs(pred_err: float, baseline_err: float) -> float:
    """Percent error reduction of pred over a baseline (positive = better)."""
    if baseline_err == 0:
        return float("nan")
    return 100.0 * (baseline_err - pred_err) / baseline_err


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def baseline_raw(df: pd.DataFrame, cfg) -> dict:
    """Error of the uncorrected frozen forecast."""
    f = df[f"fcst_{cfg.target_variable}"].values
    o = df[f"obs_{cfg.target_variable}"].values
    return {"name": "raw_forecast", "mae": mae(f, o), "rmse": rmse(f, o)}


def baseline_persistence(df: pd.DataFrame, cfg, lag_steps: int = 4) -> dict:
    """Error of 'observation lag_steps ago' as the forecast.

    lag_steps defaults to 4 = 24h at 6-hour cadence (yesterday-same-time), a
    strong, classic naive baseline. Computed per location to respect series
    boundaries. Rows without a valid lag are dropped from this baseline only.
    """
    ocol = f"obs_{cfg.target_variable}"
    d = df.sort_values(["location", "valid_time"]).copy()
    d["persist"] = d.groupby("location")[ocol].shift(lag_steps)
    valid = d.dropna(subset=["persist"])
    return {
        "name": f"persistence_{lag_steps*cfg.step_hours}h",
        "mae": mae(valid["persist"].values, valid[ocol].values),
        "rmse": rmse(valid["persist"].values, valid[ocol].values),
        "n": int(len(valid)),
    }


def baseline_nwp(df: pd.DataFrame, cfg) -> dict:
    """Classical physics-model (IFS HRES) error.

    PLACEHOLDER: in synthetic mode we don't have a separate physics forecast,
    so this returns NaN. On your machine, add an 'fcst_ifs_<var>' column from
    WeatherBench2 IFS-HRES in the data layer and compute its error here. This
    is the third baseline that contextualizes against operational practice.
    """
    col = f"fcst_ifs_{cfg.target_variable}"
    if col not in df.columns:
        return {"name": "ifs_hres", "mae": float("nan"),
                "rmse": float("nan"), "note": "not available in synthetic mode"}
    o = df[f"obs_{cfg.target_variable}"].values
    return {"name": "ifs_hres", "mae": mae(df[col].values, o),
            "rmse": rmse(df[col].values, o)}


def all_baselines(df_block: pd.DataFrame, cfg) -> list[dict]:
    """Compute the full baseline panel on a given block (usually val or test)."""
    return [
        baseline_raw(df_block, cfg),
        baseline_persistence(df_block, cfg),
        baseline_nwp(df_block, cfg),
    ]


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config.config import Config
    from src.data_layer import load_aligned_table

    cfg = Config()
    df = load_aligned_table(cfg)
    df = assign_blocks(df, cfg)
    print("block counts:\n", df["block"].value_counts(), "\n")

    for blk in ("val", "test"):
        sub = df[df["block"] == blk]
        print(f"--- baselines on {blk} ({len(sub):,} rows) ---")
        for b in all_baselines(sub, cfg):
            print(f"  {b['name']:>22}  MAE={b['mae']:.3f}  RMSE={b['rmse']:.3f}")
        print()

    # show the purged CV folds on the train block
    tr = df[df["block"] == "train"]
    print("--- purged walk-forward folds (train block) ---")
    for i, (tri, vli) in enumerate(purged_walk_forward(tr, cfg)):
        print(f"  fold {i}: train_rows={len(tri):,}  val_rows={len(vli):,}")
