"""
Data layer for the NY bias-correction agent.
===========================================

Responsibilities (and NOTHING else):
  1. Acquire frozen-forecast data and ground-truth observations.
  2. Subset to the NY bounding box AT READ TIME (never load a global array).
  3. Align forecast and truth on `valid_time` into one tidy table.
  4. Cache aggressively to local disk so the agent loop is insulated from
     network latency.

Explicit non-responsibilities: this module does NOT compute derived/lagged
features. It emits the RAW aligned table (forecast + truth + raw context
columns). Feature derivation lives in the feature builder so the verifier can
police leakage at a single chokepoint.

------------------------------------------------------------------------------
THE ALIGNED TABLE CONTRACT  (the single output of this layer)
------------------------------------------------------------------------------
A pandas DataFrame, one row per (valid_time, location), with columns:

    valid_time        : tz-aware UTC timestamp the forecast/obs is valid for
    location          : station id (e.g. 'KALB') or 'cell_<lat>_<lon>'
    lat, lon          : location coordinates (degrees, -180..180 lon)
    fcst_<var>        : the FROZEN foundation-model forecast value (the thing
                        we correct).  e.g. fcst_t2m
    obs_<var>         : the observed truth value (the target).  e.g. obs_t2m
    <context cols>    : RAW context available at/after valid_time that features
                        will later be DERIVED from, e.g. precip_6h, wind_10m,
                        cloud_frac. These are raw; turning them into forecast-
                        time-safe features is the feature builder's job.

Invariants guaranteed here:
  * valid_time is UTC, tz-aware, on the configured step (e.g. 6-hourly).
  * No NaN in fcst_<var> or obs_<var> (rows with missing pairs are dropped and
    the drop count is reported).
  * Longitudes are in -180..180.
------------------------------------------------------------------------------
"""
from __future__ import annotations

import os
import hashlib
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

# xarray is only needed for the REAL path. Import lazily so the synthetic path
# (and CI on a machine without xarray) still works.
try:  # pragma: no cover - exercised only in real mode
    import xarray as xr
except Exception:  # noqa: BLE001
    xr = None  # type: ignore


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_aligned_table(cfg, force_rebuild: bool = False) -> pd.DataFrame:
    """Return the aligned table for the full configured date span.

    Reads from cache when available unless force_rebuild=True. Dispatches to the
    synthetic or real builder based on cfg.data_mode.
    """
    cache_path = _cache_path(cfg)
    if (not force_rebuild) and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        _validate_aligned_table(df, cfg)
        print(f"[data] loaded aligned table from cache: {cache_path} "
              f"({len(df):,} rows)")
        return df

    if cfg.data_mode == "synthetic":
        df = _build_synthetic(cfg)
    else:
        df = _build_real(cfg)

    _validate_aligned_table(df, cfg)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"[data] built & cached aligned table: {cache_path} ({len(df):,} rows)")
    return df


# ---------------------------------------------------------------------------
# Synthetic builder  --  lets you run the WHOLE pipeline offline (Phase 0/1)
# ---------------------------------------------------------------------------
def _build_synthetic(cfg) -> pd.DataFrame:
    """Generate fake-but-structurally-realistic data.

    The point is NOT meteorological realism. The point is to produce a table
    with the right schema, a real seasonal+diurnal signal, a KNOWN systematic
    bias, and genuine lagged structure -- so that:
      * the alignment/caching/splitting code can be exercised end to end, and
      * bias correction has something real to recover (so you can sanity-check
        that your pipeline actually reduces error before touching real data).

    Construction (so you know exactly what's recoverable):
      truth(t) =  seasonal(t) + diurnal(t) + AR(1) weather noise + station offset
      forecast = truth - BIAS(station, season, hour) + small fcst noise
    so a perfect correction model would learn BIAS(...) and the AR(1) gives a
    real (but bounded) benefit to lag features. A deliberately-wrong heuristic
    (e.g. 7-day-lag precip -> humidity) will find NO signal, which is what you
    want for the verifier demo.
    """
    rng = np.random.default_rng(cfg.seed)
    times = _time_index(cfg.train_start, cfg.test_end, cfg.step_hours)
    n = len(times)

    # locations: reuse the configured station ids, give them NY-ish coords
    coords = {
        "KNYC": (40.78, -73.97), "KALB": (42.75, -73.80),
        "KBUF": (42.94, -78.74), "KSYR": (43.11, -76.10),
        "KMSS": (44.94, -74.85),
    }
    stations = [s for s in cfg.stations if s in coords] or list(coords)[:3]

    # time-of-year in [0,1) and hour-of-day for seasonal/diurnal signals
    doy = np.array([t.timetuple().tm_yday for t in times], dtype=float)
    hod = np.array([t.hour for t in times], dtype=float)
    seasonal = 12.0 * np.cos(2 * np.pi * (doy - 200) / 365.25)   # warm ~ summer
    diurnal = 5.0 * np.cos(2 * np.pi * (hod - 14) / 24.0)        # peak ~14:00

    frames = []
    for s in stations:
        lat, lon = coords[s]
        # AR(1) "weather" anomaly -- this is the genuine lagged structure
        ar = np.zeros(n)
        phi = 0.85
        for i in range(1, n):
            ar[i] = phi * ar[i - 1] + rng.normal(0, 2.0)
        station_offset = (lat - 42.0) * -0.7   # cooler to the north
        truth = seasonal + diurnal + ar + station_offset

        # KNOWN systematic bias the correction model should recover:
        # forecast runs warm in winter mornings, cool in summer afternoons.
        winterness = np.cos(2 * np.pi * (doy - 15) / 365.25)     # +1 mid-Jan
        morningness = np.cos(2 * np.pi * (hod - 7) / 24.0)       # +1 at 07:00
        bias = 1.5 * winterness * morningness                    # deg C
        forecast = truth - bias + rng.normal(0, 0.8, n)

        # raw context columns (features will be DERIVED from these later)
        precip_6h = np.clip(rng.gamma(0.3, 1.2, n) - 0.2, 0, None)
        wind_10m = np.clip(3 + ar * 0.1 + rng.normal(0, 1.5, n), 0, None)
        cloud_frac = np.clip(0.4 + 0.3 * np.sin(2 * np.pi * doy / 365.25)
                             + rng.normal(0, 0.2, n), 0, 1)

        frames.append(pd.DataFrame({
            "valid_time": times,
            "location": s,
            "lat": lat, "lon": lon,
            f"fcst_{cfg.target_variable}": forecast,
            f"obs_{cfg.target_variable}": truth,
            "precip_6h": precip_6h,
            "wind_10m": wind_10m,
            "cloud_frac": cloud_frac,
        }))

    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["valid_time", "location"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Real builder  --  STUBS with precise contracts. Fill in on your machine.
# ---------------------------------------------------------------------------
def _build_real(cfg) -> pd.DataFrame:
    """Assemble the aligned table from real sources.

    This is intentionally a thin orchestrator over two helpers you complete in
    Phase 0/1. Each helper has a strict output contract so that alignment is a
    trivial merge. DO the NY subsetting inside each helper, at read time.
    """
    fcst = _load_forecast_real(cfg)     # contract below
    truth = _load_truth_real(cfg)       # contract below

    # Align on the (valid_time, location) keys. Inner join => only rows where we
    # have BOTH a forecast and an observation survive. Report the attrition.
    merged = fcst.merge(truth, on=["valid_time", "location"], how="inner",
                        suffixes=("", "_dup"))
    dropped = len(fcst) - len(merged)
    print(f"[data] alignment: {len(merged):,} paired rows "
          f"({dropped:,} forecast rows had no matching obs and were dropped)")
    return merged.sort_values(["valid_time", "location"]).reset_index(drop=True)


def _load_forecast_real(cfg) -> pd.DataFrame:
    """CONTRACT -> DataFrame[valid_time, location, lat, lon, fcst_<var>, <context>]

    Implementation notes for your machine (do NOT run here; container is offline):
      * Open WeatherBench2 forecast Zarr lazily:
            ds = xr.open_zarr(WB2_URL, storage_options={'token': 'anon'})
      * SUBSET BEFORE COMPUTE. Use .sel with the NY bbox slice and the configured
        time range and lead/init you want, THEN .load(). Slicing a lazy Zarr is
        what keeps you inside 64 GB:
            ds_ny = ds.sel(latitude=slice(lat_max, lat_min),     # note order!
                           longitude=slice(lon_min % 360, lon_max % 360),
                           time=slice(train_start, test_end))
      * If truth_source == 'station', interpolate the gridded forecast to each
        station lat/lon (ds_ny.interp(latitude=..., longitude=...)) so the
        forecast 'location' keys match station ids. If 'era5', keep cells and
        name locations 'cell_<lat>_<lon>'.
      * Convert longitudes back to -180..180 for the output table.
      * Convert units to match truth (e.g. Kelvin -> Celsius for t2m).
      * Melt to long/tidy form with one row per (valid_time, location).
    """
    raise NotImplementedError(
        "Fill in _load_forecast_real on your machine. See docstring contract. "
        "Until then, run with cfg.data_mode='synthetic'."
    )


def _load_truth_real(cfg) -> pd.DataFrame:
    """CONTRACT -> DataFrame[valid_time, location, obs_<var>]

    Implementation notes:
      * station mode: pull ASOS obs (Iowa Environmental Mesonet, Synoptic, or
        Meteostat). Resample/snap to the 6-hour grid; a value must be the obs
        AT or nearest-within-tolerance to each valid_time. Drop stations/timestamps
        with no obs rather than imputing.
      * era5 mode: open ERA5 Zarr (WeatherBench2 mirror or CDS), subset NY bbox
        + time range lazily, .load(), melt to long form keyed by cell.
      * Units MUST match the forecast (Celsius for t2m). Make valid_time tz-aware
        UTC.
    """
    raise NotImplementedError(
        "Fill in _load_truth_real on your machine. See docstring contract. "
        "Until then, run with cfg.data_mode='synthetic'."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _time_index(start, end, step_hours: int) -> list[datetime]:
    """Inclusive UTC timestamp list on the configured step."""
    t = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    end_dt = datetime(end.year, end.month, end.day, 23, 59, tzinfo=timezone.utc)
    out = []
    while t <= end_dt:
        out.append(t)
        t = t + timedelta(hours=step_hours)
    return out


def _cache_path(cfg) -> str:
    """Deterministic cache filename keyed by the parameters that affect data."""
    key = {
        "mode": cfg.data_mode, "var": cfg.target_variable,
        "model": cfg.forecast_model, "truth": cfg.truth_source,
        "stations": cfg.stations, "step": cfg.step_hours,
        "span": [cfg.train_start.isoformat(), cfg.test_end.isoformat()],
        "bbox": asdict(cfg.bbox), "seed": cfg.seed,
    }
    h = hashlib.md5(repr(sorted(key.items())).encode()).hexdigest()[:10]
    return os.path.join(cfg.cache_dir, f"aligned_{cfg.data_mode}_{h}.parquet")


def _validate_aligned_table(df: pd.DataFrame, cfg) -> None:
    """Fail loudly if the output contract is violated. Cheap insurance."""
    fcol = f"fcst_{cfg.target_variable}"
    ocol = f"obs_{cfg.target_variable}"
    required = {"valid_time", "location", "lat", "lon", fcol, ocol}
    missing = required - set(df.columns)
    assert not missing, f"aligned table missing columns: {missing}"
    assert df[fcol].notna().all(), "NaNs in forecast column"
    assert df[ocol].notna().all(), "NaNs in obs column"
    # valid_time must be tz-aware UTC
    assert pd.api.types.is_datetime64_any_dtype(df["valid_time"]), \
        "valid_time not datetime"
    # longitudes in -180..180
    assert df["lon"].between(-180, 180).all(), "longitudes not in -180..180"
    # no duplicate keys
    dup = df.duplicated(subset=["valid_time", "location"]).sum()
    assert dup == 0, f"{dup} duplicate (valid_time, location) rows"


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config.config import Config

    cfg = Config()  # synthetic by default
    df = load_aligned_table(cfg, force_rebuild=True)
    print(df.head(8).to_string())
    print("\nshape:", df.shape)
    print("locations:", sorted(df['location'].unique()))
    print("time span:", df['valid_time'].min(), "->", df['valid_time'].max())
