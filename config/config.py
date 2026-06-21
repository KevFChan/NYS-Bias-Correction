"""
Central configuration for the NY bias-correction agent.

Everything that parameterizes a run lives here so that runs are reproducible
and a single object can be logged at the top of every trajectory. Treat this
as the single source of truth; downstream modules should accept a Config
instance rather than reading globals.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Literal
import json


# --- Geographic scope: New York State bounding box (with a small buffer) -----
# NY spans roughly 40.5-45.0 N and 71.8-79.8 W. We pad outward so that any
# 0.25-degree grid cell touching the state is included. Latitude/longitude are
# in degrees; longitude here is in the -180..180 convention. If your forecast
# source uses 0..360 longitudes, convert at READ time inside the data layer,
# never after loading a global array.
@dataclass(frozen=True)
class BBox:
    lat_min: float = 40.3
    lat_max: float = 45.2
    lon_min: float = -80.0
    lon_max: float = -71.6


@dataclass(frozen=True)
class Config:
    # --- scope ---------------------------------------------------------------
    bbox: BBox = field(default_factory=BBox)
    target_variable: str = "t2m"          # 2 m temperature: the friendly first target
    # Date ranges define the three TIME-ORDERED blocks. They must be contiguous
    # and non-overlapping. train < validation < test, always in calendar order.
    train_start: date = date(2020, 1, 1)
    train_end: date = date(2021, 12, 31)
    val_start: date = date(2022, 1, 1)
    val_end: date = date(2022, 12, 31)
    test_start: date = date(2023, 1, 1)
    test_end: date = date(2023, 12, 31)

    # --- temporal resolution -------------------------------------------------
    step_hours: int = 6                   # 6-hour forecast cadence

    # --- truth source --------------------------------------------------------
    # 'station' is meteorologically preferable; 'era5' is the lower-friction
    # fallback if station alignment proves slow in Phase 0.
    truth_source: Literal["station", "era5"] = "station"
    # A small set of representative NY ASOS stations spanning climate zones:
    # NYC (coastal/urban), Albany (Hudson valley), Buffalo (lake-effect),
    # Syracuse (central/snowbelt), Massena (north country).
    stations: tuple[str, ...] = ("KNYC", "KALB", "KBUF", "KSYR", "KMSS")

    # --- forecast (frozen foundation model) source ---------------------------
    # Identifier used by the data layer to pick the WeatherBench2 path / variable
    # mapping. Kept abstract here so swapping GraphCast <-> Pangu is a one-liner.
    forecast_model: Literal["graphcast", "pangu", "ifs_hres"] = "graphcast"

    # --- leakage / cross-validation controls ---------------------------------
    # Embargo (in hours) dropped between any train fold and its validation fold
    # in purged CV. MUST be >= the longest feature lookback you allow, otherwise
    # a trailing-window feature can straddle the fold boundary and leak.
    embargo_hours: int = 72
    max_feature_lookback_hours: int = 48  # hard cap the verifier enforces

    # --- agent bounds (the deck's "bound everything" principle) --------------
    max_iterations: int = 30
    max_no_improve_steps: int = 6         # stop if no val gain for this many steps
    token_budget: int = 400_000           # hard cap across the whole run
    dollar_budget: float = 15.0           # belt-and-suspenders cost ceiling
    max_features: int = 25                # cap correction-model dimensionality

    # --- data mode -----------------------------------------------------------
    # 'synthetic' generates fake-but-realistic data so the FULL pipeline can be
    # exercised offline before any real download works. Switch to 'real' once
    # Phase 0 alignment succeeds on your machine.
    data_mode: Literal["synthetic", "real"] = "synthetic"

    # --- paths ---------------------------------------------------------------
    cache_dir: str = "data/cache"
    raw_dir: str = "data/raw"
    log_dir: str = "logs"

    # --- reproducibility -----------------------------------------------------
    seed: int = 13

    def to_json(self) -> str:
        d = asdict(self)
        # dates and tuples need to be made JSON-safe
        for k, v in list(d.items()):
            if isinstance(v, date):
                d[k] = v.isoformat()
        return json.dumps(d, indent=2, default=str)


# A module-level default instance for convenience in notebooks/spikes.
DEFAULT = Config()


if __name__ == "__main__":
    print(DEFAULT.to_json())
