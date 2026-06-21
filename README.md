# NY Bias-Correction Agent — Scaffold (Phases 0–3)

An autonomous, bounded, observable agent that does Model Output Statistics-style
bias correction on a **frozen** foundation weather forecast (GraphCast / Pangu /
FGN) for **New York State**, with a leakage-aware verifier and a heuristic-as-
hypothesis layer.

This scaffold delivers **Phases 0–3** of the build plan: the data layer, the
temporal splits + baselines, and a scripted (non-agentic) correction that
already beats the baselines. The agent (Phases 4–7) bolts onto the
train→predict→score path these files establish.

> **It runs offline right now** in `synthetic` data mode, so you can exercise the
> entire pipeline before any real download succeeds. Switch to `real` once
> Phase 0 alignment works on your machine.

---

## Quick start

```bash
pip install pandas numpy pyarrow scikit-learn          # core
# (real mode also needs: xarray zarr gcsfs cdsapi  + a station-data client)

python config/config.py            # print the run config
python src/data_layer.py           # build & cache the aligned table (synthetic)
python src/splits_baselines.py     # blocks, baselines, purged CV folds
python src/phase3_sanity.py        # PROOF: correction beats baseline on TEST
```

Expected Phase 3 result (synthetic): gradient boosting cuts test MAE ~28% vs.
the raw forecast. That is your day-one "presentable result already exists" gate.

---

## File → plan mapping

| File | Phase | Role |
|------|-------|------|
| `config/config.py` | all | Single source of truth: scope, dates, leakage controls, agent bounds. Frozen dataclass; log it at the top of every run. |
| `src/data_layer.py` | 0–1 | Acquire → NY-subset (at read time) → align on `valid_time` → cache. Emits the **aligned table** (raw forecast + truth + raw context). Synthetic builder works now; real builders are stubbed with precise contracts. |
| `src/splits_baselines.py` | 2 | Time-ordered `train<val<test` blocks; **purged + embargoed** walk-forward CV; the three baselines (raw / persistence / NWP). The scientific foundation. |
| `src/phase3_sanity.py` | 3 | Scripted, no-LLM correction proving the pipeline reduces error on held-out TEST. Your insurance policy. |

---

## The load-bearing design decisions (don't drift from these)

1. **The foundation model is frozen.** You consume pre-computed forecasts
   (WeatherBench2 Zarr). You never run/fine-tune GraphCast/Pangu/FGN. This is
   why it fits on your hardware.
2. **Subset NY at READ time.** `data_layer` slices the lazy Zarr to the NY bbox
   before `.load()`. Never materialize a global array (that's the only way to
   blow past 64 GB).
3. **TEST is touched once.** The agent optimizes against VALIDATION; the TEST
   block is read exactly once, at the very end, by you. This is your defense
   against the agent overfitting the metric.
4. **One leakage chokepoint.** The data layer emits only RAW context. All
   derived/lagged features are built downstream so the verifier can police them
   in one place. Every feature must be a pure function of info available at or
   before `valid_time`.
5. **Bound everything.** Iterations, tokens, dollars, retries, feature count —
   all capped in `config.py`.

---

## What to do next, in order

**Phase 0 (your machine, time-box ~1–2h, HARD GATE).**
Fill in `_load_forecast_real` and `_load_truth_real` in `data_layer.py` just far
enough to align **one** NY slice. Follow the docstring contracts exactly. If
alignment doesn't work in the time box, fall back: `truth_source='era5'` instead
of station obs, or a simpler forecast source. **Do not build anything else until
one real aligned row exists.**

**Phase 1.** Generalize the spike to the full date span; confirm caching and that
you never load a global array. Set `data_mode='real'`.

**Phase 2.** Wire the IFS-HRES column so `baseline_nwp` returns a real number
(third baseline). Re-confirm baselines on real data.

**Phase 3.** Re-run `phase3_sanity.py` on real data. A model beating raw +
persistence + NWP on real TEST is a complete project.

**Phases 4–7 (next scaffold):** observability harness (6-layer JSONL logging) →
planner–executor–verifier loop over the Anthropic API → heuristic library +
Reflexion memory → final eval + poster. The executor reuses the exact
`_make_features` → fit → predict → score path from `phase3_sanity.py`.

---

## Synthetic data: what's recoverable (so you can trust your sanity checks)

The synthetic builder bakes in a **known** systematic bias (forecast runs warm on
winter mornings, cool on summer afternoons) plus AR(1) weather structure. So:
- A correct pipeline **should** recover most of that bias → big skill over raw.
- A deliberately-wrong heuristic (e.g. "rain last week → lower humidity") finds
  **no** signal → the verifier rejects it. That's your signature demo, and you
  can test it offline before real data exists.
