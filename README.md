# 2026 House midterm forecast

This project is a district-level 2026 U.S. House forecast with:

- a full generic-ballot archive
- a daily state-space national filter
- a weak Trump-approval cross-check in the national layer
- structural district priors
- Monte Carlo chamber simulation
- a Streamlit UI built around auditability, not just toplines

## What changed in this build

This version fixes the main comparability problem in the prior trend view and expands the national polling layer.

### Main upgrades

- keeps the **current district and campaign inputs fixed** through the reconstructed daily odds line so the endpoint no longer jumps because the model definition changed
- adds **Trump approval** as an explicit national cross-check in the model rather than only a display metric
- writes a dedicated **Trump approval curve** with separate approve and disapprove lines plus 90% bands
- surfaces both the **raw filtered generic-ballot line** and the **approval-adjusted national House environment** used in simulation
- records endpoint-gap diagnostics in the audit output so any future discontinuity is visible immediately

## Files

- `app.py` — Streamlit app
- `src/data_sources.py` — source ingestion and seed/live loaders
- `src/model.py` — national filter, district priors, simulation engine, output writer
- `scripts/run_update.py` — batch runner
- `data/seed/generic_ballot_polls_master.csv` — bundled generic-ballot archive
- `data/seed/trump_approval_recent_polls.csv` — bundled recent Trump approval rows for seed mode
- `data/runtime/latest/summary.json` — latest topline output
- `data/runtime/latest/run_audit.json` — latest audit manifest
- `data/runtime/latest/trump_approval_curve.csv` — latest approval curve for the UI
- `data/history/forecast_history.csv` — daily comparable odds history

## Methodology in one pass

### 1. Generic-ballot archive

The main national input is a dated generic-ballot archive assembled from RealClearPolling / RealClearPolitics pages. In seed mode, the model reads the bundled archive directly. In live mode, it refreshes the current RealClearPolling average and appends newly published rows when possible.

### 2. Latent national House environment

Each generic-ballot poll row is treated as an observation of a latent national House environment. The model uses a local-level Gaussian filter:

- latent margin today = latent margin yesterday + process noise
- poll row = latent margin + observation noise

Observation uncertainty is larger when:

- the row is adults instead of LV
- the row lacks exact field dates
- the row lacks sample size or population metadata
- the row is flagged as partisan or internal

### 3. Trump approval as a weak cross-check

Trump approval is now explicitly in the model, but only as a weak prior on the national House environment. The model converts Trump net approval into an approval-implied House margin using a shallow slope, then combines that with the filtered generic ballot using a large prior SD.

That is deliberate. Approval and the generic ballot are strongly correlated, so letting approval hit the model too hard would double-count the same national mood. In this build, approval mostly matters when generic polling is sparse or unusually noisy.

### 4. Comparable daily trend line

The continuous odds line is **not** a historical archive of exact saved forecasts. It is a reconstructed daily curve. To make that line internally comparable, the project now holds today's campaign structure fixed through the full trend reconstruction:

- current open seats
- current expert ratings
- current finance inputs
- current approval cross-check

What changes day by day is the national generic-ballot state and any dated district polls. Exact saved runs are still preserved separately in `run_history.csv`.

### 5. District priors and chamber simulation

District forecasts start from:

- 2024 House results
- presidential vote by district when available
- persistence rules for incumbent, successor, and open seats
- optional ratings, district polling, and finance

The chamber forecast then runs Monte Carlo draws with:

- national uncertainty
- correlated state error
- district-specific uncertainty

Outputs include:

- GOP odds to hold the House
- expected GOP seats
- interval estimates
- full seat distribution
- district-level win probabilities and projected margins

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_update.py --seed-only --no-fec --no-district-polls
streamlit run app.py
```

## Notes on accuracy

This build is materially better than a thin recent average or a one-point saved-run chart, but it is still not a full multi-cycle calibration system. The strongest next upgrade would be cycle-by-cycle backtesting so pollster house effects, horizon error, process variance, and approval-to-House link strength can be estimated empirically rather than set conservatively.
