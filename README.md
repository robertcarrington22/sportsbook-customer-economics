# Sportsbook Customer Economics

CAC, lifetime value, payback, and early-value prediction for a US online sportsbook, built for the Analyst I, Customer Economics role at DraftKings.

**Read the report: [REPORT.pdf](REPORT.pdf)** (typeset) or [REPORT.md](REPORT.md) (on GitHub)

The player-level data is simulated, because no public sportsbook dataset has acquisition channels, CAC, or promo costs. Seasonality, hold, and hold volatility are calibrated to DraftKings' monthly filings with the New York State Gaming Commission. Everything else is an assumption in [`assumptions.toml`](assumptions.toml). None of it is DraftKings internal data.

## What's in it

- **Unit economics.** Revenue from GGR down to contribution, value concentration, payback and LTV/CAC by channel with bootstrap intervals, channel by offer returns, and break-even CAC by channel and state.
- **Retention.** Cohort retention, retention by channel, and six-month value by signup month.
- **Lifetime value.** 24- and 36-month LTV projections with a retention-decay model, backtested on held-out months.
- **Test design.** Sample sizes, winsorization, and CUPED variance reduction for the recommended experiments.
- **Early prediction.** Month-3 retention and 180-day value from the first 14 days, using XGBoost (including a two-part model for the heavy tail) on an out-of-time split, compared with TabPFN, a pretrained tabular foundation model.

## Layout

```
assumptions.toml       every input to the simulation
real_data/             DraftKings' NY monthly filings and the parser that reads them
simulate.py            40,000 simulated first-time depositors and their daily betting
sql/                   DuckDB queries: player-month table, retention, LTV, channel/offer/state cuts, features
analyze.py             runs sql/ in order, adds bootstrap intervals, writes outputs/
model.py               XGBoost and logistic regression
model_tabpfn.py        TabPFN comparison (optional; 15 to 40 minutes on CPU, then cached)
ltv_forecast.py        projects LTV to 36 months and backtests the method
experiment_design.py   sample sizes and CUPED for the recommended tests
build_report.py        writes REPORT.md and figures/
make_pdf.py            typesets REPORT.md as REPORT.pdf (Windows, uses Edge)
run.py                 simulate, analyze, model, LTV projection, test design, report
```

## Running it

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python run.py
```

`run.py` takes under a minute and is deterministic. To refresh the TabPFN comparison, run `model_tabpfn.py` (its predictions are cached after the first run) and then `build_report.py`.

TabPFN v2 weights are released by Prior Labs under the Prior Labs License (Apache 2.0 with an attribution requirement). Built with TabPFN.
