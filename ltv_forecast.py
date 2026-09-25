"""Project lifetime value beyond the observed window, and backtest the method.

Model, per acquisition channel:
  - Retention: the share of FTDs who bet in life month t follows a
    shifted-beta-geometric (sBG) curve, S(t) = B(a, b + t) / B(a, b). Two
    parameters, fit by least squares to the observed monthly active rates.
    The sBG allows for heterogeneous churn across players, which gives the
    long, slowly decaying tail that a single churn rate cannot.
  - Value: theoretical contribution per active player (expected rather than
    realized GGR, so bet-outcome luck in the fit window does not carry into the
    projection), held at its average over the last three fitted months.
  - Projected contribution in month t = S(t) x value per active player.

Backtests, on data the fit never saw:
  A. fit months 1-6, predict cumulative 12-month contribution
  B. fit months 1-12, predict cumulative 18-month contribution (older cohorts)
Compared with two naive baselines: stop counting at the end of the fit window,
and carry the last-three-month run rate forward with no decay.

Then fit months 1-12 and project 24- and 36-month LTV per channel, with
bootstrap intervals. Writes outputs/ltv_forecast.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln

import analyze

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
N_BOOT = 300


def load_panel(min_months: int) -> dict[str, dict]:
    """Per-channel matrices of monthly active flags and contribution for players with >= min_months observed."""
    con = analyze.connect()
    for stmt in analyze.statements((analyze.SQL / "01_player_month.sql").read_text()):
        con.execute(stmt)
    df = con.execute(f"""
        SELECT player_id, channel, cac, life_month,
               CAST(active_days > 0 AS INTEGER) AS active, theo_contribution AS contribution
        FROM player_month
        WHERE last_full_month >= {min_months - 1} AND life_month < {min_months}
        ORDER BY player_id, life_month
    """).df()
    out = {}
    for ch, g in df.groupby("channel"):
        A = g.pivot(index="player_id", columns="life_month", values="active").to_numpy(float)
        V = g.pivot(index="player_id", columns="life_month", values="contribution").to_numpy(float)
        cac = g.groupby("player_id").cac.first().to_numpy(float)
        out[ch] = {"A": A, "V": V, "cac": cac}
    return out


def sbg_survival(a: float, b: float, t: np.ndarray) -> np.ndarray:
    return np.exp(betaln(a, b + t) - betaln(a, b))


def fit_sbg(rate: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    def loss(p):
        a, b = np.exp(p)
        return float(np.sum((sbg_survival(a, b, t) - rate) ** 2))
    res = minimize(loss, x0=np.log([1.0, 2.0]), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000})
    a, b = np.exp(res.x)
    return float(a), float(b)


def project(A: np.ndarray, V: np.ndarray, t_fit: int, horizon: int) -> dict:
    """Fit on months [0, t_fit) and return projected mean cumulative contribution per FTD through `horizon` months."""
    act = A[:, :t_fit].mean(axis=0)
    t = np.arange(1, t_fit)
    a, b = fit_sbg(act[1:], t)
    val_per_active = V[:, :t_fit].sum(axis=0) / np.maximum(A[:, :t_fit].sum(axis=0), 1)
    v_bar = float(val_per_active[t_fit - 3:t_fit].mean())
    observed = V[:, :t_fit].mean(axis=0)
    future_t = np.arange(t_fit, horizon)
    future = sbg_survival(a, b, future_t) * v_bar
    monthly = np.concatenate([observed, future])
    return {
        "a": a, "b": b, "v_bar": v_bar,
        "cum": float(monthly.sum()),
        "monthly": monthly,
        "stop": float(observed.sum()),
        "run_rate": float(observed.sum() + observed[t_fit - 3:t_fit].mean() * (horizon - t_fit)),
    }


def backtest(panel: dict, t_fit: int, horizon: int) -> dict:
    rows, tot = [], {"actual": 0.0, "sbg": 0.0, "stop": 0.0, "run_rate": 0.0, "n": 0}
    for ch, d in panel.items():
        p = project(d["A"], d["V"], t_fit, horizon)
        actual = float(d["V"][:, :horizon].sum(axis=1).mean())
        n = len(d["A"])
        rows.append({
            "channel": ch, "ftds": n, "actual": round(actual, 1), "sbg": round(p["cum"], 1),
            "stop": round(p["stop"], 1), "run_rate": round(p["run_rate"], 1),
            "err_sbg": round(p["cum"] / actual - 1, 4), "err_stop": round(p["stop"] / actual - 1, 4),
            "err_run_rate": round(p["run_rate"] / actual - 1, 4),
        })
        for k, val in (("actual", actual), ("sbg", p["cum"]), ("stop", p["stop"]), ("run_rate", p["run_rate"])):
            tot[k] += val * n
        tot["n"] += n
    overall = {k: tot[k] / tot["n"] for k in ("actual", "sbg", "stop", "run_rate")}
    mape = lambda key: float(np.mean([abs(r[key]) for r in rows]))  # noqa: E731
    return {
        "t_fit": t_fit, "horizon": horizon, "channels": rows,
        "overall": {k: round(v, 1) for k, v in overall.items()},
        "overall_err": {k: round(overall[k] / overall["actual"] - 1, 4) for k in ("sbg", "stop", "run_rate")},
        "mean_abs_channel_err": {k: round(mape(f"err_{k}"), 4) for k in ("sbg", "stop", "run_rate")},
    }


def main() -> None:
    p12, p18 = load_panel(12), load_panel(18)
    results = {"backtest_6_to_12": backtest(p12, 6, 12), "backtest_12_to_18": backtest(p18, 12, 18)}

    rng = np.random.default_rng(11)
    proj = []
    for ch, d in p12.items():
        A, V, cac = d["A"], d["V"], d["cac"]
        base = project(A, V, 12, 36)
        cum = np.cumsum(base["monthly"])
        boot24, boot36, bootr36, curves = [], [], [], []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(A), len(A))
            b = project(A[idx], V[idx], 12, 36)
            c = np.cumsum(b["monthly"])
            boot24.append(c[23]); boot36.append(c[35]); bootr36.append(c[35] / cac[idx].mean()); curves.append(c)
        pay = int(np.argmax(cum >= cac.mean())) if (cum >= cac.mean()).any() else -1
        proj.append({
            "channel": ch, "ftds": len(A), "cac": round(float(cac.mean()), 1),
            "a": round(base["a"], 4), "b": round(base["b"], 4), "value_per_active": round(base["v_bar"], 2),
            "ltv_12": round(float(cum[11]), 1), "ltv_24": round(float(cum[23]), 1), "ltv_36": round(float(cum[35]), 1),
            "ltv_24_lo": round(float(np.quantile(boot24, 0.025)), 1), "ltv_24_hi": round(float(np.quantile(boot24, 0.975)), 1),
            "ltv_36_lo": round(float(np.quantile(boot36, 0.025)), 1), "ltv_36_hi": round(float(np.quantile(boot36, 0.975)), 1),
            "ratio_36": round(float(cum[35] / cac.mean()), 2),
            "ratio_36_lo": round(float(np.quantile(bootr36, 0.025)), 2), "ratio_36_hi": round(float(np.quantile(bootr36, 0.975)), 2),
            "survival_36": round(float(sbg_survival(base["a"], base["b"], np.array([35]))[0]), 4),
            "payback_month": pay,
            "cum_curve": [round(float(x), 1) for x in cum],
            "cum_lo": [round(float(x), 1) for x in np.quantile(np.array(curves), 0.025, axis=0)],
            "cum_hi": [round(float(x), 1) for x in np.quantile(np.array(curves), 0.975, axis=0)],
            "actual_cum": [round(float(x), 1) for x in np.cumsum(V.mean(axis=0))],
        })
    results["projection"] = sorted(proj, key=lambda r: -r["ratio_36"])
    (OUT / "ltv_forecast.json").write_text(json.dumps(results, indent=2))

    for k in ("backtest_6_to_12", "backtest_12_to_18"):
        bt = results[k]
        print(f"{k}: overall error sBG {bt['overall_err']['sbg']:+.1%}, stop {bt['overall_err']['stop']:+.1%}, "
              f"run-rate {bt['overall_err']['run_rate']:+.1%} | mean abs channel error sBG {bt['mean_abs_channel_err']['sbg']:.1%}")
    for r in results["projection"]:
        print(f"{r['channel']:<12} LTV12 {r['ltv_12']:>7.0f}  LTV24 {r['ltv_24']:>7.0f} [{r['ltv_24_lo']:.0f}, {r['ltv_24_hi']:.0f}]  "
              f"LTV36 {r['ltv_36']:>7.0f}  36m LTV/CAC {r['ratio_36']:.2f} [{r['ratio_36_lo']:.2f}, {r['ratio_36_hi']:.2f}]")


if __name__ == "__main__":
    main()
