"""Design the tests that the report's recommendations call for.

Test A, welcome offer (new depositors, randomized at signup):
    no-sweat first bet vs bet $5 get $200, outcome = 180-day contribution.
    Only pre-signup information (channel, state, signup month) may be used as
    covariates. Early betting behavior is itself affected by the offer, so
    adjusting for it would bias the estimate.

Test B, retention promo (existing players, randomized at the start of month 4):
    outcome = contribution in months 4 to 9. Months 1 to 3 happened before
    randomization, so they are valid CUPED covariates.

For each test: required sample size per arm at 5% significance and 80% power
for a range of minimum detectable effects, raw, winsorized at the 99th
percentile, and covariate-adjusted. For test B, 1,000 random splits check the
variance reduction empirically. Writes outputs/experiment_design.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import analyze

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
Z = norm.ppf(0.975) + norm.ppf(0.80)
MDES = [10, 25, 50, 75, 100, 150, 200]


def n_per_arm(sd: float, mde: float) -> int:
    return int(np.ceil(2 * (Z * sd / mde) ** 2))


def r2_ols(y: np.ndarray, X: np.ndarray) -> float:
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    return float(1 - resid.var() / y.var())


def dummies(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return pd.get_dummies(df[cols].astype(str), drop_first=True).to_numpy(float)


def winsor(y: np.ndarray, q: float = 0.99) -> np.ndarray:
    return np.minimum(y, np.quantile(y, q))


def main() -> None:
    feats = pd.read_parquet(OUT / "features.parquet").sort_values("player_id", ignore_index=True)

    # ---- Test A: welcome offer ---------------------------------------------
    y = feats["contribution_180"].to_numpy(float)
    yw = winsor(y)
    pre = dummies(feats, ["channel", "state", "signup_month"])
    post = feats[["active_days_14", "bets_14", "handle_14", "ggr_14", "avg_stake_14"]].to_numpy(float)
    post = np.sign(post) * np.log1p(np.abs(post))
    r2_pre, r2_pre_w = r2_ols(y, pre), r2_ols(yw, pre)
    by_offer = feats.groupby("offer")["contribution_180"].mean()
    observed_gap = float(by_offer["no_sweat_1000"] - by_offer["bet5_get200"])

    sd, sd_w = float(y.std()), float(yw.std())
    sd_adj = sd_w * np.sqrt(1 - r2_pre_w)
    test_a = {
        "outcome": "contribution over days 0 to 179",
        "mean": round(float(y.mean()), 2), "sd": round(sd, 2), "sd_winsor": round(sd_w, 2),
        "winsor_cap": round(float(np.quantile(y, 0.99)), 2),
        "r2_pre_signup": round(r2_pre, 4), "r2_pre_signup_winsor": round(r2_pre_w, 4),
        "r2_post_treatment_if_misused": round(r2_ols(yw, np.column_stack([pre, post])), 4),
        "observed_gap_180": round(observed_gap, 2),
        "n_observed_gap": {"raw": n_per_arm(sd, observed_gap), "winsor": n_per_arm(sd_w, observed_gap),
                           "winsor_adjusted": n_per_arm(sd_adj, observed_gap)},
        "table": [{"mde": m, "raw": n_per_arm(sd, m), "winsor": n_per_arm(sd_w, m),
                   "winsor_adjusted": n_per_arm(sd_adj, m)} for m in MDES],
        "ftds_per_month_all_channels": int(round(len(feats) / 12)),
        "ftds_per_month_paid_social": int(round((feats.channel == "paid_social").sum() / 12)),
    }

    # ---- Test B: retention promo with CUPED ---------------------------------
    con = analyze.connect()
    for stmt in analyze.statements((analyze.SQL / "01_player_month.sql").read_text()):
        con.execute(stmt)
    df = con.execute("""
        SELECT player_id,
               sum(contribution) FILTER (WHERE life_month BETWEEN 3 AND 8) AS y,
               sum(contribution) FILTER (WHERE life_month <= 2)            AS pre_contrib,
               sum(handle)       FILTER (WHERE life_month <= 2)            AS pre_handle,
               sum(active_days)  FILTER (WHERE life_month <= 2)            AS pre_days,
               sum(active_days)  FILTER (WHERE life_month = 2)             AS m3_days
        FROM player_month
        WHERE last_full_month >= 8
        GROUP BY player_id
        HAVING sum(active_days) FILTER (WHERE life_month = 2) > 0
        ORDER BY player_id
    """).df()
    yb = winsor(df["y"].to_numpy(float))
    x = df["pre_contrib"].to_numpy(float)
    rho = float(np.corrcoef(yb, x)[0, 1])
    Xmulti = np.column_stack([np.sign(v) * np.log1p(np.abs(v)) for v in
                              (df.pre_contrib.to_numpy(float), df.pre_handle.to_numpy(float),
                               df.pre_days.to_numpy(float), df.m3_days.to_numpy(float))])
    r2_multi = r2_ols(yb, Xmulti)
    sd_b = float(yb.std())

    # Empirical check: random 50/50 splits, compare the spread of the plain
    # difference in means with the CUPED estimate.
    rng = np.random.default_rng(3)
    theta = float(np.cov(yb, x)[0, 1] / x.var(ddof=1))
    y_cuped = yb - theta * (x - x.mean())
    X1 = np.column_stack([np.ones(len(Xmulti)), Xmulti])
    beta, *_ = np.linalg.lstsq(X1, yb, rcond=None)
    y_reg = yb - (X1 @ beta - yb.mean())
    diffs = {"raw": [], "cuped": [], "regression": []}
    for _ in range(1000):
        t = rng.random(len(yb)) < 0.5
        for k, arr in (("raw", yb), ("cuped", y_cuped), ("regression", y_reg)):
            diffs[k].append(arr[t].mean() - arr[~t].mean())
    emp_sd = {k: float(np.std(v)) for k, v in diffs.items()}
    test_b = {
        "population": "players who bet in month 3, with 9 months observed",
        "n_players": int(len(df)), "outcome": "contribution in months 4 to 9 (winsorized at the 99th percentile)",
        "mean": round(float(yb.mean()), 2), "sd": round(sd_b, 2),
        "rho_pre_contrib": round(rho, 4), "var_reduction_cuped": round(rho ** 2, 4),
        "r2_multi": round(r2_multi, 4),
        "empirical_sd_of_estimate": {k: round(v, 3) for k, v in emp_sd.items()},
        "empirical_var_reduction": {k: round(1 - (emp_sd[k] / emp_sd["raw"]) ** 2, 4) for k in ("cuped", "regression")},
        "table": [{"mde": m, "raw": n_per_arm(sd_b, m), "cuped": n_per_arm(sd_b * np.sqrt(1 - rho ** 2), m),
                   "regression": n_per_arm(sd_b * np.sqrt(1 - r2_multi), m)} for m in MDES],
        "split_diffs": {k: [round(float(d), 3) for d in v] for k, v in diffs.items()},
    }

    (OUT / "experiment_design.json").write_text(json.dumps({"offer_test": test_a, "retention_test": test_b}, indent=2))
    print(json.dumps({k: v for k, v in test_a.items() if k != "table"}, indent=1))
    print("offer test n per arm:", test_a["table"])
    print(json.dumps({k: v for k, v in test_b.items() if k not in ("table", "split_diffs")}, indent=1))
    print("retention test n per arm:", test_b["table"])


if __name__ == "__main__":
    main()
