"""Daily day-14 scoring job.

Every day, score the first-time depositors who have just completed their first
14 days (days 0 to 13), assign each one an action, and check the incoming
players for drift against the training data.

    python score.py --as-of 2025-06-15

Outputs:
    outputs/scores/day14_scores_<date>.parquet   one row per scored FTD
    outputs/scores/drift_<date>.csv               PSI per feature, trailing 30 days

Actions (thresholds are illustrative and belong to the business owner):
    VIP review         predicted theo value in the training top decile, or 14-day handle of $5,000+
    Retention offer    probability of betting in month 3 below 40%, and predicted value above zero
    No action          everyone else

The feature table only uses days 0 to 13, so reading it for players whose 14th
day has passed is equivalent to computing features on the as-of date. In
production the same SQL (sql/08_early_features.sql) would run on the warehouse,
filtered to the day's cohort.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model import CATS, load

ROOT = Path(__file__).parent
OUT = ROOT / "outputs" / "scores"
MODELS = ROOT / "models" / "day14_models.joblib"
VIP_HANDLE = 5000
RETENTION_P = 0.40
PSI_ALERT = 0.2


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index on quantile bins of the training distribution."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(np.clip(expected, edges[0], edges[-1]), edges)[0] / len(expected)
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / max(len(actual), 1)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def cohort(df: pd.DataFrame, as_of: pd.Timestamp, days: int) -> pd.DataFrame:
    """FTDs whose 14th day (day 13) fell within the `days` days ending on as_of."""
    day14 = pd.to_datetime(df["signup_date"]) + pd.Timedelta(days=13)
    return df[(day14 <= as_of) & (day14 > as_of - pd.Timedelta(days=days))]


def main(as_of: str | None = None) -> dict:
    if as_of is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--as-of", default="2025-06-15")
        as_of = parser.parse_args().as_of
    as_of_ts = pd.Timestamp(as_of)
    bundle = joblib.load(MODELS)
    feats = bundle["features"]

    train, test = load()
    everyone = pd.concat([train, test], ignore_index=True)
    players = pd.read_parquet(ROOT / "data" / "players.parquet", columns=["player_id", "signup_date"])
    everyone = everyone.merge(players, on="player_id")

    today = cohort(everyone, as_of_ts, 1)
    X = today[feats]
    scores = pd.DataFrame({
        "player_id": today["player_id"].to_numpy(),
        "signup_date": pd.to_datetime(today["signup_date"]).dt.date.to_numpy(),
        "channel": today["channel"].astype(str).to_numpy(),
        "state": today["state"].astype(str).to_numpy(),
        "handle_14": today["handle_14"].round(2).to_numpy(),
        "p_active_m3": bundle["retention"].predict_proba(X)[:, 1].round(4) if len(X) else [],
        "predicted_theo_value_180": bundle["value"].predict(X).round(2) if len(X) else [],
    })
    vip = (scores.predicted_theo_value_180 >= bundle["train_value_p90"]) | (scores.handle_14 >= VIP_HANDLE)
    retain = (scores.p_active_m3 < RETENTION_P) & (scores.predicted_theo_value_180 > 0)
    scores["action"] = np.select([vip, retain], ["VIP review", "Retention offer"], "No action")
    scores["scored_on"] = as_of_ts.date()
    scores["model_trained_on"] = bundle["train_cohorts"]

    # Drift: compare the last 30 days of incoming FTDs with the training data.
    recent = cohort(everyone, as_of_ts, 30)
    train_x = pd.read_parquet(ROOT / "models" / "train_features.parquet")
    if len(recent) < 200:  # too few incoming players for a meaningful comparison
        drift = pd.DataFrame({"feature": [f for f in feats if f not in CATS], "psi": np.nan})
    else:
        drift = pd.DataFrame([
            {"feature": f, "psi": round(psi(train_x[f].to_numpy(float), recent[f].to_numpy(float)), 4)}
            for f in feats if f not in CATS
        ]).sort_values("psi", ascending=False)
    drift["alert"] = drift["psi"] > PSI_ALERT
    drift["n_recent"] = len(recent)

    OUT.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(OUT / f"day14_scores_{as_of_ts.date()}.parquet", index=False)
    drift.to_csv(OUT / f"drift_{as_of_ts.date()}.csv", index=False)
    summary = {
        "as_of": str(as_of_ts.date()), "scored": len(scores),
        "actions": scores["action"].value_counts().to_dict(),
        "drift_alerts": drift.loc[drift.alert, "feature"].tolist(), "n_recent": len(recent),
        "top_psi": drift.head(3)[["feature", "psi"]].to_dict("records"),
    }
    print(summary)
    return summary


if __name__ == "__main__":
    main()
