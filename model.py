"""Predict player value from the first 14 days.

Two questions a customer economics team gets asked:
  1. Which new depositors will still be betting in month 3?  (classifier)
  2. What will each new depositor be worth over 180 days?     (regressor)

Train on cohorts that signed up before May 2025, test on May to August 2025
cohorts. An out-of-time split is how these models would actually be used:
fit on the past, score new signups.

Writes outputs/model_results.json for the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
TEST_START = pd.Timestamp("2025-05-01")

CATS = ["channel", "offer", "state", "signup_month"]
NUMS = [
    "welcome_promo_cost",
    "active_days_14",
    "active_days_wk2",
    "bets_14",
    "handle_14",
    "ggr_14",
    "max_day_handle_14",
    "avg_stake_14",
    "days_idle_at_d14",
]
LABELS = {
    "welcome_promo_cost": "Welcome promo cost",
    "active_days_14": "Active days, first 14",
    "active_days_wk2": "Active days, week 2",
    "bets_14": "Bets, first 14 days",
    "handle_14": "Handle, first 14 days",
    "ggr_14": "GGR, first 14 days",
    "max_day_handle_14": "Biggest single-day handle",
    "avg_stake_14": "Average stake",
    "days_idle_at_d14": "Days idle at day 14",
    "channel": "Acquisition channel",
    "offer": "Welcome offer",
    "state": "State",
    "signup_month": "Signup month",
}


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Sort explicitly: DuckDB does not guarantee row order, and row order
    # decides which rows HistGradientBoosting holds out for early stopping.
    df = pd.read_parquet(OUT / "features.parquet").sort_values("player_id", ignore_index=True)
    df["cohort_month"] = pd.to_datetime(df["cohort_month"])
    for c in CATS:
        df[c] = df[c].astype("category")
    train = df[df["cohort_month"] < TEST_START].reset_index(drop=True)
    test = df[df["cohort_month"] >= TEST_START].reset_index(drop=True)
    return train, test


def decile_table(pred: np.ndarray, actual: np.ndarray) -> list[dict]:
    order = pd.qcut(pd.Series(pred).rank(method="first"), 10, labels=False)
    t = pd.DataFrame({"decile": 10 - order, "pred": pred, "actual": actual})
    g = t.groupby("decile").agg(predicted=("pred", "mean"), actual=("actual", "mean"), n=("actual", "size"))
    return [
        {"decile": int(d), "predicted": round(float(r.predicted), 2), "actual": round(float(r.actual), 2), "n": int(r.n)}
        for d, r in g.sort_index().iterrows()
    ]


def top_decile(score: np.ndarray, actual: np.ndarray) -> dict:
    cut = np.quantile(score, 0.9)
    top = score >= cut
    return {
        "mean_top_decile": round(float(actual[top].mean()), 2),
        "mean_all": round(float(actual.mean()), 2),
        "lift": round(float(actual[top].mean() / actual.mean()), 2),
        "share_of_total": round(float(actual[top].sum() / actual.sum()), 3),
    }


def main() -> None:
    train, test = load()
    X_tr, X_te = train[CATS + NUMS], test[CATS + NUMS]
    results: dict = {
        "split": {
            "train_cohorts": f"{train.cohort_month.min():%b %Y} to {train.cohort_month.max():%b %Y}",
            "test_cohorts": f"{test.cohort_month.min():%b %Y} to {test.cohort_month.max():%b %Y}",
            "n_train": len(train),
            "n_test": len(test),
        }
    }

    # ---- 1. Month-3 retention -------------------------------------------
    y_tr, y_te = train["active_m3"].to_numpy(), test["active_m3"].to_numpy()
    linear = make_pipeline(
        ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATS),
                ("num", make_pipeline(FunctionTransformer(signed_log), StandardScaler()), NUMS),
            ]
        ),
        LogisticRegression(max_iter=2000),
    )
    linear.fit(X_tr, y_tr)
    p_lin = linear.predict_proba(X_te)[:, 1]

    gbm_c = HistGradientBoostingClassifier(
        categorical_features="from_dtype", learning_rate=0.05, max_iter=400, early_stopping=True, random_state=0
    )
    gbm_c.fit(X_tr, y_tr)
    p_gbm = gbm_c.predict_proba(X_te)[:, 1]

    at_risk = p_gbm <= np.quantile(p_gbm, 0.1)
    results["retention"] = {
        "base_rate_active_m3": round(float(y_te.mean()), 3),
        "auc_logistic": round(float(roc_auc_score(y_te, p_lin)), 3),
        "auc_gbm": round(float(roc_auc_score(y_te, p_gbm)), 3),
        "avg_precision_gbm": round(float(average_precision_score(y_te, p_gbm)), 3),
        "lapse_rate_bottom_decile": round(float(1 - y_te[at_risk].mean()), 3),
        "lapse_rate_all": round(float(1 - y_te.mean()), 3),
        "deciles": decile_table(p_gbm, y_te.astype(float)),
    }

    # ---- 2. 180-day contribution -----------------------------------------
    v_tr, v_te = train["contribution_180"].to_numpy(), test["contribution_180"].to_numpy()
    gbm_r = HistGradientBoostingRegressor(
        categorical_features="from_dtype", learning_rate=0.05, max_iter=500, early_stopping=True, random_state=0
    )
    gbm_r.fit(X_tr, v_tr)
    v_hat = gbm_r.predict(X_te)
    heuristic = test["handle_14"].to_numpy()  # "sort by what they bet in two weeks"

    imp = permutation_importance(gbm_r, X_te, v_te, n_repeats=5, random_state=0, scoring="r2")
    importance = sorted(
        ({"feature": LABELS[c], "importance": round(float(m), 4)} for c, m in zip(X_te.columns, imp.importances_mean)),
        key=lambda d: -d["importance"],
    )

    # Validation only a simulation allows: does the model find the hidden high-value players?
    truth = pd.read_parquet(ROOT / "data" / "_truth.parquet").set_index("player_id")
    ttype = truth.loc[test["player_id"], "player_type"].to_numpy()
    top = v_hat >= np.quantile(v_hat, 0.9)

    results["value"] = {
        "mean_actual": round(float(v_te.mean()), 2),
        "mae_model": round(float(mean_absolute_error(v_te, v_hat)), 2),
        "mae_predict_mean": round(float(mean_absolute_error(v_te, np.full_like(v_te, v_tr.mean()))), 2),
        "spearman_model": round(float(spearmanr(v_hat, v_te).statistic), 3),
        "spearman_heuristic": round(float(spearmanr(heuristic, v_te).statistic), 3),
        "top_decile_model": top_decile(v_hat, v_te),
        "top_decile_heuristic": top_decile(heuristic, v_te),
        "deciles": decile_table(v_hat, v_te),
        "importance": importance[:10],
        "hidden_high_value_share_top_decile": round(float((ttype[top] == "high_value").mean()), 3),
        "hidden_high_value_share_all": round(float((ttype == "high_value").mean()), 3),
        "hidden_bonus_hunter_share_bottom_half": round(
            float((ttype[v_hat <= np.median(v_hat)] == "bonus_hunter").mean()), 3
        ),
        "hidden_bonus_hunter_share_all": round(float((ttype == "bonus_hunter").mean()), 3),
    }

    (OUT / "model_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items()}, indent=2)[:4000])


if __name__ == "__main__":
    main()
