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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# XGBoost settings shared by model.py and model_tabpfn.py. Native categorical
# splits, histogram trees, and early stopping on a held-out 10% of the training
# rows, so the number of trees is chosen by validation rather than by hand.
XGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
    early_stopping_rounds=100, random_state=0, n_jobs=-1,
)


def fit_xgb(kind: str, X: pd.DataFrame, y: np.ndarray):
    """Fit an XGBoost classifier ('clf') or squared-error regressor ('reg') with early stopping."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X, y, test_size=0.1, random_state=0, stratify=y if kind == "clf" else None
    )
    model = XGBClassifier(**XGB_PARAMS, eval_metric="auc") if kind == "clf" else \
        XGBRegressor(**XGB_PARAMS, objective="reg:squarederror", eval_metric="rmse")
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model


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
    # decides which rows XGBoost holds out for early stopping.
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
        "gbm": "XGBoost",
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

    gbm_c = fit_xgb("clf", X_tr, y_tr)
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
    gbm_r = fit_xgb("reg", X_tr, v_tr)
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

    # ---- 3. Two-part value model -------------------------------------------
    # The squared-error model shrinks the heavy right tail and predicts too low
    # for the most valuable players. Split the problem:
    #   P(profitable)                     XGBoost classifier
    #   E[value | profitable]             XGBoost on log(1 + value), retransformed
    #                                     with Duan's smearing factor from held-out rows
    #   E[value | not profitable]         XGBoost on the (bounded) losses
    # and combine: E[value] = p * E[pos] + (1 - p) * E[neg].
    pos = v_tr > 0
    clf_pos = fit_xgb("clf", X_tr, pos.astype(int))
    p_pos = clf_pos.predict_proba(X_te)[:, 1]
    Xp, vp = X_tr[pos].reset_index(drop=True), np.log1p(v_tr[pos])
    fit_idx, smear_idx = train_test_split(np.arange(len(Xp)), test_size=0.2, random_state=1)
    reg_pos = fit_xgb("reg", Xp.iloc[fit_idx], vp[fit_idx])
    smear = float(np.mean(np.exp(vp[smear_idx] - reg_pos.predict(Xp.iloc[smear_idx]))))
    e_pos = np.exp(reg_pos.predict(X_te)) * smear - 1
    reg_neg = fit_xgb("reg", X_tr[~pos], v_tr[~pos])
    e_neg = reg_neg.predict(X_te)
    v_two = p_pos * e_pos + (1 - p_pos) * e_neg
    top2 = v_two >= np.quantile(v_two, 0.9)
    results["value_two_part"] = {
        "smearing_factor": round(smear, 4),
        "auc_profitable": round(float(roc_auc_score(v_te > 0, p_pos)), 3),
        "mae_model": round(float(mean_absolute_error(v_te, v_two)), 2),
        "spearman_model": round(float(spearmanr(v_two, v_te).statistic), 3),
        "top_decile_model": top_decile(v_two, v_te),
        "deciles": decile_table(v_two, v_te),
        "mean_predicted": round(float(v_two.mean()), 2),
        "mean_predicted_squared_error_model": round(float(v_hat.mean()), 2),
        "hidden_high_value_share_top_decile": round(float((ttype[top2] == "high_value").mean()), 3),
    }

    # ---- 4. Calibration of the retention models ----------------------------
    def reliability(p: np.ndarray, y: np.ndarray) -> list[dict]:
        bins = pd.qcut(p, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"b": bins, "p": p, "y": y}).groupby("b").agg(pred=("p", "mean"), actual=("y", "mean"))
        return [{"pred": round(float(r.pred), 4), "actual": round(float(r.actual), 4)} for r in g.itertuples()]

    def brier(p, y):
        return round(float(np.mean((p - y) ** 2)), 4)

    results["calibration"] = {
        "xgboost": {"brier": brier(p_gbm, y_te), "bins": reliability(p_gbm, y_te)},
        "logistic": {"brier": brier(p_lin, y_te), "bins": reliability(p_lin, y_te)},
        "brier_base_rate": brier(np.full_like(p_gbm, y_tr.mean()), y_te),
    }

    (OUT / "model_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k in ("value_two_part", "calibration")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
