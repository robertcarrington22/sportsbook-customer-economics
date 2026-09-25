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
from sklearn.base import BaseEstimator, RegressorMixin
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


class TwoPart(RegressorMixin, BaseEstimator):
    """Two-part value model for a heavy-tailed target that can be negative.

    P(value > 0)          XGBoost classifier
    E[value | value > 0]  XGBoost on log(1 + value), retransformed with Duan's smearing
                          factor estimated on held-out rows
    E[value | value <= 0] XGBoost on the (bounded) losses
    E[value] = p * E[pos] + (1 - p) * E[neg]
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TwoPart":
        pos = y > 0
        self.clf = fit_xgb("clf", X, pos.astype(int))
        Xp, vp = X[pos].reset_index(drop=True), np.log1p(y[pos])
        fit_idx, smear_idx = train_test_split(np.arange(len(Xp)), test_size=0.2, random_state=1)
        self.reg_pos = fit_xgb("reg", Xp.iloc[fit_idx], vp[fit_idx])
        self.smear = float(np.mean(np.exp(vp[smear_idx] - self.reg_pos.predict(Xp.iloc[smear_idx]))))
        self.reg_neg = fit_xgb("reg", X[~pos], y[~pos])
        return self

    def p_pos(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.p_pos(X)
        e_pos = np.exp(self.reg_pos.predict(X)) * self.smear - 1
        return p * e_pos + (1 - p) * self.reg_neg.predict(X)


def hold_luck() -> dict:
    """Realized vs expected hold over days 14-179, for training and test cohorts."""
    import duckdb
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT CASE WHEN p.cohort_month < DATE '{TEST_START.date()}' THEN 'train' ELSE 'test' END AS grp,
               sum(b.ggr) / sum(b.stake) AS realized_hold, sum(b.theo_ggr) / sum(b.stake) AS expected_hold
        FROM read_parquet('{(ROOT / "data" / "bets.parquet").as_posix()}') AS b
        JOIN read_parquet('{(ROOT / "data" / "players.parquet").as_posix()}') AS p USING (player_id)
        WHERE date_diff('day', p.signup_date, b.activity_date) BETWEEN 14 AND 179
        GROUP BY 1
    """).df().set_index("grp")
    return {g: {"realized_hold": round(float(r.realized_hold), 4), "expected_hold": round(float(r.expected_hold), 4),
                "luck_pts": round(float((r.realized_hold - r.expected_hold) * 100), 2)} for g, r in df.iterrows()}


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
    "parlay_share_14",
    "live_share_14",
    "n_sports_14",
    "football_share_14",
    "season_ahead_180",
    "season_m3",
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
    "parlay_share_14": "Parlay share of handle",
    "live_share_14": "Live-bet share of handle",
    "n_sports_14": "Sports bet",
    "football_share_14": "Football share of handle",
    "season_ahead_180": "Season ahead, days 14 to 179",
    "season_m3": "Season ahead, month 3",
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

    # ---- 2. 180-day value -------------------------------------------------
    # Target: theoretical contribution, computed on expected GGR (stake x the book's
    # expected hold for each bet type) instead of realized GGR. Six months of realized
    # results carry a lot of bet-outcome luck that no behavioral model can predict,
    # which is why sportsbooks value players on theo. Realized value is still reported.
    t_tr, t_te = train["theo_contribution_180"].to_numpy(), test["theo_contribution_180"].to_numpy()
    r_te = test["contribution_180"].to_numpy()
    heuristic = test["handle_14"].to_numpy()  # "sort by what they bet in two weeks"

    two = TwoPart().fit(X_tr, t_tr)          # primary model
    v_two = two.predict(X_te)
    sq = fit_xgb("reg", X_tr, t_tr)          # plain squared-error XGBoost, for comparison
    v_sq = sq.predict(X_te)
    realized_model = fit_xgb("reg", X_tr, train["contribution_180"].to_numpy())  # the naive target
    v_real = realized_model.predict(X_te)

    imp = permutation_importance(two, X_te, t_te, n_repeats=5, random_state=0, scoring="r2")
    importance = sorted(
        ({"feature": LABELS[c], "importance": round(float(m), 4)} for c, m in zip(X_te.columns, imp.importances_mean)),
        key=lambda d: -d["importance"],
    )

    # Validation only a simulation allows: does the model find the hidden high-value players?
    truth = pd.read_parquet(ROOT / "data" / "_truth.parquet").set_index("player_id")
    ttype = truth.loc[test["player_id"], "player_type"].to_numpy()
    hv_share = lambda pred: round(float((ttype[pred >= np.quantile(pred, 0.9)] == "high_value").mean()), 3)  # noqa: E731
    sp = lambda a_, b_: round(float(spearmanr(a_, b_).statistic), 3)  # noqa: E731

    results["value"] = {
        "target": "theoretical contribution, days 0 to 179",
        "model": "two-part XGBoost",
        "mean_actual": round(float(t_te.mean()), 2),
        "mean_predicted": round(float(v_two.mean()), 2),
        "mae_model": round(float(mean_absolute_error(t_te, v_two)), 2),
        "mae_predict_mean": round(float(mean_absolute_error(t_te, np.full_like(t_te, t_tr.mean()))), 2),
        "spearman_model": sp(v_two, t_te),
        "spearman_heuristic": sp(heuristic, t_te),
        "top_decile_model": top_decile(v_two, t_te),
        "top_decile_heuristic": top_decile(heuristic, t_te),
        "deciles": decile_table(v_two, t_te),
        "importance": importance[:12],
        "smearing_factor": round(two.smear, 4),
        "auc_profitable": round(float(roc_auc_score(t_te > 0, two.p_pos(X_te))), 3),
        "hidden_high_value_share_top_decile": hv_share(v_two),
        "hidden_high_value_share_all": round(float((ttype == "high_value").mean()), 3),
        "realized": {
            "mean_actual": round(float(r_te.mean()), 2),
            "spearman_model": sp(v_two, r_te),
            "spearman_heuristic": sp(heuristic, r_te),
            "spearman_realized_trained_model": sp(v_real, r_te),
            "sd_theo": round(float(t_te.std()), 2),
            "sd_realized": round(float(r_te.std()), 2),
            "corr_theo_realized": round(float(np.corrcoef(t_te, r_te)[0, 1]), 3),
        },
    }
    results["value_squared_error"] = {
        "spearman_model": sp(v_sq, t_te),
        "mae_model": round(float(mean_absolute_error(t_te, v_sq)), 2),
        "top_decile_model": top_decile(v_sq, t_te),
        "deciles": decile_table(v_sq, t_te),
        "mean_predicted": round(float(v_sq.mean()), 2),
        "hidden_high_value_share_top_decile": hv_share(v_sq),
    }
    results["value_realized_target"] = {
        "spearman_vs_theo": sp(v_real, t_te),
        "top_ratio_vs_theo": round(float(v_real[v_real >= np.quantile(v_real, 0.9)].mean()
                                         / t_te[v_real >= np.quantile(v_real, 0.9)].mean()), 3),
        "mean_predicted": round(float(v_real.mean()), 2),
    }
    results["luck"] = hold_luck()

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

    # Persist the models the daily scoring job uses (score.py), with the training
    # feature distributions it needs for drift monitoring.
    import joblib
    MODELS = ROOT / "models"
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"retention": gbm_c, "value": two, "features": CATS + NUMS,
                 "train_value_p90": float(np.quantile(two.predict(X_tr), 0.9)),
                 "train_cohorts": results["split"]["train_cohorts"]}, MODELS / "day14_models.joblib")
    train[CATS + NUMS].to_parquet(MODELS / "train_features.parquet")
    (OUT / "model_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k not in ("retention",)}, indent=1)[:3500])


if __name__ == "__main__":
    # Run through the imported module so saved models reference model.TwoPart,
    # not __main__.TwoPart, and can be loaded by score.py.
    import model as _model

    _model.main()
