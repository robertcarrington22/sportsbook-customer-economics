"""TabPFN as an alternative to gradient boosting, on the same features and test set.

TabPFN is a transformer pretrained on millions of synthetic tables. It predicts
on a new table in one forward pass, using the training rows as context, with no
tuning. The question here is practical: how does it compare with gradient
boosting when it only sees a small sample of players?

Three setups, all scored on the same May to August 2025 test cohorts:
  1. TabPFN v2, 3,000 training players as context
  2. XGBoost on the same 3,000 players
  3. XGBoost on all ~33,000 training players (from model.py)

TabPFN inference takes about 15 minutes on CPU, so it's a separate step and its
predictions are cached in outputs/tabpfn_predictions.parquet (use --refresh to redo). Results are
cached to outputs/model_alternatives.json and picked up by build_report.py.

Model weights: TabPFN v2 from Hugging Face (Prior-Labs/TabPFN-v2-clf, -reg),
Prior Labs License 1.1 (Apache 2.0 with attribution). Built with TabPFN.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

from model import CATS, NUMS, OUT, ROOT, fit_xgb, load, top_decile

N_CONTEXT = 3000
N_ESTIMATORS = 4
BATCH = 1000


def encode(df: pd.DataFrame) -> np.ndarray:
    X = df[CATS + NUMS].copy()
    for c in CATS:
        X[c] = X[c].cat.codes  # categories are shared across train and test via load()
    return X.to_numpy(dtype=float)


def predict_batched(fn, X: np.ndarray, label: str) -> np.ndarray:
    out, t0 = [], time.time()
    for i in range(0, len(X), BATCH):
        out.append(fn(X[i : i + BATCH]))
        print(f"  {label}: {min(i + BATCH, len(X)):,}/{len(X):,}  ({time.time() - t0:.0f}s)", flush=True)
    return np.concatenate(out)


def main() -> None:
    torch.set_num_threads(os.cpu_count() or 4)
    train, test = load()
    rng = np.random.default_rng(0)
    ctx = train.iloc[np.sort(rng.choice(len(train), N_CONTEXT, replace=False))]

    X_ctx, X_te = encode(ctx), encode(test)
    y_ctx, y_te = ctx["active_m3"].to_numpy(), test["active_m3"].to_numpy()
    v_ctx, v_te = ctx["contribution_180"].to_numpy(), test["contribution_180"].to_numpy()
    cat_idx = list(range(len(CATS)))

    # ---- TabPFN ----------------------------------------------------------
    # Predictions are cached: TabPFN takes ~15 minutes on CPU, the baselines take seconds.
    cache = OUT / "tabpfn_predictions.parquet"
    if cache.exists() and "--refresh" not in sys.argv:
        cached = pd.read_parquet(cache)
        assert cached["player_id"].tolist() == test["player_id"].tolist(), "test set changed; run with --refresh"
        p_pfn, v_pfn = cached["p_active_m3"].to_numpy(), cached["contribution_180_hat"].to_numpy()
        print("using cached TabPFN predictions")
    else:
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, device="cpu", n_estimators=N_ESTIMATORS, categorical_features_indices=cat_idx,
            random_state=0, ignore_pretraining_limits=True,
        )
        clf.fit(X_ctx, y_ctx)
        p_pfn = predict_batched(lambda x: clf.predict_proba(x)[:, 1], X_te, "TabPFN retention")
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, device="cpu", n_estimators=N_ESTIMATORS, categorical_features_indices=cat_idx,
            random_state=0, ignore_pretraining_limits=True,
        )
        reg.fit(X_ctx, v_ctx)
        v_pfn = predict_batched(reg.predict, X_te, "TabPFN value")
        pd.DataFrame({"player_id": test["player_id"], "p_active_m3": p_pfn, "contribution_180_hat": v_pfn}).to_parquet(cache)

    # ---- Gradient boosting on the same 3,000 players ---------------------
    gb_c = fit_xgb("clf", ctx[CATS + NUMS], y_ctx)
    p_gb = gb_c.predict_proba(test[CATS + NUMS])[:, 1]
    gb_r = fit_xgb("reg", ctx[CATS + NUMS], v_ctx)
    v_gb = gb_r.predict(test[CATS + NUMS])

    # ---- Full-data gradient boosting, from model.py ---------------------
    full = json.loads((OUT / "model_results.json").read_text())

    truth = pd.read_parquet(ROOT / "data" / "_truth.parquet").set_index("player_id")
    ttype = truth.loc[test["player_id"], "player_type"].to_numpy()

    def value_metrics(pred: np.ndarray) -> dict:
        top = pred >= np.quantile(pred, 0.9)
        return {
            "spearman": round(float(spearmanr(pred, v_te).statistic), 3),
            "mae": round(float(mean_absolute_error(v_te, pred)), 2),
            "top_decile_lift": top_decile(pred, v_te)["lift"],
            "hidden_high_value_share_top_decile": round(float((ttype[top] == "high_value").mean()), 3),
        }

    results = {
        "n_context": N_CONTEXT,
        "n_test": len(test),
        "n_estimators": N_ESTIMATORS,
        "models": [
            {"name": f"TabPFN v2, {N_CONTEXT:,} players", "auc": round(float(roc_auc_score(y_te, p_pfn)), 3), **value_metrics(v_pfn)},
            {"name": f"XGBoost, {N_CONTEXT:,} players", "auc": round(float(roc_auc_score(y_te, p_gb)), 3), **value_metrics(v_gb)},
            {
                "name": f"XGBoost, {full['split']['n_train']:,} players",
                "auc": full["retention"]["auc_gbm"],
                "spearman": full["value"]["spearman_model"],
                "mae": full["value"]["mae_model"],
                "top_decile_lift": full["value"]["top_decile_model"]["lift"],
                "hidden_high_value_share_top_decile": full["value"]["hidden_high_value_share_top_decile"],
            },
        ],
        "baseline_handle_only": {
            "spearman": full["value"]["spearman_heuristic"],
            "top_decile_lift": full["value"]["top_decile_heuristic"]["lift"],
        },
    }
    (OUT / "model_alternatives.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
