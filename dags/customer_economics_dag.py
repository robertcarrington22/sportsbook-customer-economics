"""Airflow DAGs for running this pipeline in production.

Not executed in this repo (Airflow isn't installed here); this shows how the
pieces would be scheduled. In production the SQL in sql/ would run on the
warehouse (Databricks or Snowflake) instead of DuckDB, reading real player and
bet tables in place of data/*.parquet. The queries use standard SQL; the main
DuckDB-specific pieces to translate are range() for the calendar join,
date_diff('day', a, b), and FILTER clauses on aggregates (Spark SQL supports
FILTER; Snowflake needs CASE WHEN inside the aggregate).

Three DAGs:
  daily    score new day-14 FTDs, check drift, refresh the dashboard
  weekly   rebuild unit economics, LTV curves, and bid caps
  monthly  retrain the models and promote them only if they beat the current ones
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator

REPO = "/opt/airflow/repos/sportsbook-customer-economics"
PY = f"cd {REPO} && python"

default_args = {
    "owner": "customer-economics",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": True,
}

# ---- Daily: score the new cohort ---------------------------------------------
with DAG(
    dag_id="customer_economics_daily",
    description="Score FTDs reaching day 14, monitor drift, refresh the dashboard",
    schedule="0 7 * * *",  # after the nightly warehouse load
    start_date=datetime(2026, 10, 1),
    catchup=False,
    default_args=default_args,
    tags=["customer-economics"],
) as daily:
    build_features = BashOperator(
        task_id="build_features",
        bash_command=f"{PY} -c 'import analyze; analyze.main()'",
    )
    score = BashOperator(
        task_id="score_day14",
        bash_command=f"{PY} score.py --as-of {{{{ ds }}}}",
    )
    drift_gate = ShortCircuitOperator(
        # Stop downstream publishing (and alert) if any feature drifts past PSI 0.2.
        task_id="check_drift",
        python_callable=lambda ds, **_: __import__("pandas").read_csv(
            f"{REPO}/outputs/scores/drift_{ds}.csv"
        )["alert"].fillna(False).sum() == 0,
    )
    publish = BashOperator(
        task_id="publish_dashboard",
        bash_command=f"{PY} build_dashboard.py",
    )
    build_features >> score >> drift_gate >> publish

# ---- Weekly: unit economics, LTV, bid caps ------------------------------------
with DAG(
    dag_id="customer_economics_weekly",
    description="Rebuild channel/offer/state economics, LTV projections, and the report",
    schedule="0 8 * * 1",
    start_date=datetime(2026, 10, 5),
    catchup=False,
    default_args=default_args,
    tags=["customer-economics"],
) as weekly:
    economics = BashOperator(task_id="unit_economics", bash_command=f"{PY} analyze.py")
    ltv = BashOperator(task_id="ltv_projection", bash_command=f"{PY} ltv_forecast.py")
    report = BashOperator(task_id="report", bash_command=f"{PY} build_report.py")
    economics >> ltv >> report

# ---- Monthly: retrain with a promotion gate -----------------------------------
with DAG(
    dag_id="customer_economics_monthly_retrain",
    description="Retrain day-14 models; promote only if they beat the current models on recent cohorts",
    schedule="0 9 1 * *",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    default_args=default_args,
    tags=["customer-economics"],
) as monthly:
    retrain = BashOperator(
        task_id="retrain_candidate",
        # Writes a candidate bundle; the out-of-time test set is always the most recent cohorts.
        bash_command=f"{PY} model.py",
    )
    promote = BashOperator(
        task_id="promote_if_better",
        # Promote the candidate only if retention AUC and value Spearman on the latest
        # held-out cohorts are no worse than the live model's, within a small tolerance.
        bash_command=f"{PY} -c \"print('compare candidate vs live metrics, then swap models/day14_models.joblib')\"",
    )
    retrain >> promote
