"""Run every file in sql/ in order against DuckDB and export the results.

Each .sql file may contain several statements. The result of the last
statement in each file is written to outputs/<file>.csv. The modeling table
is also written to outputs/features.parquet for model.py.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SQL = ROOT / "sql"
OUT = ROOT / "outputs"


def statements(text: str) -> list[str]:
    no_comments = re.sub(r"--[^\n]*", "", text)
    return [s.strip() for s in no_comments.split(";") if s.strip()]


def connect() -> duckdb.DuckDBPyConnection:
    with open(ROOT / "assumptions.toml", "rb") as f:
        cfg = tomllib.load(f)
    con = duckdb.connect()
    con.execute(f"CREATE VIEW players AS SELECT * FROM read_parquet('{(DATA / 'players.parquet').as_posix()}')")
    con.execute(f"CREATE VIEW activity AS SELECT * FROM read_parquet('{(DATA / 'activity.parquet').as_posix()}')")
    con.execute(f"CREATE VIEW states AS SELECT * FROM read_csv_auto('{(DATA / 'states.csv').as_posix()}')")
    con.execute(
        "CREATE TABLE params AS SELECT "
        f"DATE '{cfg['simulation']['observation_end']}' AS observation_end, "
        f"{cfg['economics']['variable_cost_pct_handle']} AS variable_cost_pct_handle"
    )
    return con


def main() -> None:
    OUT.mkdir(exist_ok=True)
    con = connect()
    for path in sorted(SQL.glob("*.sql")):
        result = None
        for stmt in statements(path.read_text()):
            result = con.execute(stmt)
        df = result.df()
        df.to_csv(OUT / f"{path.stem}.csv", index=False)
        print(f"\n== {path.name}  ({len(df):,} rows)")
        print(df.head(12).to_string(index=False))
    con.execute(f"COPY features TO '{(OUT / 'features.parquet').as_posix()}' (FORMAT PARQUET)")
    bootstrap_channel_ci(con)


def bootstrap_channel_ci(con: duckdb.DuckDBPyConnection, n_boot: int = 2000, seed: int = 7) -> None:
    """95% bootstrap intervals for 12-month LTV and LTV/CAC by channel.

    Resamples players within each channel. Value is heavy-tailed, so a few
    whales can move a channel's average a lot; the interval shows how much.
    """
    import numpy as np
    import pandas as pd

    pp = con.execute(
        """
        SELECT player_id, channel, any_value(cac) AS cac, sum(contribution) AS c12
        FROM player_month
        WHERE last_full_month >= 11 AND life_month <= 11
        GROUP BY player_id, channel
        ORDER BY player_id
        """
    ).df()
    rng = np.random.default_rng(seed)
    rows = []
    for channel, g in pp.groupby("channel"):
        c12, cac = g["c12"].to_numpy(), g["cac"].to_numpy()
        idx = rng.integers(0, len(g), size=(n_boot, len(g)))
        ltv = c12[idx].mean(axis=1)
        ratio = ltv / cac[idx].mean(axis=1)
        rows.append(
            {
                "channel": channel,
                "ftds": len(g),
                "ltv_12m": round(float(c12.mean()), 0),
                "ltv_lo": round(float(np.quantile(ltv, 0.025)), 0),
                "ltv_hi": round(float(np.quantile(ltv, 0.975)), 0),
                "ratio": round(float(c12.mean() / cac.mean()), 2),
                "ratio_lo": round(float(np.quantile(ratio, 0.025)), 2),
                "ratio_hi": round(float(np.quantile(ratio, 0.975)), 2),
                "p_ratio_below_1": round(float((ratio < 1).mean()), 3),
            }
        )
    df = pd.DataFrame(rows).sort_values("ratio", ascending=False)
    df.to_csv(OUT / "16_channel_bootstrap.csv", index=False)
    print("\n== bootstrap: 12-month LTV / CAC, 95% interval")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
