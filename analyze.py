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


if __name__ == "__main__":
    main()
