"""Parse DraftKings' monthly New York mobile sports wagering filings.

Source: New York State Gaming Commission, "Mobile Sports Wagering Report,
DraftKings", monthly. Public PDF at gaming.ny.gov. Handle and GGR are reported
by the operator to the state; New York taxes mobile sports GGR at 51%.

Writes:
  ny_draftkings_monthly.csv   month, handle, ggr, hold, state_share
  ny_seasonality.csv          handle index by calendar month (trend removed)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber

HERE = Path(__file__).parent
PDF = HERE / "raw" / "ny-monthly-mobile-sports-wagering-report-draftkings.pdf"
ROW = re.compile(r"^([A-Z][a-z]{2}-\d{2})\s+\$([\d,]+)\s+\$([\d,]+)\s+\$([\d,]+)(?:.*\$([\d,]+))?")


def money(s: str) -> float:
    return float(s.replace(",", ""))


def main() -> None:
    rows = []
    with pdfplumber.open(PDF) as pdf:
        for page in pdf.pages:
            for line in (page.extract_text() or "").splitlines():
                m = ROW.match(line.strip())
                if not m:
                    continue
                month, handle, ggr, to_platform, to_state = m.groups()
                if money(handle) == 0:
                    continue
                rows.append(
                    {
                        "month": pd.to_datetime(month, format="%b-%y"),
                        "handle": money(handle),
                        "ggr": money(ggr),
                        "to_operator": money(to_platform),
                        "to_state": money(to_state) if to_state else np.nan,
                    }
                )
    df = pd.DataFrame(rows).drop_duplicates("month").sort_values("month").reset_index(drop=True)
    df["hold"] = df.ggr / df.handle
    df["state_share"] = df.to_state / df.ggr
    df.to_csv(HERE / "ny_draftkings_monthly.csv", index=False)

    # Seasonality: handle relative to a centered 12-month moving average, which
    # removes market growth. Average the ratio by calendar month, scale to mean 1.
    s = df.set_index("month")["handle"]
    trend = s.rolling(12, center=True, min_periods=12).mean()
    ratio = (s / trend).dropna()
    idx = ratio.groupby(ratio.index.month).mean()
    idx = (idx / idx.mean()).round(3)
    seas = pd.DataFrame({"month_of_year": idx.index, "handle_index": idx.values, "n_years": ratio.groupby(ratio.index.month).size().values})
    seas.to_csv(HERE / "ny_seasonality.csv", index=False)

    print(f"{len(df)} months, {df.month.min():%b %Y} to {df.month.max():%b %Y}")
    last12 = df.tail(12)
    print(f"last 12 months: handle ${last12.handle.sum() / 1e9:.2f}B, GGR ${last12.ggr.sum() / 1e6:.0f}M, "
          f"hold {last12.ggr.sum() / last12.handle.sum():.1%}")
    print(f"monthly hold range (last 24): {df.tail(24).hold.min():.1%} to {df.tail(24).hold.max():.1%}")
    print(f"state share of GGR: {df.state_share.median():.3f}")
    print(seas.to_string(index=False))


if __name__ == "__main__":
    main()
