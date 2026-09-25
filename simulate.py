"""Generate a simulated sportsbook player base from assumptions.toml.

Writes to data/:
  players.parquet   one row per first-time depositor (what an analyst would see)
  activity.parquet  one row per player per day they placed at least one bet
  states.csv        state tax rates
  _truth.parquet    hidden player type and lifetime, for validating the models only.
                    Nothing in sql/ or model.py reads it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
TYPES = ["bonus_hunter", "casual", "regular", "high_value"]
CHUNK = 4000


def load_config() -> dict:
    with open(ROOT / "assumptions.toml", "rb") as f:
        return tomllib.load(f)


def normalized(weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


def lognormal_mean_one(rng: np.random.Generator, sigma: float, size) -> np.ndarray:
    """Multiplicative noise with mean exactly 1."""
    return rng.lognormal(-(sigma**2) / 2, sigma, size)


def main() -> None:
    cfg = load_config()
    sim = cfg["simulation"]
    rng = np.random.default_rng(sim["seed"])
    n = sim["n_ftds"]

    start = pd.Timestamp(sim["signup_start"])
    end = pd.Timestamp(sim["signup_end"])
    obs_end = pd.Timestamp(sim["observation_end"])

    # --- signup dates, weighted by season and a small weekend bump ----------
    days = pd.date_range(start, end, freq="D")
    w = np.array([sim["signup_seasonality"][str(d.month)] for d in days], dtype=float)
    w *= np.where(days.dayofweek >= 5, 1.25, 1.0)
    signup = np.sort(days.values[rng.choice(len(days), size=n, p=w / w.sum())])
    signup = signup.astype("datetime64[D]")

    # --- channel, offer, state ---------------------------------------------
    channels = list(cfg["channels"])
    offers = list(cfg["offers"])
    states = list(cfg["states"])
    channel = rng.choice(channels, n, p=normalized([cfg["channels"][c]["share"] for c in channels]))
    offer = rng.choice(offers, n, p=normalized([cfg["offers"][o]["share"] for o in offers]))
    state = rng.choice(states, n, p=normalized([cfg["states"][s]["share"] for s in states]))

    # --- hidden player type: channel mix tilted by offer ---------------------
    ptype = np.empty(n, dtype=np.int8)
    for c in channels:
        base = np.array([cfg["channels"][c]["type_mix"][t] for t in TYPES], dtype=float)
        for o in offers:
            tilt = np.array([cfg["offers"][o]["type_tilt"].get(t, 1.0) for t in TYPES])
            mask = (channel == c) & (offer == o)
            ptype[mask] = rng.choice(len(TYPES), mask.sum(), p=normalized(base * tilt))

    tp = {k: np.array([cfg["types"][t][k] for t in TYPES], dtype=float) for k in cfg["types"]["casual"]}

    # --- player-level economics and persistent traits -------------------------
    cac = np.array([cfg["channels"][c]["cac"] for c in channel]) * lognormal_mean_one(rng, 0.2, n)
    welcome_cost = (
        np.array([cfg["offers"][o]["expected_cost"] for o in offer])
        * tp["promo_use"][ptype]
        * lognormal_mean_one(rng, 0.25, n)
    )
    engagement = lognormal_mean_one(rng, 0.4, n)
    stake_mult = rng.lognormal(0, tp["stake_sigma"][ptype])

    # --- lifetime: early exit within 30 days, else geometric churn after day 30
    early = rng.random(n) < tp["early_exit_prob"][ptype]
    daily_hazard = 1 - (1 - tp["monthly_churn"][ptype]) ** (1 / 30)
    lifetime = np.where(early, rng.integers(1, 31, n), 30 + rng.geometric(daily_hazard))
    obs_days = (np.datetime64(obs_end.date(), "D") - signup).astype(int) + 1
    alive = np.minimum(lifetime, obs_days)

    act_season = np.array([sim["activity_seasonality"][str(m)] for m in range(1, 13)])

    # One hold shock per calendar month, shared by every player (same games, same results).
    first_month = np.datetime64(start.date(), "M")
    n_months = int((np.datetime64(obs_end.date(), "M") - first_month).astype(int)) + 1
    market_shock = rng.normal(0, cfg["economics"]["market_hold_sd"], n_months)
    hold = 1 - tp["win_prob"] * tp["decimal_odds"]
    if (hold <= 0).any():
        raise ValueError("Every player type needs win_prob * decimal_odds < 1 (positive hold).")

    # --- daily activity, in chunks to keep memory flat ------------------------
    frames = []
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        ids = np.arange(lo, hi)
        counts = alive[ids]
        pid = np.repeat(ids, counts)
        offs = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        dates = signup[pid] + offs.astype("timedelta64[D]")
        month = dates.astype("datetime64[M]").astype(int) % 12  # 0 = January
        p = tp["p_active"][ptype[pid]] * engagement[pid] * act_season[month]
        active = (offs == 0) | (rng.random(len(pid)) < np.clip(p, 0, 0.95))
        pid, dates = pid[active], dates[active]

        t = ptype[pid]
        bets = 1 + rng.poisson(tp["bets_per_day"][t] - 1)
        stake = tp["stake_median"][t] * stake_mult[pid] * lognormal_mean_one(rng, 0.5, len(pid))
        handle = bets * stake
        wins = rng.binomial(bets, tp["win_prob"][t])
        ggr = handle - wins * stake * tp["decimal_odds"][t]
        month_idx = (dates.astype("datetime64[M]") - first_month).astype(int)
        ggr = ggr + handle * market_shock[month_idx]
        reinvest = tp["reinvest_pct"][t] * handle * hold[t]
        frames.append(
            pd.DataFrame(
                {
                    "player_id": pid.astype(np.int32),
                    "activity_date": dates,
                    "bets": bets.astype(np.int16),
                    "handle": handle.round(2),
                    "ggr": ggr.round(2),
                    "reinvest": reinvest.round(2),
                }
            )
        )
    activity = pd.concat(frames, ignore_index=True)

    players = pd.DataFrame(
        {
            "player_id": np.arange(n, dtype=np.int32),
            "signup_date": signup,
            "cohort_month": signup.astype("datetime64[M]").astype("datetime64[D]"),
            "channel": channel,
            "offer": offer,
            "state": state,
            "cac": cac.round(2),
            "welcome_promo_cost": welcome_cost.round(2),
        }
    )
    truth = pd.DataFrame(
        {
            "player_id": players["player_id"],
            "player_type": np.array(TYPES)[ptype],
            "lifetime_days": lifetime,
            "engagement": engagement.round(3),
            "stake_mult": stake_mult.round(3),
        }
    )
    state_df = pd.DataFrame(
        {"state": states, "tax_rate": [cfg["states"][s]["tax_rate"] for s in states]}
    )

    DATA.mkdir(exist_ok=True)
    players.to_parquet(DATA / "players.parquet", index=False)
    activity.to_parquet(DATA / "activity.parquet", index=False)
    truth.to_parquet(DATA / "_truth.parquet", index=False)
    state_df.to_csv(DATA / "states.csv", index=False)

    print(f"players:  {len(players):>10,}")
    print(f"activity: {len(activity):>10,} active player-days")
    print(f"handle:   ${activity['handle'].sum():>14,.0f}")
    print(f"GGR:      ${activity['ggr'].sum():>14,.0f}  (hold {activity['ggr'].sum() / activity['handle'].sum():.1%})")
    print("type mix:", truth["player_type"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
