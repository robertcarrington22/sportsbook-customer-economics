"""Generate a simulated sportsbook player base from assumptions.toml.

Writes to data/:
  players.parquet   one row per first-time depositor (what an analyst would see)
  bets.parquet      one row per bet: date, bet type, sport, stake, the book's result (ggr),
                    and its theoretical result (theo_ggr = stake x expected hold for the bet type)
  activity.parquet  one row per player per day with at least one bet (bets aggregated)
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


def normalized(weights) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    return w / w.sum()


def lognormal_mean_one(rng: np.random.Generator, sigma: float, size) -> np.ndarray:
    """Multiplicative noise with mean exactly 1."""
    return rng.lognormal(-(sigma**2) / 2, sigma, size)


def pick(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Draw one category per row from a matrix of row-wise probabilities."""
    cum = np.cumsum(probs, axis=1)
    cum /= cum[:, -1:]
    return (rng.random((len(probs), 1)) > cum).sum(axis=1)


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

    scalar_keys = [k for k, v in cfg["types"]["casual"].items() if isinstance(v, (int, float))]
    tp = {k: np.array([cfg["types"][t][k] for t in TYPES], dtype=float) for k in scalar_keys}

    # --- bet types and sports ------------------------------------------------
    bet_types = list(cfg["bet_types"])
    bt_win = np.array([cfg["bet_types"][b]["win_prob"] for b in bet_types])
    bt_odds = np.array([cfg["bet_types"][b]["decimal_odds"] for b in bet_types])
    bt_stake = np.array([cfg["bet_types"][b]["stake_mult"] for b in bet_types])
    bt_hold = 1 - bt_win * bt_odds
    if (bt_hold <= 0).any():
        raise ValueError("Every bet type needs win_prob * decimal_odds < 1 (positive hold).")
    sports = list(cfg["sports"])
    season = np.array([[cfg["sports"][s]["season"].get(str(m), 0.0) for m in range(1, 13)] for s in sports])  # S x 12
    gp = cfg["gameplay"]

    type_mix = np.array([[cfg["types"][t]["bet_mix"][b] for b in bet_types] for t in TYPES])
    player_mix = np.vstack([rng.dirichlet(type_mix[t] * gp["mix_concentration"] + 1e-3) for t in ptype])
    type_aff = np.array([[cfg["types"][t]["sport_affinity"].get(s, 1.0) for s in sports] for t in TYPES])
    player_aff = type_aff[ptype] * lognormal_mean_one(rng, gp["sport_affinity_sigma"], (n, len(sports)))
    # How available a player's preferred sports are in each calendar month, relative to their own average.
    avail = player_aff @ season  # n x 12
    avail_rel = (avail / avail.mean(axis=1, keepdims=True)) ** gp["sport_season_strength"]
    # Divide out each player type's average pattern: the market seasonality already contains it, so this
    # only makes players differ from others of their type. Normalizing within type (not across everyone)
    # matters because survivors skew toward high-value players with different sport tastes.
    for t in range(len(TYPES)):
        m = ptype == t
        avail_rel[m] = avail_rel[m] / avail_rel[m].mean(axis=0, keepdims=True)

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
    # Moment-matched: rescale the draws to exactly zero mean and the target standard deviation, so a
    # single simulated history has the calibrated spread rather than whatever 22 random draws give.
    z = rng.normal(0, 1, n_months)
    market_shock = (z - z.mean()) / z.std() * cfg["economics"]["market_hold_sd"]

    # --- bets, in chunks of players to keep memory flat ----------------------
    frames = []
    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        ids = np.arange(lo, hi)
        counts = alive[ids]
        pid = np.repeat(ids, counts)
        offs = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        dates = signup[pid] + offs.astype("timedelta64[D]")
        month = dates.astype("datetime64[M]").astype(int) % 12  # 0 = January
        p = tp["p_active"][ptype[pid]] * engagement[pid] * act_season[month] * avail_rel[pid, month]
        active = (offs == 0) | (rng.random(len(pid)) < np.clip(p, 0, 0.95))
        pid, dates, month = pid[active], dates[active], month[active]

        # expand active days into bets
        t = ptype[pid]
        n_bets = 1 + rng.poisson(tp["bets_per_day"][t] - 1)
        day_stake = tp["stake_median"][t] * stake_mult[pid] * lognormal_mean_one(rng, 0.5, len(pid))
        b_pid = np.repeat(pid, n_bets)
        b_date = np.repeat(dates, n_bets)
        b_month = np.repeat(month, n_bets)
        b_day_stake = np.repeat(day_stake, n_bets)

        b_type = pick(rng, player_mix[b_pid])
        b_sport = pick(rng, player_aff[b_pid] * season[:, b_month].T)
        stake = b_day_stake * bt_stake[b_type] * lognormal_mean_one(rng, 0.3, len(b_pid))
        won = rng.random(len(b_pid)) < bt_win[b_type]
        month_idx = (b_date.astype("datetime64[M]") - first_month).astype(int)
        ggr = stake - won * stake * bt_odds[b_type] + stake * market_shock[month_idx]
        reinvest = tp["reinvest_pct"][ptype[b_pid]] * stake * bt_hold[b_type]
        frames.append(
            pd.DataFrame(
                {
                    "player_id": b_pid.astype(np.int32),
                    "activity_date": b_date,
                    "bet_type": np.array(bet_types)[b_type],
                    "sport": np.array(sports)[b_sport],
                    "stake": stake.round(2),
                    "ggr": ggr.round(2),
                    "theo_ggr": (stake * bt_hold[b_type]).round(4),
                    "reinvest": reinvest.round(4),
                }
            )
        )
    bets = pd.concat(frames, ignore_index=True)
    bets["bet_type"] = bets["bet_type"].astype("category")
    bets["sport"] = bets["sport"].astype("category")

    activity = (
        bets.groupby(["player_id", "activity_date"], observed=True, sort=True)
        .agg(bets=("stake", "size"), handle=("stake", "sum"), ggr=("ggr", "sum"), theo_ggr=("theo_ggr", "sum"),
             reinvest=("reinvest", "sum"))
        .reset_index()
    )
    activity["bets"] = activity["bets"].astype(np.int16)
    for c in ("handle", "ggr", "theo_ggr", "reinvest"):
        activity[c] = activity[c].round(2)

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
    state_df = pd.DataFrame({"state": states, "tax_rate": [cfg["states"][s]["tax_rate"] for s in states]})

    DATA.mkdir(exist_ok=True)
    players.to_parquet(DATA / "players.parquet", index=False)
    bets.to_parquet(DATA / "bets.parquet", index=False)
    activity.to_parquet(DATA / "activity.parquet", index=False)
    truth.to_parquet(DATA / "_truth.parquet", index=False)
    state_df.to_csv(DATA / "states.csv", index=False)

    print(f"players:  {len(players):>10,}")
    print(f"bets:     {len(bets):>10,}")
    print(f"activity: {len(activity):>10,} active player-days")
    print(f"handle:   ${activity['handle'].sum():>14,.0f}")
    print(f"GGR:      ${activity['ggr'].sum():>14,.0f}  (hold {activity['ggr'].sum() / activity['handle'].sum():.1%})")
    print("type mix:", truth["player_type"].value_counts(normalize=True).round(3).to_dict())
    by_type = bets.groupby("bet_type", observed=True).agg(handle=("stake", "sum"), ggr=("ggr", "sum"))
    print("hold by bet type:", (by_type.ggr / by_type.handle).round(3).to_dict(),
          "| handle share:", (by_type.handle / by_type.handle.sum()).round(3).to_dict())


if __name__ == "__main__":
    main()
