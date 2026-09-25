"""Write REPORT.md and the figures in figures/ from the files in outputs/.

Every number in the report is read from outputs/, so re-running the pipeline
regenerates the whole document.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
REAL = ROOT / "real_data"

INK, INK2, GRID = "#1B232B", "#5A646E", "#E6E9E4"
WIN, LOSS, ACCENT, AMBER, PURPLE = "#1E7A4F", "#B23A2E", "#1D4E89", "#C27C0E", "#7A5AA6"
CH_COLORS = {"referral": WIN, "tv_brand": ACCENT, "search": PURPLE, "affiliate": AMBER, "paid_social": LOSS}
NAMES = {
    "referral": "Referral", "tv_brand": "TV and brand", "search": "Search", "affiliate": "Affiliate",
    "paid_social": "Paid social", "bet5_get200": "Bet $5, get $200", "no_sweat_1000": "No-sweat first bet",
    "deposit_match_250": "Deposit match",
}

plt.rcParams.update({
    "font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 10.5, "axes.edgecolor": "#C9CEC7",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "text.parse_math": False, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 10, "legend.frameon": False, "figure.dpi": 100,
})


def nm(k: str) -> str:
    return NAMES.get(k, k)


def usd(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def pct(x: float, d: int = 0) -> str:
    return f"{x * 100:.{d}f}%"


def money_axis(ax, axis="y"):
    fmt = mtick.FuncFormatter(lambda v, _: usd(v))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def save(fig, name: str) -> str:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{name}"


def table(df: pd.DataFrame, align: str) -> str:
    """Markdown table. `align` is one character per column: l or r."""
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(":---" if a == "l" else "---:" for a in align) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_real(ny: pd.DataFrame, seas: pd.DataFrame) -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.7, 1]})
    a1.bar(ny.month, ny.handle / 1e6, width=22, color="#AFC3D9", label="Handle ($M)")
    a1.set_ylabel("Handle, $M")
    a1.set_title("DraftKings, New York mobile sports: handle and hold")
    b = a1.twinx()
    b.plot(ny.month, ny.hold, color=ACCENT, lw=2, label="Hold")
    b.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    b.set_ylim(0, 0.14); b.grid(False); b.spines["right"].set_visible(True); b.set_ylabel("Hold (GGR / handle)")
    h1, l1 = a1.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, loc="upper left")
    months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    cols = [LOSS if v < 0.9 else ACCENT for v in seas.handle_index]
    a2.bar(range(12), seas.handle_index, color=cols)
    a2.axhline(1, color=INK2, lw=1)
    a2.set_xticks(range(12), months)
    a2.set_ylim(0.5, 1.3)
    a2.set_title("Handle by calendar month, trend removed")
    a2.set_ylabel("Index (1.0 = average month)")
    fig.tight_layout()
    return save(fig, "01_real_ny.png")


def fig_waterfall(wf: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    running, bottoms, heights, colors = 0.0, [], [], []
    for i, r in enumerate(wf.itertuples()):
        if i == 0 or r.step == "Contribution":
            bottoms.append(0); heights.append(r.per_ftd); colors.append(ACCENT if i == 0 else WIN)
            running = r.per_ftd
        else:
            bottoms.append(running + r.per_ftd); heights.append(-r.per_ftd); colors.append(LOSS)
            running += r.per_ftd
    x = np.arange(len(wf))
    ax.bar(x, heights, bottom=bottoms, color=colors, width=0.6)
    for xi, (bt, h, r) in enumerate(zip(bottoms, heights, wf.itertuples())):
        ax.text(xi, bt + h + 18, usd(r.per_ftd), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, [s.replace(" ", "\n", 1) for s in wf.step])
    money_axis(ax)
    ax.set_ylim(0, wf.per_ftd.iloc[0] * 1.12)
    ax.set_title("Year-one revenue per FTD, gross gaming revenue to contribution")
    return save(fig, "02_waterfall.png")


def fig_concentration(conc: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    c = conc[conc.pctile <= 30]
    ax.plot(c.pctile, c.cum_share_of_total, color=ACCENT, lw=2.5)
    ax.axhline(1, color=INK2, lw=1, ls="--")
    for p in (1, 5, 10):
        y = conc.loc[conc.pctile == p, "cum_share_of_total"].iloc[0]
        ax.scatter([p], [y], color=ACCENT, zorder=3)
        ax.annotate(f"top {p}%: {pct(y)}", (p, y), textcoords="offset points", xytext=(9, -3), va="center",
                    fontsize=9.5, color=INK)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Top X% of FTDs by year-one contribution")
    ax.set_ylabel("Share of total contribution")
    ax.set_title("Year-one contribution is concentrated in a small share of players")
    return save(fig, "03_concentration.png")


def fig_distributions(feats: pd.DataFrame) -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    c = feats.contribution_180.clip(-400, 3000)
    a1.hist(c, bins=np.linspace(-400, 3000, 69), color=ACCENT, alpha=0.85)
    a1.axvline(0, color=INK2, lw=1)
    med, mean = feats.contribution_180.median(), feats.contribution_180.mean()
    a1.axvline(med, color=LOSS, lw=1.5, ls="--", label=f"median {usd(med)}")
    a1.axvline(mean, color=WIN, lw=1.5, ls="--", label=f"mean {usd(mean)}")
    money_axis(a1, "x")
    a1.set_yscale("log")
    a1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a1.yaxis.set_minor_formatter(mtick.NullFormatter())
    a1.legend()
    a1.set_xlabel("180-day contribution per FTD (clipped at -$400 and $3,000)")
    a1.set_ylabel("FTDs (log scale)")
    a1.set_title("Distribution of 180-day contribution")
    h = feats.handle_14.clip(lower=1)
    a2.hist(np.log10(h), bins=50, color=AMBER, alpha=0.85)
    a2.set_xticks([0, 1, 2, 3, 4, 5], ["$1", "$10", "$100", "$1k", "$10k", "$100k"])
    a2.set_xlabel("Handle in first 14 days (log scale)")
    a2.set_ylabel("FTDs")
    a2.set_title("Distribution of early handle")
    fig.tight_layout()
    return save(fig, "03a_distributions.png")


def fig_payback(curves: pd.DataFrame, ch: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    months = np.arange(1, 13)
    # End-of-line labels, pushed apart so close finishes don't overlap.
    ends = sorted(((curves[curves.channel == c].sort_values("life_month").cum_contribution_per_ftd.iloc[-1], c)
                   for c in ch.channel))
    span = max(e for e, _ in ends) - min(e for e, _ in ends)
    gap, placed = max(span, 400) * 0.085, []
    for y_end, c in ends:
        y_lab = max(y_end, placed[-1][0] + gap) if placed else y_end
        placed.append((y_lab, c))
    label_y = {c: y for y, c in placed}
    for c in ch.channel:
        g = curves[curves.channel == c].sort_values("life_month")
        y, cac, col = g.cum_contribution_per_ftd.to_numpy(), g.cac_per_ftd.iloc[0], CH_COLORS[c]
        ax.plot(months, y, color=col, lw=2.2)
        ax.hlines(cac, 1, 12, color=col, lw=1.1, ls=":")
        hit = np.argmax(y >= cac) if (y >= cac).any() else None
        if hit is not None:
            ax.scatter([months[hit]], [y[hit]], s=45, facecolor="white", edgecolor=col, lw=2, zorder=3)
        ax.annotate(nm(c), (12.15, label_y[c]), va="center", color=col, fontsize=9.5, annotation_clip=False)
    money_axis(ax)
    ax.set_xticks(months)
    ax.set_xlim(0.7, 13.6)
    ax.set_xlabel("Month since first deposit")
    ax.set_ylabel("Cumulative contribution per FTD")
    ax.set_title("Payback by channel (dotted line = CAC, circle = payback month)")
    return save(fig, "04_payback.png")


def fig_matrix(mx: pd.DataFrame, ch_order: list[str]) -> str:
    offers = ["no_sweat_1000", "deposit_match_250", "bet5_get200"]
    z = np.array([[mx[(mx.channel == c) & (mx.offer == o)].ltv_to_cac.iloc[0] for o in offers] for c in ch_order])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    im = ax.imshow(z, cmap="YlGn", vmin=0.8, vmax=max(2.5, z.max()), aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i, j]:.2f}×", ha="center", va="center", fontsize=10, color="white" if z[i, j] > 3.2 else INK)
    ax.set_xticks(range(3), [nm(o) for o in offers])
    ax.set_yticks(range(len(ch_order)), [nm(c) for c in ch_order])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("12-month LTV / CAC by channel and welcome offer")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    return save(fig, "05_channel_offer.png")


def fig_states(st: pd.DataFrame) -> str:
    s = st.sort_values(["tax_rate", "state"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(s.state, s.contribution_per_ftd, color=[LOSS if t >= 0.3 else ACCENT for t in s.tax_rate], width=0.65)
    for x, (v, t) in enumerate(zip(s.contribution_per_ftd, s.tax_rate)):
        ax.text(x, v + 12, pct(t, 1 if t * 100 % 1 else 0), ha="center", fontsize=9, color=INK2)
    money_axis(ax)
    ax.set_ylim(0, s.contribution_per_ftd.max() * 1.15)
    ax.set_ylabel("12-month contribution per FTD")
    ax.set_title("Same betting, different state: contribution per FTD (labels = tax rate)")
    return save(fig, "06_states.png")


def fig_retention(ret: pd.DataFrame) -> str:
    h = ret.pivot(index="cohort_month", columns="life_month", values="active_rate")
    h = h.loc[:, [c for c in h.columns if c >= 1 and c <= 18]]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(h.to_numpy(), cmap="Greens", vmin=0, vmax=0.7, aspect="auto")
    ax.set_yticks(range(len(h)), [pd.Timestamp(d).strftime("%b %Y") for d in h.index])
    ax.set_xticks(range(h.shape[1]), [int(c) + 1 for c in h.columns])
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.iat[i, j]
            if not np.isnan(v) and j % 3 == 0:
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7.5, color="white" if v > 0.45 else INK)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Month since first deposit (month 1 = first 30 days, always 100%)")
    ax.set_title("Share of each signup cohort still betting, months 2 to 19")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, format=mtick.PercentFormatter(1, decimals=0))
    return save(fig, "07_retention_cohorts.png")


def fig_retention_split(ret_ch: pd.DataFrame, cv: pd.DataFrame, ch_order: list[str]) -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for c in ch_order:
        g = ret_ch[ret_ch.channel == c].sort_values("life_month")
        a1.plot(g.life_month + 1, g.active_rate, color=CH_COLORS[c], lw=2, label=nm(c))
    a1.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    a1.set_xticks(range(1, 13)); a1.set_xlabel("Month since first deposit"); a1.set_ylim(0, 1.02)
    a1.set_title("Still betting, by channel"); a1.legend(fontsize=8.5)
    labels = [pd.Timestamp(d).strftime("%b\n%y") for d in cv.cohort_month]
    a2.bar(range(len(cv)), cv.ftds, color="#C9D5E3")
    a2.set_ylabel("FTDs"); a2.set_xticks(range(len(cv)), labels, fontsize=8)
    b = a2.twinx()
    b.plot(range(len(cv)), cv.ltv_6m, color=WIN, lw=2.2, marker="o", ms=4)
    money_axis(b); b.grid(False); b.spines["right"].set_visible(True); b.set_ylabel("6-month contribution per FTD")
    b.set_ylim(0, cv.ltv_6m.max() * 1.2)
    a2.set_title("Signup volume (bars) and 6-month value (line)")
    fig.tight_layout()
    return save(fig, "08_retention_channel_cohort.png")


def fig_deciles(model: dict) -> str:
    dec = pd.DataFrame(model["value"]["deciles"])
    imp = pd.DataFrame(model["value"]["importance"])
    imp = imp[imp.importance > 0.001].iloc[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    a1.bar(dec.decile, dec.actual, color=[LOSS if v < 0 else WIN for v in dec.actual], label="Actual")
    a1.plot(dec.decile, dec.predicted, color=INK, marker="o", ms=4, lw=1.6, label="Predicted")
    money_axis(a1); a1.set_xticks(dec.decile); a1.set_xlabel("Predicted decile (1 = most valuable)")
    a1.set_ylabel("180-day contribution per player"); a1.legend(); a1.set_title("Predicted vs actual, test set")
    a2.barh(imp.feature, imp.importance, color=ACCENT)
    a2.set_xlabel("Drop in test R² when shuffled"); a2.set_title("Permutation importance"); a2.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, "09_model.png")


def fig_tabpfn(alt: dict) -> str:
    models = alt["models"]
    names = [m["name"].replace(", ", "\n") for m in models]
    cols = [ACCENT, "#9AA7B4", INK2]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, key, title, fmt in (
        (axes[0], "auc", "Month-3 retention, AUC", "{:.3f}"),
        (axes[1], "spearman", "180-day value, rank correlation", "{:.3f}"),
        (axes[2], "hidden_high_value_share_top_decile", "True high-value share of top decile", "{:.0%}"),
    ):
        vals = [m[key] for m in models]
        # Dot plot rather than bars: the axis is zoomed in, and bars on a truncated axis exaggerate gaps.
        ax.scatter(range(3), vals, s=110, color=cols, zorder=3)
        for i, val in enumerate(vals):
            ax.annotate(fmt.format(val), (i, val), xytext=(0, 10), textcoords="offset points", ha="center",
                        fontsize=9.5, color=INK)
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.9, 0.01)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.6, 2.6)
        ax.set_xticks(range(3), names, fontsize=8)
        if key == "hidden_high_value_share_top_decile":
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return save(fig, "10_tabpfn.png")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def main() -> None:
    ch = pd.read_csv(OUT / "04_channel_economics.csv")
    offers = pd.read_csv(OUT / "05_offer_economics.csv")
    states = pd.read_csv(OUT / "06_state_economics.csv")
    curves = pd.read_csv(OUT / "03_ltv_curves.csv")
    ret = pd.read_csv(OUT / "02_cohort_retention.csv")
    wf = pd.read_csv(OUT / "09_revenue_waterfall.csv")
    conc = pd.read_csv(OUT / "10_value_concentration.csv")
    mx = pd.read_csv(OUT / "11_channel_offer_matrix.csv")
    bid = pd.read_csv(OUT / "12_breakeven_cac.csv")
    cv = pd.read_csv(OUT / "13_cohort_value.csv")
    ret_ch = pd.read_csv(OUT / "14_retention_by_channel.csv")
    seg = pd.read_csv(OUT / "15_day14_segments.csv")
    boot = pd.read_csv(OUT / "16_channel_bootstrap.csv").set_index("channel")
    feats = pd.read_parquet(OUT / "features.parquet")
    model = json.loads((OUT / "model_results.json").read_text())
    alt_path = OUT / "model_alternatives.json"
    alt = json.loads(alt_path.read_text()) if alt_path.exists() else None
    ny = pd.read_csv(REAL / "ny_draftkings_monthly.csv", parse_dates=["month"])
    seas = pd.read_csv(REAL / "ny_seasonality.csv")
    act = pd.read_parquet(ROOT / "data" / "activity.parquet", columns=["activity_date", "handle", "ggr"])

    ch_order = ch.channel.tolist()  # best to worst LTV/CAC
    w = ch.ftds
    blended_cac = float((ch.cac * w).sum() / w.sum())
    blended_ltv = float((ch.ltv_12m * w).sum() / w.sum())
    net_neg = float((feats.contribution_180 < 0).mean())
    top = lambda p: float(conc.loc[conc.pctile == p, "cum_share_of_total"].iloc[0])  # noqa: E731
    top_all = int(conc.loc[conc.cum_share_of_total >= 1, "pctile"].iloc[0])

    act["m"] = pd.to_datetime(act.activity_date).dt.to_period("M")
    sm = act.groupby("m")[["handle", "ggr"]].sum()
    sim_hold, sim_hold_sd = float(sm.ggr.sum() / sm.handle.sum()), float((sm.ggr / sm.handle).std())
    recent = ny[ny.month >= "2024-09-01"]
    real_hold, real_hold_sd = float(recent.ggr.sum() / recent.handle.sum()), float(recent.hold.std())
    last12 = ny.tail(12)

    best, worst = ch.iloc[0], ch.iloc[-1]
    wb = boot.loc[worst.channel]
    mxv = lambda c, o: float(mx[(mx.channel == c) & (mx.offer == o)].ltv_to_cac.iloc[0])  # noqa: E731
    off = offers.set_index("offer")
    st = states.set_index("state")
    seg_top, seg_bot = seg.iloc[-1], seg.iloc[0]
    months_of = pd.to_datetime(cv.cohort_month).dt.month
    fall, spring = cv[months_of.isin([9, 10, 11, 12, 1])], cv[months_of.isin([2, 3, 4, 5, 6, 7])]
    fall_v = float((fall.ltv_6m * fall.ftds).sum() / fall.ftds.sum())
    spring_v = float((spring.ltv_6m * spring.ftds).sum() / spring.ftds.sum())
    v, r = model["value"], model["retention"]
    bet5_worst_everywhere = all(
        mxv(c, "bet5_get200") <= min(mxv(c, "no_sweat_1000"), mxv(c, "deposit_match_250")) for c in ch_order
    )

    figs = {
        "real": fig_real(ny, seas),
        "waterfall": fig_waterfall(wf),
        "conc": fig_concentration(conc),
        "dist": fig_distributions(feats),
        "payback": fig_payback(curves, ch),
        "matrix": fig_matrix(mx, ch_order),
        "states": fig_states(states),
        "ret": fig_retention(ret),
        "ret2": fig_retention_split(ret_ch, cv, ch_order),
        "model": fig_deciles(model),
    }
    if alt:
        figs["tabpfn"] = fig_tabpfn(alt)

    md: list[str] = []
    add = md.append
    m2s = ret[ret.life_month == 1].set_index("cohort_month").active_rate
    m2_hi, m2_lo = m2s.idxmax(), m2s.idxmin()
    m2 = ret[ret.life_month == 1].active_rate.mean()
    m12 = ret[ret.life_month == 11].active_rate.mean()
    n_12m = int(ch.ftds.sum())
    pm_summary = pd.read_csv(OUT / "01_player_month.csv").iloc[0]
    c180 = feats.contribution_180
    q = c180.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    skew = float(c180.skew())
    num_feats = ["active_days_14", "active_days_wk2", "bets_14", "handle_14", "ggr_14", "max_day_handle_14",
                 "avg_stake_14", "days_idle_at_d14", "welcome_promo_cost"]
    feat_labels = {
        "active_days_14": "Active days, days 0 to 13", "active_days_wk2": "Active days, days 7 to 13",
        "bets_14": "Bets placed", "handle_14": "Handle (total staked)", "ggr_14": "GGR (book's win)",
        "max_day_handle_14": "Largest single-day handle", "avg_stake_14": "Average stake per bet",
        "days_idle_at_d14": "Days since last bet, at day 13", "welcome_promo_cost": "Welcome offer cost",
    }
    corr = pd.DataFrame([{
        "Feature": feat_labels[f_],
        "Spearman with 180-day contribution": f"{feats[f_].corr(feats.contribution_180, method='spearman'):.2f}",
        "Spearman with active in month 3": f"{feats[f_].corr(feats.active_m3, method='spearman'):.2f}",
    } for f_ in num_feats])
    tlog = (ROOT / "tabpfn.log").read_text(errors="ignore") if (ROOT / "tabpfn.log").exists() else ""
    import re as _re
    secs = {k: int(v) for k, v in _re.findall(r"TabPFN (retention|value): 6,754/6,754\s+\((\d+)s\)", tlog)}

    # ---- Title and abstract -----------------------------------------------
    add("# Sportsbook Customer Economics\n")
    add("Robert Carrington · September 2026\n")
    add("## Abstract\n")
    add(
        f"This report estimates what a sportsbook's first-time depositors (FTDs) cost to acquire, what they contribute in "
        f"their first year, and how early their value can be predicted. The player-level data is a simulation of "
        f"{len(feats):,} FTDs whose seasonality, hold, and hold volatility are calibrated to DraftKings' public monthly "
        f"filings in New York. Year-one contribution averages {usd(blended_ltv)} per FTD against a blended CAC of "
        f"{usd(blended_cac)}, a {blended_ltv / blended_cac:.2f}× return, but value is highly concentrated: the top 1% of "
        f"FTDs produce {pct(top(1))} of it and {pct(net_neg)} are net negative at day 180. Channel, welcome offer, and state "
        f"tax each move payback materially. A gradient boosting model on the first 14 days of activity ranks players' "
        f"180-day value with a Spearman correlation of {v['spearman_model']:.2f}, against {v['spearman_heuristic']:.2f} "
        f"for sorting by early handle alone."
        + (f" A separate section tests TabPFN, a pretrained tabular foundation model. Given {alt['n_context']:,} training "
           f"players and no tuning, it ranks 180-day value at {alt['models'][0]['spearman']:.2f} Spearman, better than "
           f"gradient boosting trained on all {int(alt['models'][2]['name'].split(', ')[1].split()[0].replace(',', '')):,}, "
           f"and comes within {abs(alt['models'][0]['auc'] - alt['models'][2]['auc']):.3f} AUC of it on retention."
           if alt and alt["models"][0]["spearman"] > alt["models"][2]["spearman"] else
           f" A separate section tests TabPFN, a pretrained tabular foundation model, which reaches "
           f"{alt['models'][0]['auc']:.3f} AUC on month-3 retention from {alt['n_context']:,} training players with no tuning."
           if alt else "")
        + "\n"
    )
    add(
        "I built this for the Analyst I, Customer Economics role at DraftKings. None of it uses DraftKings internal data, "
        "and where a result follows directly from an assumption rather than from the analysis, I say so.\n"
    )
    add(table(pd.DataFrame([
        ["Simulated FTDs", f"{len(feats):,}"],
        ["Blended CAC", usd(blended_cac)],
        ["12-month contribution per FTD", usd(blended_ltv)],
        ["12-month LTV / CAC", f"{blended_ltv / blended_cac:.2f}×"],
        ["Share of FTDs net negative at day 180", pct(net_neg)],
        ["Share of year-one contribution from the top 1%", pct(top(1))],
    ], columns=["Headline", "Value"]), "lr") + "\n")

    add("## Contents\n")
    add("\n".join([
        "1. [Objective](#1-objective)", "2. [Data collection](#2-data-collection)",
        "3. [Data cleaning and transformation](#3-data-cleaning-and-transformation)",
        "4. [Exploratory data analysis](#4-exploratory-data-analysis)", "5. [Feature engineering](#5-feature-engineering)",
        "6. [Modeling choices](#6-modeling-choices)", "7. [Findings](#7-findings)", "8. [Conclusion](#8-conclusion)",
        "9. [TabPFN: process and comparison](#9-tabpfn-process-and-comparison)",
        "10. [Reproducibility](#10-reproducibility)",
    ]) + "\n")

    # ---- 1. Objective -------------------------------------------------------
    add("## 1. Objective\n")
    add("A customer economics team decides how much to pay for a new customer and where to spend to keep them. This analysis answers four questions:\n")
    add("\n".join([
        "1. What is a first-time depositor worth over 12 months, after promos, tax, and variable cost?",
        "2. How do acquisition channel, welcome offer, and state change that value and the payback period?",
        "3. How do retention and value evolve by cohort?",
        "4. How well can a player's value be predicted from their first 14 days, and which model should do it?",
    ]) + "\n")

    # ---- 2. Data collection -----------------------------------------------
    add("## 2. Data collection\n")
    add("### 2.1 Real data: DraftKings' New York filings\n")
    add(
        f"The New York State Gaming Commission publishes each mobile sports operator's monthly handle (total amount wagered) "
        f"and gross gaming revenue (GGR, the amount the book keeps). I downloaded DraftKings' report, a PDF covering "
        f"{ny.month.min():%B %Y} through {ny.month.max():%B %Y}, {len(ny)} months in all. Over the last 12 months DraftKings "
        f"took ${last12.handle.sum() / 1e9:.2f}B in New York handle at a {pct(last12.ggr.sum() / last12.handle.sum(), 1)} "
        f"hold, and New York taxes GGR at 51%.\n"
    )
    add(f"![DraftKings New York handle, hold, and seasonality]({figs['real']})\n")
    add("### 2.2 Simulated player-level data\n")
    add(
        "No public dataset has sportsbook customers with acquisition channel, CAC, and promo cost, which is the core of "
        "customer economics. The one academic dataset of real bettors (the Transparency Project's bwin data) is licensed "
        "for non-commercial research only. So I simulated the player level, and used the real filings to calibrate it.\n"
    )
    add(
        f"`simulate.py` generates {len(feats):,} FTDs who signed up between September 2024 and August 2025 and records every "
        f"day each one bet through June 2026, {len(act):,} player-days in total. Each FTD has:\n"
    )
    add("\n".join([
        "- an acquisition channel (referral, TV and brand, search, affiliate, paid social), each with its own CAC",
        "- a welcome offer (no-sweat first bet, deposit match, or bet $5 get $200), each with its own expected cost",
        "- a state (nine, each with its tax rate)",
        "- a hidden player type (bonus hunter, casual, regular, high value) that sets churn, betting frequency, stake size, "
        "odds preference, and how much of the welcome offer the player extracts",
    ]) + "\n")
    add(
        "Channels and offers shift the mix of player types. That is the main assumption driving channel and offer results. "
        "The analysis never sees player type. It works only from behavior, as it would on real data. Type is used once, at "
        "the end, to check whether the models find the right players.\n"
    )
    add("**What is calibrated to real data, and how close the simulation lands:**\n")
    add(table(pd.DataFrame([
        ["Hold, Sep 2024 onward", pct(real_hold, 1), pct(sim_hold, 1)],
        ["Monthly hold, standard deviation", f"{real_hold_sd * 100:.2f} pts", f"{sim_hold_sd * 100:.2f} pts"],
        ["Seasonal pattern of activity", "Index by calendar month (2.1)", "Same index, used as input"],
        ["New York tax rate", "51% of GGR", "51%"],
    ], columns=["Measure", "DraftKings NY (real)", "Simulation"]), "lrr") + "\n")
    add(
        "Hold swings month to month because every customer bets on the same games, so outcomes are correlated. The "
        "simulation adds one shared shock per month, sized to the real standard deviation. Channel CACs, offer costs, "
        "player-type behavior, and the non-New York tax rates are my assumptions, all listed in `assumptions.toml`.\n"
    )

    # ---- 3. Cleaning and transformation -----------------------------------
    add("## 3. Data cleaning and transformation\n")
    add("### 3.1 Regulator filings\n")
    add("\n".join([
        "- Extracted the monthly rows from each page of the PDF with `pdfplumber` and a regular expression, then dropped "
        "the pre-launch months reported as $0.",
        "- De-duplicated months that appear on more than one fiscal-year page.",
        "- Computed hold as GGR divided by handle, and checked the state's share of GGR against the statutory 51% "
        "(it matches to three decimals every month).",
        "- Removed market growth before measuring seasonality: each month's handle was divided by a centered 12-month "
        "moving average, and the ratios were averaged by calendar month and scaled to a mean of 1.",
    ]) + "\n")
    add("### 3.2 Player data\n")
    add(
        f"All transformation is in SQL (DuckDB), in `sql/01` through `sql/15`. The central table is a **player-month "
        f"panel**: one row per FTD per 30-day \"life month\" since first deposit, {int(pm_summary.player_months):,} rows in "
        f"all. Three decisions shape it:\n"
    )
    add("\n".join([
        "- **Zero months are kept.** A month with no bets is a row with zeros, not a missing row. Averages are per FTD, "
        "not per active player, so retention and value are not overstated.",
        "- **Right censoring is handled by truncation.** Each player is cut at their last fully observed month. Any "
        f"12-month figure uses only players with 12 full months, which is {n_12m:,} of the {len(feats):,} FTDs. Younger "
        "cohorts never pull a curve down.",
        "- **Life months, not calendar months.** Month 1 is each player's first 30 days, so cohorts that signed up at "
        "different times are compared at the same age.",
    ]) + "\n")
    add("**Contribution** is the value measure throughout:\n")
    add("```\ncontribution = GGR - welcome promo - ongoing promos - tax x (GGR - promos) - 0.6% x handle\n```\n")
    add(
        "It is a contribution margin, not profit: it excludes fixed costs and overhead. Tax is applied to GGR net of promos "
        "as a simplification (section 8.3).\n"
    )

    # ---- 4. EDA -------------------------------------------------------------
    add("## 4. Exploratory data analysis\n")
    add("### 4.1 Where revenue goes\n")
    add(f"![Revenue waterfall]({figs['waterfall']})\n")
    ggr = float(wf.per_ftd.iloc[0])
    promo = float(-(wf.set_index("step").loc[["Welcome offer", "Ongoing promos"], "per_ftd"].sum()))
    add(
        f"Of {usd(ggr)} in year-one GGR per FTD, promos take {usd(promo)} ({pct(promo / ggr)}) and tax "
        f"{usd(-wf.set_index('step').loc['Gaming tax', 'per_ftd'])}. Contribution is {usd(wf.per_ftd.iloc[-1])}, "
        f"{pct(wf.per_ftd.iloc[-1] / ggr)} of GGR.\n"
    )
    add("### 4.2 The distribution of player value\n")
    add(f"![Distribution of player value]({figs['dist']})\n")
    add(table(pd.DataFrame([[
        usd(c180.mean()), usd(q[0.5]), usd(q[0.1]), usd(q[0.25]), usd(q[0.75]), usd(q[0.9]), usd(q[0.99]), f"{skew:.1f}",
    ]], columns=["Mean", "Median", "10th pct", "25th pct", "75th pct", "90th pct", "99th pct", "Skewness"]),
        "rrrrrrrr") + "\n")
    add(
        f"180-day contribution is extremely right-skewed. The mean is {usd(c180.mean())} while the median is "
        f"{usd(q[0.5])}, and {pct(net_neg)} of FTDs are below zero, mostly because the welcome offer costs more than the book "
        f"ever wins back from them. This shapes every later choice: averages are fragile, so channel results get bootstrap intervals, "
        f"and model evaluation leans on rank-based metrics rather than squared error.\n"
    )
    add(f"![Value concentration]({figs['conc']})\n")
    add(table(pd.DataFrame(
        [[f"Top {p}%", pct(top(p)), usd(conc.loc[conc.pctile <= p, "mean_contribution"].mean())] for p in (1, 5, 10, 20)],
        columns=["Players", "Share of year-one contribution", "Average contribution"]), "lrr") + "\n")
    add(f"The top {top_all}% of players account for all year-one contribution. The share passes 100% because the rest are net negative in aggregate.\n")
    add("### 4.3 Retention\n")
    add(f"![Cohort retention]({figs['ret']})\n")
    add(
        f"On average {pct(1 - m2)} of FTDs place no bet in their second month and {pct(m12)} are still betting in month 12. "
        f"Seasonality shows up in the second month: {pct(m2s[m2_hi])} of {pd.Timestamp(m2_hi):%B %Y} signups bet again, "
        f"against {pct(m2s[m2_lo])} of {pd.Timestamp(m2_lo):%B %Y} signups, whose second month is July.\n"
    )
    add(f"![Retention by channel and cohort value]({figs['ret2']})\n")
    add(
        f"FTDs who sign up between September and January are worth {usd(fall_v)} over six months, against "
        f"{usd(spring_v)} for February through July signups, whose early months overlap the summer lull.\n"
    )
    add("### 4.4 Early behavior against later value\n")
    t = pd.DataFrame([{
        "Handle, days 0 to 13": s_.handle_band.split("  ", 1)[1], "Share of FTDs": pct(s_.share_of_ftds, 1),
        "Avg active days": f"{s_.avg_active_days_14:.1f}", "Active month 3": pct(s_.active_m3_rate),
        "Mean 180-day contribution": usd(s_.mean_contribution_180), "Share profitable": pct(s_.share_profitable_180),
        "Share of total 180-day": pct(s_.share_of_total_180),
    } for s_ in seg.itertuples()])
    add(table(t, "lrrrrrr") + "\n")
    add(
        f"Early handle separates players sharply. The {pct(seg_top.share_of_ftds, 1)} of FTDs who stake $5,000 or more in "
        f"their first two weeks produce {pct(seg_top.share_of_total_180)} of 180-day contribution. This is the baseline any "
        f"model has to beat.\n"
    )

    # ---- 5. Feature engineering -------------------------------------------
    add("## 5. Feature engineering\n")
    add(
        "The prediction point is **day 14**: a model scores each FTD using only what is known by the end of their 14th day "
        "(days 0 to 13). Features are built in `sql/08_early_features.sql`.\n"
    )
    add("**Targets**\n")
    add("\n".join([
        "- `active_m3`: whether the player bets at all in life month 3 (days 60 to 89). Binary, "
        f"{pct(feats.active_m3.mean())} positive.",
        "- `contribution_180`: total contribution over days 0 to 179. It includes the first 14 days by design, because the "
        "business question is the player's total value. The model's real work is the remaining 166 days.",
    ]) + "\n")
    add("**Features**, with their rank correlation to each target\n")
    add(table(corr, "lrr") + "\n")
    add("\n".join([
        "- Categorical context: acquisition channel, welcome offer, state, and signup month (for seasonality).",
        "- Week 2 activity and days idle are included because the *trend* of engagement matters: a player who bet heavily on "
        "day 1 and then stopped differs from one still betting on day 13.",
        "- Largest single-day handle is included alongside total handle because a few large days signal a high-stakes "
        "player differently from many small ones.",
        "- **Leakage check:** no feature uses anything after day 13, and nothing downstream of the outcome (such as churn "
        "date or lifetime) is available to the model.",
    ]) + "\n")
    add(
        "Transformations depend on the model. For logistic regression, numeric features get a signed log transform, "
        "`sign(x) * log(1 + |x|)`, to tame the heavy tails (GGR can be negative), then standardization, and categoricals "
        "are one-hot encoded. Gradient boosting takes raw values and native categorical splits, since trees are invariant "
        "to monotone transforms.\n"
    )

    # ---- 6. Modeling choices ----------------------------------------------
    add("## 6. Modeling choices\n")
    add("\n".join([
        f"- **Out-of-time split.** Train on {model['split']['train_cohorts']} signups ({model['split']['n_train']:,} players), "
        f"test on {model['split']['test_cohorts']} ({model['split']['n_test']:,}). A random split would leak seasonal "
        f"information across the boundary, and in practice the model is fit on past cohorts and used on new ones.",
        "- **Baselines first.** Logistic regression for retention, and two naive rules for value: sort by 14-day handle, and "
        "predict the training mean.",
        "- **Gradient boosting** (scikit-learn `HistGradientBoosting`) as the main model. It handles missing values, "
        "categoricals, and interactions without manual work, and is the standard strong baseline for tabular data. Early "
        "stopping on an internal validation split, learning rate 0.05.",
        "- **Metrics chosen for the skew.** AUC for retention (ranking quality, insensitive to the threshold). For value, "
        "Spearman rank correlation and top-decile lift, because the business action is ranking players and a few whales "
        "dominate any squared-error metric. Mean absolute error is reported for completeness.",
        "- **Permutation importance** on the test set, which measures how much accuracy drops when a feature is shuffled "
        "and is less biased toward high-cardinality features than split-based importance.",
        "- **Bootstrap intervals** for channel LTV/CAC: 2,000 resamples of players within each channel, 95% percentile "
        "intervals. Given the skew in 4.2, point estimates alone would overstate precision.",
    ]) + "\n")

    # ---- 7. Findings --------------------------------------------------------
    add("## 7. Findings\n")
    add("### 7.1 Model performance\n")
    add(f"![Model results]({figs['model']})\n")
    td, th = v["top_decile_model"], v["top_decile_heuristic"]
    add(table(pd.DataFrame([
        ["Bets in month 3?", "AUC", f"{r['auc_gbm']:.3f}", f"{r['auc_logistic']:.3f} (logistic regression)"],
        ["Flag the at-risk decile", "Share who lapse", pct(r["lapse_rate_bottom_decile"]), f"{pct(r['lapse_rate_all'])} (all players)"],
        ["Rank by 180-day value", "Spearman", f"{v['spearman_model']:.3f}", f"{v['spearman_heuristic']:.3f} (sort by 14-day handle)"],
        ["Find the top decile", "Lift over average", f"{td['lift']:.2f}×", f"{th['lift']:.2f}× (sort by 14-day handle)"],
        ["Predict dollar value", "Mean absolute error", usd(v["mae_model"]), f"{usd(v['mae_predict_mean'])} (predict the mean)"],
        ["Find hidden high-value players", "Share of top decile", pct(v["hidden_high_value_share_top_decile"]), f"{pct(v['hidden_high_value_share_all'])} (base rate)"],
    ], columns=["Question", "Metric", "Gradient boosting", "Baseline"]), "llrl") + "\n")
    top_pred = v["deciles"][0]
    add("\n".join([
        f"- On retention, logistic regression matches gradient boosting ({r['auc_logistic']:.3f} vs {r['auc_gbm']:.3f} "
        f"AUC). The simpler model is the one to ship there.",
        f"- On value, gradient boosting ranks the whole base better than early handle alone ({v['spearman_model']:.2f} vs "
        f"{v['spearman_heuristic']:.2f}), but ties it on the top decile. Use the model for bids and retention spend across "
        f"everyone, and a handle threshold for routing likely high-value players to VIP.",
        f"- The value model is biased low at the top: it predicts {usd(top_pred['predicted'])} for its top decile, which "
        f"actually averages {usd(top_pred['actual'])}. Squared-error loss shrinks a heavy right tail. Modeling log value or "
        f"recalibrating the top decile would fix it before anyone sets bids from it.",
        f"- {pct(v['hidden_high_value_share_top_decile'])} of the model's top decile are truly high-value players, against a "
        f"{pct(v['hidden_high_value_share_all'])} base rate, which confirms it is finding the right people rather than "
        f"fitting noise.",
    ]) + "\n")
    add("### 7.2 Channel economics\n")
    add(f"![Payback by channel]({figs['payback']})\n")
    t = pd.DataFrame([{
        "Channel": nm(c.channel), "FTDs": f"{c.ftds:,}", "CAC": usd(c.cac), "12-mo LTV": usd(c.ltv_12m),
        "Median LTV": usd(c.median_ltv_12m), "LTV / CAC": f"{c.ltv_to_cac:.2f}×",
        "95% interval": f"{boot.loc[c.channel, 'ratio_lo']:.2f} to {boot.loc[c.channel, 'ratio_hi']:.2f}",
        "Active month 3": pct(c.active_m3_rate),
        "Payback": "none in 12 mo" if pd.isna(c.payback_month) else f"month {int(c.payback_month) + 1}",
    } for c in ch.itertuples()])
    add(table(t, "lrrrrrrrr") + "\n")
    add(
        f"Every channel pays back inside a year, from month {int(best.payback_month) + 1} for {nm(best.channel).lower()} to "
        f"month {int(worst.payback_month) + 1} for {nm(worst.channel).lower()}, and every interval stays above 1.0×. The "
        f"median FTD is net negative in every channel. Channels differ in how many high-value players they bring, not in "
        f"how the typical player behaves.\n"
        if not ch.payback_month.isna().any() else
        "The median FTD is net negative in every channel. Channels differ in how many high-value players they bring, not "
        "in how the typical player behaves.\n"
    )
    add("### 7.3 Channel and offer\n")
    add(f"![Channel by offer]({figs['matrix']})\n")
    t = pd.DataFrame([{
        "Offer": nm(o.offer), "FTDs": f"{o.ftds:,}", "Offer cost per FTD": usd(o.welcome_cost_per_ftd),
        "Active month 3": pct(o.active_m3_rate), "12-mo LTV": usd(o.ltv_12m), "LTV / CAC": f"{o.ltv_to_cac:.2f}×",
    } for o in offers.itertuples()])
    add(table(t, "lrrrrr") + "\n")
    add(
        f"The channel matters most, but within a channel the offer still moves the return: {nm(worst.channel).lower()} goes "
        f"from {mxv(worst.channel, 'bet5_get200'):.2f}× with bet $5 get $200 to {mxv(worst.channel, 'no_sweat_1000'):.2f}× "
        f"with the no-sweat offer. The direction of this result is an assumption (I assumed the richest offer draws more "
        f"bonus hunters). The size of the gap, net of offer cost, is the output.\n"
    )
    add("### 7.4 State tax and bid caps\n")
    add(f"![Contribution by state]({figs['states']})\n")
    add(
        f"A New York FTD contributes {usd(st.loc['NY', 'contribution_per_ftd'])} in year one at 51% tax, against "
        f"{usd(st.loc['MI', 'contribution_per_ftd'])} in Michigan at 8.4%. That gap is arithmetic, but it implies bid caps "
        f"should be set by state. The table gives the highest CAC that still pays back inside 12 months for each channel "
        f"and state. **Bold** marks cells where the channel's current CAC is above the cap. Cells hold roughly 300 to "
        f"1,600 FTDs, so small differences are noise.\n"
    )
    piv = bid.pivot(index="channel", columns="state", values="breakeven_cac")
    cur = bid.pivot(index="channel", columns="state", values="current_cac")
    order_states = states.sort_values(["tax_rate", "state"], ascending=[False, True]).state.tolist()
    rows = []
    for c in ch_order:
        row = {"Channel": f"{nm(c)} (CAC {usd(ch.set_index('channel').loc[c, 'cac'])})"}
        for s_ in order_states:
            val = usd(piv.loc[c, s_])
            row[f"{s_} {pct(st.loc[s_, 'tax_rate'], 1 if st.loc[s_, 'tax_rate'] * 100 % 1 else 0)}"] = (
                f"**{val}**" if cur.loc[c, s_] > piv.loc[c, s_] else val)
        rows.append(row)
    add(table(pd.DataFrame(rows), "l" + "r" * len(order_states)) + "\n")

    # ---- 8. Conclusion -----------------------------------------------------
    add("## 8. Conclusion\n")
    add("### 8.1 Summary\n")
    add(
        f"An FTD in this simulation returns {blended_ltv / blended_cac:.2f}× its acquisition cost in year one, but that "
        f"average hides extreme concentration: {pct(top(1))} of contribution comes from 1% of players and most players "
        f"lose money after promos. The biggest levers are which channel a player comes from, which offer they receive, "
        f"and which state they bet in. Two weeks of behavior is enough to rank players usefully, and a simple model on "
        f"those two weeks ranks them better than early handle alone. Section 9 tests a pretrained tabular model, TabPFN, "
        f"on the same problem.\n"
    )
    add("### 8.2 Recommendations\n")
    add("\n".join(f"{i}. {t_}" for i, t_ in enumerate([
        f"Test the no-sweat offer against bet $5 get $200 in {nm(worst.channel).lower()}, with a holdout, before changing "
        f"that channel's budget.",
        "Set acquisition bid caps by state (7.4) rather than one national CAC target.",
        "Score every FTD at day 14. Use the value model for retention spend across the whole base, and a handle threshold "
        "to route likely high-value players to VIP.",
        "Report channel LTV with an interval. With value this concentrated, a few players can move a channel's average.",
        "Use logistic regression for retention scoring: it matches gradient boosting and is easier to explain.",
    ], 1)) + "\n")
    add("### 8.3 Limitations\n")
    add("\n".join(f"- {t_}" for t_ in [
        "Player-level data is simulated. Channel, offer, and player-type effects come from my assumptions, so conclusions "
        "about them demonstrate the method rather than describe real DraftKings economics.",
        "Only seasonality, hold, hold volatility, and New York's tax rate are calibrated to real data, and only from New "
        "York, the highest-tax state.",
        "Tax is a flat rate on GGR net of promos. Real states tier it, tax per wager, or limit promo deductions, and the "
        "non-New York rates are approximate.",
        "No casino or daily fantasy cross-sell, and churned players never return.",
    ]) + "\n")

    # ---- 9. TabPFN ---------------------------------------------------------
    add("## 9. TabPFN: process and comparison\n")
    add("### 9.1 What TabPFN is\n")
    add(
        "TabPFN (Hollmann et al., *Nature*, 2025) is a transformer pretrained by Prior Labs on millions of synthetic "
        "datasets drawn from a prior over how tabular data is generated: random causal structures, noise, missing values, "
        "and mixed types. It does not fit parameters to a new dataset. The labeled training rows go into the model as "
        "context, and it predicts the unlabeled rows in a single forward pass, in effect performing approximate Bayesian "
        "inference learned during pretraining. There is no hyperparameter tuning.\n"
    )
    add(
        "The practical claim is strong accuracy on small and medium tables, where there is not enough data to train and "
        "tune a model well. That is a common situation in customer economics: a new state launch, a new promotion, or a "
        "new product, with a few thousand players and a decision due before more data arrives.\n"
    )
    add("### 9.2 Setup\n")
    add("\n".join([
        "- **Model version.** The open TabPFN v2 weights from Hugging Face (`Prior-Labs/TabPFN-v2-clf` and `-reg`), loaded "
        "through the `tabpfn` package. Newer versions (3.5) require an account and license acceptance for local use, so I "
        "used v2, which is ungated.",
        "- **Same problem as section 6.** Same features, same targets, same out-of-time test set of "
        f"{alt['n_test'] if alt else model['split']['n_test']:,} players, so results are directly comparable.",
        f"- **Context size.** {alt['n_context'] if alt else 3000:,} players sampled at random from the training cohorts. "
        "TabPFN v2 was pretrained on datasets up to about 10,000 rows, and inference cost grows with context size, so a "
        "smaller context kept the run feasible on a laptop CPU.",
        "- **Encoding.** Categorical columns were integer-coded and flagged as categorical so TabPFN applies its own "
        "categorical handling. Numeric features were passed raw: TabPFN does its own preprocessing internally, including "
        "transforms suited to skewed data.",
        f"- **Ensembling.** {alt['n_estimators'] if alt else 4} ensemble members, each with a different feature ordering and "
        "preprocessing, averaged.",
        "- **CPU.** TabPFN refuses CPU runs above 1,000 rows by default because they are slow, so I set "
        "`ignore_pretraining_limits=True` and predicted in batches of 1,000."
        + (f" Scoring the test set took about {secs.get('retention', 0) // 60} minutes for retention and "
           f"{secs.get('value', 0) // 60} minutes for value." if secs else ""),
    ]) + "\n")
    add(
        "To separate the effect of the model from the effect of less data, I compared three setups on the same test "
        "players: TabPFN on the 3,000-player sample, gradient boosting on the same 3,000, and gradient boosting on the full "
        "training set.\n"
    )
    add("### 9.3 Results\n")
    if alt:
        a0, a1, a2 = alt["models"]
        add(f"![TabPFN comparison]({figs['tabpfn']})\n")
        add(table(pd.DataFrame([{
            "Model": m_["name"], "Retention AUC": f"{m_['auc']:.3f}", "Value Spearman": f"{m_['spearman']:.3f}",
            "Value MAE": usd(m_["mae"]), "Top-decile lift": f"{m_['top_decile_lift']:.2f}×",
            "High-value share, top decile": pct(m_["hidden_high_value_share_top_decile"]),
        } for m_ in alt["models"]]), "lrrrrr") + "\n")
        same = a0["auc"] - a1["auc"]
        full = a0["auc"] - a2["auc"]
        n_full = int(a2["name"].split(", ")[1].split()[0].replace(",", ""))
        ratio = round(n_full / alt["n_context"])
        if full > 0:
            vs_full = f"TabPFN still comes out ahead by {full:.3f}"
        elif full > -0.01:
            vs_full = f"TabPFN is within {abs(full):.3f}"
        else:
            vs_full = f"TabPFN trails by {abs(full):.3f}"
        add("### 9.4 Interpretation\n")
        add("\n".join([
            f"- **At equal data**, TabPFN {'beats' if same > 0 else 'trails'} gradient boosting on retention by "
            f"{abs(same):.3f} AUC, and ranks 180-day value at {a0['spearman']:.3f} Spearman against {a1['spearman']:.3f}.",
            f"- **Against {ratio}× the data**, gradient boosting on {n_full:,} players: {vs_full} on retention AUC, and "
            f"scores {a0['spearman']:.3f} against {a2['spearman']:.3f} on value ranking"
            + (". On value ranking, the model with a tenth of the data is the best of the three." if a0["spearman"] > a2["spearman"] else "."),
            "- **Why value ranking goes this way.** Gradient boosting with squared-error loss spends its capacity fitting "
            f"the few very large players, and its ranking of everyone else does not improve with more data "
            f"({a1['spearman']:.3f} on {alt['n_context']:,} players, {a2['spearman']:.3f} on all of them). TabPFN predicts "
            "from a learned prior over tables rather than minimizing squared "
            "error on this one, which seems to make it less sensitive to the tail. Modeling log value with gradient boosting "
            "would be the fair next comparison."
            if a1["spearman"] > a2["spearman"] else
            "- **Value ranking.** TabPFN's advantage holds across the distribution, not just the top decile.",
            f"- **Finding the top players.** Top-decile lift is nearly identical across all three ({a0['top_decile_lift']:.2f}×, "
            f"{a1['top_decile_lift']:.2f}×, {a2['top_decile_lift']:.2f}×), and the full-data model recovers slightly more of the "
            f"hidden high-value players ({pct(a2['hidden_high_value_share_top_decile'])} vs "
            f"{pct(a0['hidden_high_value_share_top_decile'])}). TabPFN's edge is in ranking the middle of the base.",
            "- **Cost.** Gradient boosting trains and scores in seconds. TabPFN needed minutes per model on CPU, because every "
            "prediction attends over the whole context. At production scale it wants a GPU or Prior Labs' hosted API.",
            "- **Where I'd use it.** First, small-sample questions where there is no time or data to tune a model: early "
            "reads on a new state, offer, or product. Second, as a benchmark. Its value-ranking result here says the "
            "full-data gradient boosting model is leaving accuracy on the table, and the next step would be to fix that "
            "model's loss rather than assume more data solves it.",
        ]) + "\n")
    else:
        add("TabPFN results are not in `outputs/` yet. Run `python model_tabpfn.py`, then `python build_report.py`.\n")
    add(
        "TabPFN v2 weights are released under the Prior Labs License, which is Apache 2.0 with an attribution "
        "requirement. Built with TabPFN.\n"
    )

    # ---- 10. Reproducibility -----------------------------------------------
    add("## 10. Reproducibility\n")
    add("```bash\npython -m venv .venv\n.venv/Scripts/pip install -r requirements.txt\n.venv/Scripts/python run.py            # simulate, SQL, models, report (about 30 seconds)\n.venv/Scripts/python model_tabpfn.py   # TabPFN comparison (about 15 to 20 minutes on CPU)\n.venv/Scripts/python build_report.py   # rebuild this report with the TabPFN results\n```\n")
    add(
        "The pipeline is deterministic. One bug worth recording: results first changed between runs because DuckDB does "
        "not guarantee row order out of a parallel join, and row order decides which rows gradient boosting holds out for "
        "early stopping. Sorting the feature table fixed it.\n"
    )

    text = "\n".join(md)
    text = text.replace("—", ", ").replace("–", " to ")  # keep punctuation plain
    (ROOT / "REPORT.md").write_text(text, encoding="utf-8")
    print(f"wrote REPORT.md ({len(text.split()):,} words) and {len(figs)} figures")


if __name__ == "__main__":
    main()
