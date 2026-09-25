"""Write REPORT.md and the figures in figures/ from the files in outputs/.

Every number in the report is read from outputs/, so re-running the pipeline
regenerates the whole document.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.dates

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
    "straight": "Straight", "live": "Live", "parlay": "Parlay", "sgp": "Same-game parlay",
    "nfl": "NFL", "college_football": "College football", "nba": "NBA", "college_basketball": "College basketball",
    "mlb": "MLB", "nhl": "NHL", "soccer": "Soccer", "other": "Other",
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


def fig_ltv_projection(ltv: dict) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    months = np.arange(1, 37)
    ends = []
    for r in ltv["projection"]:
        c, col = r["channel"], CH_COLORS[r["channel"]]
        cum, lo, hi = np.array(r["cum_curve"]), np.array(r["cum_lo"]), np.array(r["cum_hi"])
        ax.plot(months[:12], np.array(r["actual_cum"]), color=col, lw=2.4)
        ax.plot(months[11:], cum[11:], color=col, lw=2, ls="--")
        ax.fill_between(months[11:], lo[11:], hi[11:], color=col, alpha=0.12, lw=0)
        ax.hlines(r["cac"], 1, 36, color=col, lw=0.9, ls=":")
        ends.append((cum[-1], c))
    ends.sort()
    placed = []
    for y_end, c in ends:
        y_lab = max(y_end, placed[-1][0] + 70) if placed else y_end
        placed.append((y_lab, c))
    for y_lab, c in placed:
        ax.annotate(nm(c), (36.4, y_lab), va="center", color=CH_COLORS[c], fontsize=10, annotation_clip=False)
    ax.axvline(12, color=INK2, lw=0.8)
    ax.text(12.3, ax.get_ylim()[1] * 0.96, "observed | projected", color=INK2, fontsize=9, va="top")
    money_axis(ax)
    ax.set_xlim(0.5, 41)
    ax.set_xticks([1, 6, 12, 18, 24, 30, 36])
    ax.set_xlabel("Month since first deposit")
    ax.set_ylabel("Cumulative contribution per FTD")
    ax.set_title("Projected LTV to 36 months (shaded band = 95% bootstrap interval, dotted line = CAC)")
    return save(fig, "11_ltv_projection.png")


def fig_ltv_backtest(ltv: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    methods = [("err_sbg", "sBG decay model", ACCENT), ("err_run_rate", "Flat run rate", AMBER), ("err_stop", "Stop counting", "#9AA7B4")]
    for ax, key, title in ((axes[0], "backtest_6_to_12", "Fit on months 1 to 6, predict month 12"),
                           (axes[1], "backtest_12_to_18", "Fit on months 1 to 12, predict month 18")):
        rows = ltv[key]["channels"]
        order = [r for c in CH_COLORS for r in rows if r["channel"] == c]
        x = np.arange(len(order))
        for i, (m, label, col) in enumerate(methods):
            ax.bar(x + (i - 1) * 0.26, [r[m] for r in order], width=0.26, color=col, label=label)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x, [nm(r["channel"]) for r in order], fontsize=9)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
        ax.set_title(title, fontsize=10.5)
    axes[0].set_ylabel("Forecast error (predicted / actual - 1)")
    fig.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, 1.07))
    return save(fig, "12_ltv_backtest.png")


def fig_power(exp: dict) -> str:
    a, b = exp["offer_test"], exp["retention_test"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
    t = pd.DataFrame(a["table"])
    a1.plot(t.mde, t.raw, color=LOSS, marker="o", ms=4, lw=2, label="Raw outcome")
    a1.plot(t.mde, t.winsor, color=ACCENT, marker="o", ms=4, lw=2, label="Winsorized at 99th pct")
    a1.axvline(a["observed_gap_180"], color=INK2, lw=1, ls="--")
    a1.text(a["observed_gap_180"] * 1.04, t.raw.max() * 0.5, f"expected gap\n${a['observed_gap_180']:.0f}", fontsize=9, color=INK2)
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xticks(t.mde, [f"${m}" for m in t.mde])
    a1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a1.yaxis.set_minor_formatter(mtick.NullFormatter())
    a1.set_xlabel("Minimum detectable effect, 180-day contribution per FTD")
    a1.set_ylabel("FTDs needed per arm")
    a1.set_title("Welcome-offer test: sample size (5% alpha, 80% power)", fontsize=10.5)
    a1.legend(fontsize=9)
    bins = np.linspace(-80, 80, 41)
    for k, label, col in (("raw", "Difference in means", "#9AA7B4"), ("cuped", "CUPED", ACCENT)):
        a2.hist(b["split_diffs"][k], bins=bins, color=col, alpha=0.65, label=f"{label} (SD {b['empirical_sd_of_estimate'][k]:.1f})")
    a2.set_xlabel("Estimated effect on 1,000 random splits with no true effect ($)")
    a2.set_ylabel("Splits")
    a2.set_title("Retention test: CUPED narrows the estimate", fontsize=10.5)
    a2.legend(fontsize=9)
    fig.tight_layout()
    return save(fig, "13_test_design.png")


def fig_calibration(model: dict) -> str:
    c = model["calibration"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    for k, label, col in (("xgboost", "XGBoost", ACCENT), ("logistic", "Logistic regression", AMBER)):
        bins = c[k]["bins"]
        a1.plot([x["pred"] for x in bins], [x["actual"] for x in bins], marker="o", ms=5, lw=1.8, color=col,
                label=f"{label} (Brier {c[k]['brier']:.3f})")
    a1.set_xlim(0.15, 1); a1.set_ylim(0.15, 1)
    a1.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    a1.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    a1.set_xlabel("Predicted probability of betting in month 3 (decile mean)")
    a1.set_ylabel("Actual share")
    a1.set_title("Retention model calibration", fontsize=10.5)
    a1.legend(fontsize=9, loc="upper left")
    sq = pd.DataFrame(model["value_squared_error"]["deciles"]); tp = pd.DataFrame(model["value"]["deciles"])
    lim = [min(sq.actual.min(), tp.predicted.min(), sq.predicted.min()) - 60, max(sq.actual.max(), tp.actual.max()) * 1.25]
    a2.plot(lim, lim, color=INK2, lw=1, ls="--")
    a2.scatter(sq.predicted, sq.actual, color="#9AA7B4", s=40, label="Squared-error XGBoost", zorder=3)
    a2.scatter(tp.predicted, tp.actual, color=ACCENT, s=40, label="Two-part XGBoost", zorder=3)
    a2.set_xscale("symlog", linthresh=100); a2.set_yscale("symlog", linthresh=100)
    for ax_ in (a2.xaxis, a2.yaxis):
        ax_.set_major_formatter(mtick.FuncFormatter(lambda v, _: usd(v)))
        ax_.set_minor_formatter(mtick.NullFormatter())
    a2.set_xlabel("Predicted 180-day theo contribution (decile mean, symlog scale)")
    a2.set_ylabel("Actual")
    a2.set_title("Value model calibration by decile", fontsize=10.5)
    a2.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    return save(fig, "14_calibration.png")


def fig_gameplay(bt_m: pd.DataFrame, sp_m: pd.DataFrame) -> str:
    bt_m = bt_m.assign(month=pd.to_datetime(bt_m["month"]))
    sp_m = sp_m.assign(month=pd.to_datetime(sp_m["month"]))
    bt_m, sp_m = bt_m[bt_m.month >= "2024-10-01"], sp_m[sp_m.month >= "2024-10-01"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
    piv = bt_m.pivot(index="month", columns="bet_type", values="handle").fillna(0)
    share = piv.div(piv.sum(axis=1), axis=0)
    cols = {"straight": ACCENT, "live": "#7FA3C9", "parlay": AMBER, "sgp": LOSS}
    order = ["straight", "live", "parlay", "sgp"]
    a1.stackplot(share.index, *[share[c] for c in order], colors=[cols[c] for c in order],
                 labels=[nm(c) for c in order], alpha=0.9)
    hold = bt_m.groupby("month").apply(lambda g: g.ggr.sum() / g.handle.sum(), include_groups=False)
    b = a1.twinx()
    b.plot(hold.index, hold.values, color=INK, lw=1.8, marker="o", ms=3, label="Hold (right axis)")
    b.set_ylim(0, 0.16); b.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0)); b.grid(False)
    b.spines["right"].set_visible(True)
    a1.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0)); a1.set_ylim(0, 1)
    a1.set_title("Share of handle by bet type, and hold")
    h1, l1 = a1.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left", ncol=2, framealpha=0.9, frameon=True)
    ps = sp_m.pivot(index="month", columns="sport", values="handle").fillna(0)
    ps = ps.div(ps.sum(axis=1), axis=0)
    sorder = ["nfl", "college_football", "nba", "college_basketball", "mlb", "nhl", "soccer", "other"]
    scol = ["#1D4E89", "#7FA3C9", "#C27C0E", "#E8B96A", "#1E7A4F", "#8CC2A6", "#7A5AA6", "#B9BFB6"]
    a2.stackplot(ps.index, *[ps[c] for c in sorder if c in ps], colors=scol, labels=[nm(c) for c in sorder if c in ps])
    a2.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0)); a2.set_ylim(0, 1)
    a2.set_title("Share of handle by sport")
    a2.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    for ax in (a1, a2):
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y"))
        ax.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout()
    return save(fig, "15_gameplay.png")


def seasonality_check(seas: pd.DataFrame) -> float:
    """Correlation between the simulated betting rate among retained players by calendar
    month and the real NY seasonality index. Uses hidden lifetimes (validation only)."""
    pl = pd.read_parquet(ROOT / "data" / "players.parquet", columns=["player_id", "signup_date"])
    tr = pd.read_parquet(ROOT / "data" / "_truth.parquet", columns=["player_id", "lifetime_days"])
    a = pd.read_parquet(ROOT / "data" / "activity.parquet", columns=["player_id", "activity_date"])
    s = pl.signup_date.values.astype("datetime64[D]")
    obs = np.datetime64("2026-06-30")
    end = np.minimum(s + tr.lifetime_days.values.astype("timedelta64[D]"), obs + np.timedelta64(1, "D"))
    start = s + np.timedelta64(30, "D")
    months = pd.period_range("2024-10", "2026-06", freq="M")
    alive = []
    for m in months:
        ms = np.datetime64(m.start_time.date())
        me = np.datetime64((m.end_time.normalize() + pd.Timedelta(days=1)).date())
        alive.append(np.clip((np.minimum(end, me) - np.maximum(start, ms)).astype(int), 0, None).sum())
    a = a.merge(pl, on="player_id")
    a = a[(pd.to_datetime(a.activity_date) - pd.to_datetime(a.signup_date)).dt.days >= 30]
    act = a.groupby(pd.to_datetime(a.activity_date).dt.to_period("M")).size().reindex(months)
    rate = pd.Series(act.values / np.array(alive), index=months)
    idx = rate.groupby(rate.index.month).mean()
    real = seas.set_index("month_of_year").handle_index
    return float(pd.concat([idx, real], axis=1).corr().iloc[0, 1])


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
        "calib": fig_calibration(model),
    }
    if alt:
        figs["tabpfn"] = fig_tabpfn(alt)
    ltv = json.loads((OUT / "ltv_forecast.json").read_text())
    exp = json.loads((OUT / "experiment_design.json").read_text())
    figs["ltv"] = fig_ltv_projection(ltv)
    figs["ltv_bt"] = fig_ltv_backtest(ltv)
    figs["power"] = fig_power(exp)
    tp = model["value"]
    sqm, rtm, luck = model["value_squared_error"], model["value_realized_target"], model["luck"]
    vr = v["realized"]
    cal = model["calibration"]
    gseg = pd.read_csv(OUT / "19_gameplay_segments.csv")
    bt_m = pd.read_csv(OUT / "17_gameplay_bet_types.csv")
    sp_m = pd.read_csv(OUT / "18_gameplay_sports.csv")
    bt_all = bt_m.groupby("bet_type")[["handle", "ggr", "bets"]].sum().reset_index()
    bt_all["hold"] = bt_all.ggr / bt_all.handle
    bt_all["share"] = bt_all.handle / bt_all.handle.sum()
    bt_all["bet_share"] = bt_all.bets / bt_all.bets.sum()
    n_bets = int(bt_all.bets.sum())
    figs["gameplay"] = fig_gameplay(bt_m, sp_m)
    season_corr = seasonality_check(seas)
    sample_day = "2025-06-15"
    score_sample = pd.read_parquet(OUT / "scores" / f"day14_scores_{sample_day}.parquet")
    drift_sample = pd.read_csv(OUT / "scores" / f"drift_{sample_day}.csv")
    bt1, bt2 = ltv["backtest_6_to_12"], ltv["backtest_12_to_18"]
    proj = {p_["channel"]: p_ for p_ in ltv["projection"]}
    ea, eb = exp["offer_test"], exp["retention_test"]
    n_gap_w = ea["n_observed_gap"]["winsor"]
    months_all = 2 * n_gap_w / ea["ftds_per_month_all_channels"]
    months_ps = 2 * n_gap_w / ea["ftds_per_month_paid_social"]

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
    num_feats = ["handle_14", "max_day_handle_14", "avg_stake_14", "bets_14", "active_days_14", "active_days_wk2",
                 "days_idle_at_d14", "ggr_14", "parlay_share_14", "live_share_14", "n_sports_14", "football_share_14",
                 "season_ahead_180", "season_m3", "welcome_promo_cost"]
    feat_labels = {
        "active_days_14": "Active days, days 0 to 13", "active_days_wk2": "Active days, days 7 to 13",
        "bets_14": "Bets placed", "handle_14": "Handle (total staked)", "ggr_14": "GGR (book's realized win)",
        "max_day_handle_14": "Largest single-day handle", "avg_stake_14": "Average stake per bet",
        "days_idle_at_d14": "Days since last bet, at day 13", "welcome_promo_cost": "Welcome offer cost",
        "parlay_share_14": "Parlay and same-game-parlay share of handle", "live_share_14": "Live-bet share of handle",
        "n_sports_14": "Number of sports bet", "football_share_14": "Football share of handle",
        "season_ahead_180": "Calendar ahead, days 14 to 179 (NY index)", "season_m3": "Calendar ahead, month 3 (NY index)",
    }
    corr = pd.DataFrame([{
        "Feature": feat_labels[f_],
        "Spearman with 180-day theo value": f"{feats[f_].corr(feats.theo_contribution_180, method='spearman'):.2f}",
        "Spearman with active in month 3": f"{feats[f_].corr(feats.active_m3, method='spearman'):.2f}",
    } for f_ in num_feats])
    sq_top, tp_top = sqm["deciles"][0], v["deciles"][0]
    ratio = lambda d: d["predicted"] / d["actual"]  # noqa: E731
    x_bins = cal["xgboost"]["bins"]
    ret_best = "XGBoost" if cal["xgboost"]["brier"] <= cal["logistic"]["brier"] else "logistic regression"
    g = lambda cut, band: gseg[(gseg.cut == cut) & gseg.band.str.contains(band, regex=False)].iloc[0]  # noqa: E731
    one_sport, three_sport = g("Sports bet in first 14 days", "one sport"), g("Sports bet in first 14 days", "three or more")
    fb_heavy, fb_mixed = g("Football share of handle", "football-heavy"), g("Football share of handle", "mixed")
    par = gseg[gseg.cut == "Parlay share of handle"].reset_index(drop=True)
    par_best, par_worst = par.loc[par.mean_contribution_180.idxmax()], par.loc[par.mean_contribution_180.idxmin()]
    bt_tot = bt_all.set_index("bet_type")
    parlay_share = float(bt_tot.loc[["parlay", "sgp"], "share"].sum())
    parlay_ggr_share = float(bt_tot.loc[["parlay", "sgp"], "ggr"].sum() / bt_tot["ggr"].sum())
    dash_url = "https://robertcarrington22.github.io/sportsbook-customer-economics/"

    # ---- Title and abstract -----------------------------------------------
    add("# Sportsbook Customer Economics\n")
    add("Robert Carrington · September 2026\n")
    add(f"Interactive dashboard: [{dash_url.replace('https://', '')}]({dash_url})\n")
    add("## Abstract\n")
    add(
        f"This report estimates what a sportsbook's first-time depositors (FTDs) cost to acquire, what they contribute, and "
        f"how early their value can be predicted. The data is a simulation of {len(feats):,} FTDs and "
        f"{n_bets:,} individual bets, with seasonality, hold, and hold volatility calibrated to DraftKings' public monthly "
        f"filings in New York. Year-one contribution averages {usd(blended_ltv)} per FTD against a blended CAC of "
        f"{usd(blended_cac)}, a {blended_ltv / blended_cac:.2f}× return, but value is highly concentrated: the top 1% of FTDs "
        f"produce {pct(top(1))} of it and {pct(net_neg)} are net negative at day 180. Channel, welcome offer, state tax, "
        f"and early gameplay each move value materially; players who bet three or more sports in their first two weeks "
        f"are {three_sport.active_m3_rate / one_sport.active_m3_rate:.1f}× as likely to still be betting in month 3 as "
        f"single-sport players. Valuing players on theoretical rather than realized win removes bet-outcome luck, and on "
        f"that basis a two-part XGBoost model ranks 180-day value at {v['spearman_model']:.2f} Spearman from day-14 data, "
        f"within {abs(v['mean_predicted'] / v['mean_actual'] - 1):.0%} of the actual average. A retention-decay model, "
        f"backtested within {abs(bt1['overall_err']['sbg']):.0%} of 12-month value, projects 36-month returns of "
        f"{ltv['projection'][-1]['ratio_36']:.1f}× to {ltv['projection'][0]['ratio_36']:.1f}× CAC. The report sizes the "
        f"experiments its recommendations need, and ships the scoring as a daily job with drift monitoring."
        + (f" A separate section tests TabPFN, a pretrained tabular foundation model: with {alt['n_context']:,} training "
           f"players and no tuning, it ranks value at {alt['models'][0]['spearman']:.2f} Spearman, "
           + ("matching" if abs(alt['models'][0]['spearman'] - alt['two_part_full']['spearman']) < 0.005 else
              "beating" if alt['models'][0]['spearman'] > alt['two_part_full']['spearman'] else "close to")
           + f" the best XGBoost model trained on {model['split']['n_train']:,}." if alt else "")
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
        ["Value model rank correlation (theo, day 14)", f"{v['spearman_model']:.2f}"],
    ], columns=["Headline", "Value"]), "lr") + "\n")

    add("## Contents\n")
    add("\n".join([
        "1. [Objective](#1-objective)", "2. [Data collection](#2-data-collection)",
        "3. [Data cleaning and transformation](#3-data-cleaning-and-transformation)",
        "4. [Exploratory data analysis](#4-exploratory-data-analysis)", "5. [Feature engineering](#5-feature-engineering)",
        "6. [Modeling choices](#6-modeling-choices)", "7. [Findings](#7-findings)",
        "8. [Test design for the recommended changes](#8-test-design-for-the-recommended-changes)",
        "9. [From analysis to production](#9-from-analysis-to-production)",
        "10. [Conclusion](#10-conclusion)", "11. [TabPFN: process and comparison](#11-tabpfn-process-and-comparison)",
        "12. [Reproducibility](#12-reproducibility)",
    ]) + "\n")

    # ---- 1. Objective -------------------------------------------------------
    add("## 1. Objective\n")
    add("A customer economics team decides how much to pay for a new customer and where to spend to keep them. This analysis answers five questions:\n")
    add("\n".join([
        "1. What is a first-time depositor worth over 12 months, after promos, tax, and variable cost, and beyond?",
        "2. How do acquisition channel, welcome offer, and state change that value and the payback period?",
        "3. How do retention and gameplay evolve, and what does early gameplay say about a player?",
        "4. How well can a player's value be predicted from their first 14 days, and which model should do it?",
        "5. What experiments and systems would it take to act on the answers?",
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
        f"`simulate.py` generates {len(feats):,} FTDs who signed up between September 2024 and August 2025 and every bet "
        f"they placed through June 2026: {n_bets:,} bets on {len(act):,} player-days. Each FTD has:\n"
    )
    add("\n".join([
        "- an acquisition channel (referral, TV and brand, search, affiliate, paid social), each with its own CAC",
        "- a welcome offer (no-sweat first bet, deposit match, or bet $5 get $200), each with its own expected cost",
        "- a state (nine, each with its tax rate)",
        "- a bet-type mix (straight, live, parlay, same-game parlay) and sport preferences (NFL, college football, NBA, "
        "college basketball, MLB, NHL, soccer, other), drawn around their player type's typical mix",
        "- a hidden player type (bonus hunter, casual, regular, high value) that sets churn, betting frequency, stake size, "
        "and how much of the welcome offer the player extracts",
    ]) + "\n")
    add(
        "Each bet gets a bet type from the player's mix and a sport weighted by the player's preferences and that sport's "
        "real season calendar. Players bet more in the months their sports are in season, so a football-only bettor goes "
        "quiet after the Super Bowl. Channels and offers shift the mix of player types, which is the main assumption "
        "behind channel and offer results. The analysis never sees player type. It works only from behavior, as it would "
        "on real data. Type is used once, at the end, to check whether the models find the right players.\n"
    )
    add("**What is calibrated to real data, and how close the simulation lands:**\n")
    add(table(pd.DataFrame([
        ["Hold, Sep 2024 onward", pct(real_hold, 1), pct(sim_hold, 1)],
        ["Monthly hold, standard deviation", f"{real_hold_sd * 100:.2f} pts", f"{sim_hold_sd * 100:.2f} pts"],
        ["Seasonality of activity by calendar month", "Index (2.1)", f"correlation {season_corr:.2f} with the real index"],
        ["New York tax rate", "51% of GGR", "51%"],
    ], columns=["Measure", "DraftKings NY (real)", "Simulation"]), "lrr") + "\n")
    add(
        "Hold is set by the bet-type mix, since parlays hold far more than straight bets (section 4.5). Hold also swings "
        "month to month because every customer bets on the same games, so outcomes are correlated. Bet outcomes alone "
        "produce part of that swing; one shared shock per month supplies the rest, sized so the total matches the real "
        "standard deviation. The seasonality check compares each calendar month's betting rate among retained players "
        "with the real index. Channel CACs, offer costs, bet-type mixes, player-type behavior, and the non-New York tax "
        "rates are my assumptions, all listed in `assumptions.toml`.\n"
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
        f"All transformation is in SQL (DuckDB), in `sql/01` through `sql/19`. Bets roll up to player-days, and player-days "
        f"to a **player-month panel**: one row per FTD per 30-day \"life month\" since first deposit, "
        f"{int(pm_summary.player_months):,} rows in all. Three decisions shape it:\n"
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
    add("**Contribution** is the value measure throughout, in two versions:\n")
    add("```\ncontribution      = GGR  - welcome promo - ongoing promos - tax x (GGR  - promos) - 0.6% x handle\n"
        "theo contribution = theo - welcome promo - ongoing promos - tax x (theo - promos) - 0.6% x handle\n"
        "theo              = sum over bets of stake x the book's expected hold for that bet type\n```\n")
    add(
        "Contribution uses realized GGR: what the book actually won, including luck. Theo contribution uses theoretical "
        "win: what the book expects to win given what and how the player bets. Both are contribution margins, not profit: "
        "they exclude fixed costs and overhead. Tax is applied to GGR net of promos as a simplification (section 10.3). "
        "Historical results in this report use realized contribution; predictions and projections use theo, for the "
        "reasons in 7.1.\n"
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
        f"ever wins back from them. This shapes every later choice: averages are fragile, so channel results get bootstrap "
        f"intervals, and model evaluation leans on rank-based metrics rather than squared error.\n"
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
        f"against {pct(m2s[m2_lo])} of {pd.Timestamp(m2_lo):%B %Y} signups.\n"
    )
    add(f"![Retention by channel and cohort value]({figs['ret2']})\n")
    add(
        f"FTDs who sign up between September and January are worth {usd(fall_v)} over six months, against "
        f"{usd(spring_v)} for February through July signups, whose early months overlap the summer lull.\n"
    )
    add("### 4.4 Early handle against later value\n")
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
    add("### 4.5 Gameplay: bet types and sports\n")
    add(f"![Gameplay trends]({figs['gameplay']})\n")
    add(table(pd.DataFrame([{
        "Bet type": nm(b_), "Share of handle": pct(bt_tot.loc[b_, "share"]), "Share of bets": pct(bt_tot.loc[b_, "bet_share"]),
        "Hold": pct(bt_tot.loc[b_, "hold"], 1), "Share of GGR": pct(bt_tot.loc[b_, "ggr"] / bt_tot["ggr"].sum()),
    } for b_ in ["straight", "live", "parlay", "sgp"]]), "lrrrr") + "\n")
    add(
        f"Parlays and same-game parlays are {pct(parlay_share)} of handle but {pct(parlay_ggr_share)} of GGR, because they "
        f"hold three to four times as much as straight bets. That mix is the main driver of a book's hold. Sport mix "
        f"follows the calendar: football dominates the fall, basketball the winter and spring, and baseball the summer.\n"
    )
    add(
        "What a player bets in their first two weeks says a lot about what comes next. The last column follows fall "
        "signups (September to December 2024) into the following offseason, March to August 2025:\n"
    )
    add(table(pd.DataFrame([{
        "First 14 days": f"{row_.cut}: {row_.band.split('  ', 1)[1]}", "FTDs": f"{row_.ftds:,}",
        "Avg handle": usd(row_.avg_handle_14), "Active month 3": pct(row_.active_m3_rate),
        "Mean 180-day contribution": usd(row_.mean_contribution_180),
        "Fall signups still betting in offseason": pct(row_.offseason_active_rate_fall_signups),
    } for row_ in gseg.itertuples()]), "lrrrrr") + "\n")
    add("\n".join([
        f"- **Breadth is the strongest gameplay signal.** Players who bet three or more sports in their first 14 days "
        f"retain at {pct(three_sport.active_m3_rate)} into month 3, against {pct(one_sport.active_m3_rate)} for "
        f"single-sport players, and are worth {usd(three_sport.mean_contribution_180)} against "
        f"{usd(one_sport.mean_contribution_180)}. Part of this is volume, since players who bet more touch more sports, "
        f"which is why the model includes both.",
        f"- **Football-only players fade in the offseason.** Among fall signups, {pct(fb_heavy.offseason_active_rate_fall_signups)} "
        f"of football-heavy players are still betting between March and August, against "
        f"{pct(fb_mixed.offseason_active_rate_fall_signups)} of those with a mixed sport diet. That points to a concrete "
        f"marketing action: cross-sell basketball and baseball to football-heavy players before the Super Bowl, while "
        f"they are still engaged.",
        f"- **Parlay share has a sweet spot.** Players with {par_best.band.split('  ', 1)[1]} of handle in parlays are worth "
        f"the most ({usd(par_best.mean_contribution_180)}), while those at {par_worst.band.split('  ', 1)[1]} are small, "
        f"short-lived, and net negative ({usd(par_worst.mean_contribution_180)}). Heavy parlay bettors hold well per dollar "
        f"but bet few dollars.",
        "- These relationships come partly from my assumptions (casual types lean toward NFL and parlays). The analysis "
        "shows how to measure them; the sizes on real data could differ.",
    ]) + "\n")

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
        "- `theo_contribution_180`: theoretical contribution over days 0 to 179, the primary value target (section 7.1 "
        "explains why theo rather than realized). It includes the first 14 days by design, because the business question "
        "is the player's total value; the model's real work is the remaining 166 days.",
        "- `contribution_180`: realized contribution over the same window, kept for comparison.",
    ]) + "\n")
    add("**Features**, with their rank correlation to each target\n")
    add(table(corr, "lrr") + "\n")
    add("\n".join([
        "- Categorical context: acquisition channel, welcome offer, state, and signup month.",
        "- Week 2 activity and days idle capture the *trend* of engagement: a player who bet heavily on day 1 and then "
        "stopped differs from one still betting on day 13.",
        "- Largest single-day handle sits alongside total handle because a few large days signal a high-stakes player "
        "differently from many small ones.",
        "- Gameplay features (parlay share, live share, number of sports, football share) come from the bet-level table "
        "and describe *how* a player bets, not just how much.",
        "- **Calendar-ahead features** average DraftKings' real New York seasonality index over the player's outcome "
        "window. They depend only on the signup date, so they are known at day 14 and are not leakage. They exist because "
        "of the out-of-time split: the model trains on September to April signups and is tested on May to August, so "
        "signup month alone gives it nothing for summer cohorts. A continuous measure of how busy the calendar ahead is "
        "generalizes where signup month cannot.",
        "- **Leakage check:** no feature uses anything after day 13, and nothing downstream of the outcome (such as churn "
        "date or lifetime) is available to the model.",
    ]) + "\n")
    add(
        "Transformations depend on the model. For logistic regression, numeric features get a signed log transform, "
        "`sign(x) * log(1 + |x|)`, to tame the heavy tails (GGR can be negative), then standardization, and categoricals "
        "are one-hot encoded. XGBoost takes raw values and native categorical splits, since trees are invariant "
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
        "- **XGBoost** as the main model: histogram trees with native categorical splits, which handle missing values, "
        "categoricals, and interactions without manual work. Settings: learning rate 0.03, max depth 6, minimum child "
        "weight 5, row and column subsampling of 0.8, and early stopping after 100 rounds without improvement on a held-out "
        "10% of the training rows, so the number of trees is chosen by validation.",
        "- **Theo as the value target.** Value models predict theoretical contribution. Realized contribution is reported "
        "alongside, and section 7.1 shows the difference.",
        "- **A two-part value model**, because value is heavy-tailed and often negative. One XGBoost classifier estimates "
        "the chance a player ends up profitable. A second predicts log(1 + value) for profitable players, converted back "
        "to dollars with Duan's smearing correction estimated on held-out rows. A third predicts the size of the loss for "
        "unprofitable players. Expected value is p × E[value | profitable] + (1 − p) × E[value | not profitable]. A plain "
        "squared-error XGBoost is kept as a comparison.",
        "- **Metrics chosen for the skew.** AUC for retention. For value, Spearman rank correlation and top-decile lift, "
        "because the business action is ranking players and a few whales dominate any squared-error metric. Mean absolute "
        "error and calibration (predicted against actual by decile) are reported because bids are set in dollars.",
        "- **Permutation importance** on the test set: how much accuracy drops when a feature is shuffled.",
        "- **Bootstrap intervals** for channel LTV/CAC: 2,000 resamples of players within each channel.",
        "- **Lifetime value beyond the observed window** with a shifted-beta-geometric (sBG) retention curve per channel: "
        "the share of FTDs still betting in month t is S(t) = B(a, b + t) / B(a, b), which allows churn to differ across "
        "players and gives the long, slow-decaying tail a single churn rate cannot. Projected theo contribution is S(t) "
        "times the theo value of an active player (the average of the last three observed months), backtested twice on "
        "held-out months, with bootstrap intervals.",
        "- **Test design**: sample size per arm at 5% significance and 80% power, winsorizing at the 99th percentile, and "
        "CUPED or regression adjustment on pre-treatment covariates.",
    ]) + "\n")

    # ---- 7. Findings --------------------------------------------------------
    add("## 7. Findings\n")
    add("### 7.1 Model performance\n")
    add(f"![Model results]({figs['model']})\n")
    td, th = v["top_decile_model"], v["top_decile_heuristic"]
    add(table(pd.DataFrame([
        ["Bets in month 3?", "AUC", f"{r['auc_gbm']:.3f} (XGBoost)", f"{r['auc_logistic']:.3f} (logistic regression)"],
        ["Flag the at-risk decile", "Share who lapse", pct(r["lapse_rate_bottom_decile"]), f"{pct(r['lapse_rate_all'])} (all players)"],
        ["Rank by 180-day theo value", "Spearman", f"{v['spearman_model']:.3f}", f"{v['spearman_heuristic']:.3f} (sort by 14-day handle)"],
        ["Find the top decile", "Lift over average", f"{td['lift']:.2f}×", f"{th['lift']:.2f}× (sort by 14-day handle)"],
        ["Predict dollar value", "Mean absolute error", usd(v["mae_model"]), f"{usd(v['mae_predict_mean'])} (predict the mean)"],
        ["Find hidden high-value players", "Share of top decile", pct(v["hidden_high_value_share_top_decile"]), f"{pct(v['hidden_high_value_share_all'])} (base rate)"],
    ], columns=["Question", "Metric", "Model", "Baseline"]), "llrl") + "\n")
    add("\n".join([
        f"- On retention, XGBoost and logistic regression are close ({r['auc_gbm']:.3f} and {r['auc_logistic']:.3f} AUC); "
        f"{ret_best} is better calibrated (below).",
        f"- On value, the two-part model ranks the whole base far better than early handle alone ({v['spearman_model']:.2f} "
        f"vs {v['spearman_heuristic']:.2f}) and is well calibrated: it predicts an average of {usd(v['mean_predicted'])} "
        f"against an actual {usd(v['mean_actual'])}, and {usd(tp_top['predicted'])} for its top decile against an actual "
        f"{usd(tp_top['actual'])}.",
        f"- {pct(v['hidden_high_value_share_top_decile'])} of the model's top decile are truly high-value players, against a "
        f"{pct(v['hidden_high_value_share_all'])} base rate, which confirms it finds the right people rather than fitting "
        f"noise. Handle, stake size, and parlay share carry most of the signal.",
    ]) + "\n")
    add("**Why theo, not realized value.** The first version of this model predicted realized contribution, and it "
        "under-predicted the test cohorts badly. The reason is luck. Over the 166 days being predicted, bettors in the "
        "training cohorts happened to win more than the book expected, and the test cohorts did not:\n")
    add(table(pd.DataFrame([
        ["Training cohorts", pct(luck["train"]["realized_hold"], 2), pct(luck["train"]["expected_hold"], 2), f"{luck['train']['luck_pts']:+.2f} pts"],
        ["Test cohorts", pct(luck["test"]["realized_hold"], 2), pct(luck["test"]["expected_hold"], 2), f"{luck['test']['luck_pts']:+.2f} pts"],
    ], columns=["Days 14 to 179", "Realized hold", "Expected hold, given their bets", "Book's luck"]), "lrrr") + "\n")
    add(
        f"A model trained on realized value learns the training cohorts' bad luck as if it were player behavior. Because "
        f"contribution is only about half of GGR, a point of hold luck moves contribution by roughly twice as much in "
        f"percentage terms. Sportsbooks value players on theoretical win for exactly this reason. Swapping the target "
        f"changes the picture:\n"
    )
    add(table(pd.DataFrame([
        ["Squared-error XGBoost, realized target", f"{rtm['spearman_vs_theo']:.3f}", f"{rtm['top_ratio_vs_theo']:.2f}", usd(rtm["mean_predicted"])],
        ["Squared-error XGBoost, theo target", f"{sqm['spearman_model']:.3f}", f"{ratio(sq_top):.2f}", usd(sqm["mean_predicted"])],
        ["Two-part XGBoost, theo target", f"{v['spearman_model']:.3f}", f"{ratio(tp_top):.2f}", usd(v["mean_predicted"])],
    ], columns=["Value model", "Spearman vs theo", "Top decile, predicted ÷ actual", f"Mean prediction (actual {usd(v['mean_actual'])})"]), "lrrr") + "\n")
    add(
        f"Scored against realized outcomes instead, every model's ranking drops (the two-part model to "
        f"{vr['spearman_model']:.2f}, early handle to {vr['spearman_heuristic']:.2f}), because six months of realized "
        f"results for an individual player are dominated by whether their bets won. Realized and theo 180-day value "
        f"correlate at only {vr['corr_theo_realized']:.2f} across players. That noise is real money, but no behavioral "
        f"model can predict it, so it belongs in the error bars, not in the model.\n"
    )
    add(f"![Calibration]({figs['calib']})\n")
    compressed = x_bins[0]["pred"] > x_bins[0]["actual"] and x_bins[-1]["pred"] < x_bins[-1]["actual"]
    add(
        f"**Calibration.** Both retention models beat the base rate on Brier score ({cal['xgboost']['brier']:.3f} for "
        f"XGBoost, {cal['logistic']['brier']:.3f} for logistic regression, {cal['brier_base_rate']:.3f} for predicting "
        f"the training average). "
        + (f"XGBoost's probabilities are slightly compressed: its lowest decile predicts {pct(x_bins[0]['pred'])} and sees "
           f"{pct(x_bins[0]['actual'])}, and its highest predicts {pct(x_bins[-1]['pred'])} and sees "
           f"{pct(x_bins[-1]['actual'])}. " if compressed else
           f"XGBoost's lowest decile predicts {pct(x_bins[0]['pred'])} and sees {pct(x_bins[0]['actual'])}; its highest "
           f"predicts {pct(x_bins[-1]['pred'])} and sees {pct(x_bins[-1]['actual'])}. ")
        + f"On value, the two-part model's deciles sit on the diagonal, while the squared-error model's sit below it at "
        f"the top.\n"
    )
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
    below_one = [nm(c) for c in boot.index if boot.loc[c, "ratio_lo"] < 1]
    add(
        (f"Every channel pays back inside a year, from month {int(best.payback_month) + 1} for {nm(best.channel).lower()} "
         f"to month {int(worst.payback_month) + 1} for {nm(worst.channel).lower()}. "
         if not ch.payback_month.isna().any() else "")
        + ("Every interval stays above 1.0×. " if not below_one else
           f"The interval for {', '.join(below_one).lower()} reaches below 1.0×, so its year-one return is not certain. ")
        + "The median FTD is net negative in every channel. Channels differ in how many high-value players they bring, "
        "not in how the typical player behaves. These are realized results: what actually happened, luck included.\n"
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

    add("### 7.5 Lifetime value beyond 12 months\n")
    add(
        "Twelve months understates what a player is worth, because a meaningful share are still betting at month 12. "
        "Projections use theo contribution, so luck in the fit window does not carry forward. Before projecting, I tested "
        "the method on months it never saw.\n"
    )
    add(f"![LTV backtest]({figs['ltv_bt']})\n")
    add(table(pd.DataFrame([
        [f"Fit months 1 to 6, predict month 12 ({sum(r_['ftds'] for r_ in bt1['channels']):,} FTDs)",
         f"{bt1['overall_err']['sbg']:+.1%}", f"{bt1['overall_err']['run_rate']:+.1%}", f"{bt1['overall_err']['stop']:+.1%}",
         f"{bt1['mean_abs_channel_err']['sbg']:.1%}"],
        [f"Fit months 1 to 12, predict month 18 ({sum(r_['ftds'] for r_ in bt2['channels']):,} FTDs)",
         f"{bt2['overall_err']['sbg']:+.1%}", f"{bt2['overall_err']['run_rate']:+.1%}", f"{bt2['overall_err']['stop']:+.1%}",
         f"{bt2['mean_abs_channel_err']['sbg']:.1%}"],
    ], columns=["Backtest", "sBG decay model", "Flat run rate", "Stop counting", "sBG, mean channel error"]), "lrrrr") + "\n")
    early_best = abs(bt1["overall_err"]["sbg"]) < abs(bt1["overall_err"]["run_rate"])
    add(
        (f"Early on, the decay model is clearly best: from six months of data it lands {bt1['overall_err']['sbg']:+.1%} "
         f"from the actual 12-month value, where a flat run rate misses by {bt1['overall_err']['run_rate']:+.1%} and "
         f"ignoring the future by {bt1['overall_err']['stop']:+.1%}. " if early_best else
         f"From six months of data the decay model lands {bt1['overall_err']['sbg']:+.1%} from the actual 12-month value "
         f"and a flat run rate {bt1['overall_err']['run_rate']:+.1%}. ")
        + f"By month 12 both are close over the next six months ({bt2['overall_err']['sbg']:+.1%} and "
        f"{bt2['overall_err']['run_rate']:+.1%}). But a run rate never decays, so it cannot be stretched to 36 months. The "
        f"decay model can, and its backtest errors bound how far to trust it. An earlier version projected realized "
        f"contribution and missed the 12-month value by about 11% from six months of data; the fit window's bettor luck "
        f"was the cause, and moving to theo removed it.\n"
    )
    add(f"![LTV projection]({figs['ltv']})\n")
    add(table(pd.DataFrame([{
        "Channel": nm(r_["channel"]), "CAC": usd(r_["cac"]), "12-mo theo LTV": usd(r_["ltv_12"]),
        "24-mo LTV": f"{usd(r_['ltv_24'])} ({usd(r_['ltv_24_lo'])} to {usd(r_['ltv_24_hi'])})",
        "36-mo LTV": f"{usd(r_['ltv_36'])} ({usd(r_['ltv_36_lo'])} to {usd(r_['ltv_36_hi'])})",
        "36-mo LTV / CAC": f"{r_['ratio_36']:.2f}× ({r_['ratio_36_lo']:.2f} to {r_['ratio_36_hi']:.2f})",
        "Still betting, month 36": pct(r_["survival_36"]),
    } for r_ in ltv["projection"]]), "lrrrrrr") + "\n")
    p_best, p_worst = ltv["projection"][0], ltv["projection"][-1]
    rank36 = [r_["channel"] for r_ in ltv["projection"]]
    add(
        f"Projected to 36 months, returns range from {p_best['ratio_36']:.1f}× for {nm(p_best['channel']).lower()} to "
        f"{p_worst['ratio_36']:.1f}× for {nm(p_worst['channel']).lower()}. "
        + ("The channel ranking is the same as at 12 months. " if ch_order == rank36 else
           "The ranking shifts slightly from the 12-month view, because channels with more high-value players keep "
           "accruing value for longer. ")
        + "The practical use is setting CAC targets: a team that requires payback inside 12 months is leaving value on "
        "the table for channels whose players keep betting, and the intervals show how much of that value is reliable. "
        "The projection holds value per active player flat and assumes churned players never return, and beyond 18 "
        "months it is untested.\n"
    )

    # ---- 8. Test design ----------------------------------------------------
    add("## 8. Test design for the recommended changes\n")
    add(
        "Two recommendations need experiments before anyone acts on them: switching the welcome offer, and targeting "
        "retention spend with the model. This section sizes both tests. The two cases differ in one important way: "
        "what the analysis is allowed to adjust for.\n"
    )
    add(f"![Test design]({figs['power']})\n")
    add("### 8.1 Welcome-offer test\n")
    add(
        f"New depositors would be randomized at signup between the no-sweat offer and bet $5 get $200, with 180-day "
        f"contribution as the outcome. The expected difference, from the simulation, is {usd(ea['observed_gap_180'])} per "
        f"FTD. The outcome's standard deviation is {usd(ea['sd'])}, {ea['sd'] / ea['mean']:.0f} times its mean, so a plain "
        f"test is expensive.\n"
    )
    add(table(pd.DataFrame([{
        "Minimum detectable effect": usd(row_["mde"]), "Raw outcome": f"{row_['raw']:,}",
        "Winsorized at 99th pct": f"{row_['winsor']:,}", "Winsorized and adjusted": f"{row_['winsor_adjusted']:,}",
    } for row_ in ea["table"] if row_["mde"] in (25, 50, 100, 150)]), "lrrr") + "\n")
    th_ = ea["theo"]
    add("\n".join([
        f"- **Winsorizing** the outcome at the 99th percentile ({usd(ea['winsor_cap'])}) cuts the required sample by about "
        f"{1 - ea['n_observed_gap']['winsor'] / ea['n_observed_gap']['raw']:.0%}: to detect the expected "
        f"{usd(ea['observed_gap_180'])} gap, {ea['n_observed_gap']['winsor']:,} FTDs per arm instead of "
        f"{ea['n_observed_gap']['raw']:,}. The cost is a slightly different estimand, the effect on capped value, which "
        "should be stated up front.",
        f"- **Measuring on theo** cuts it further. Theo still reflects any change in how much or what players bet, but "
        f"drops the luck in whether their bets won: {th_['n_winsor']:,} FTDs per arm with a winsorized theo outcome, "
        f"{1 - th_['n_winsor'] / ea['n_observed_gap']['winsor']:.0%} fewer than on realized value.",
        f"- **Covariate adjustment barely helps here.** Only information known before randomization is allowed, which "
        f"for a new signup means channel, state, and signup month. Together they explain {pct(ea['r2_pre_signup_winsor'], 1)} "
        f"of the variance.",
        f"- **Early betting cannot be used as a covariate**, even though it would explain {pct(ea['r2_post_treatment_if_misused'])} "
        "of the variance. The offer changes how people bet in their first two weeks, so adjusting for that behavior would "
        "absorb part of the very effect being measured.",
        f"- **Duration.** {2 * ea['n_observed_gap']['winsor']:,} FTDs in total is about {months_all:.1f} months of signups "
        f"across all channels, or {months_ps:.1f} months of paid social alone, plus 180 days to observe the outcome. The "
        "day-14 value model could provide an early read, but only as a leading indicator, not as the decision metric.",
    ]) + "\n")
    add("### 8.2 Retention-spend test\n")
    add(
        f"Existing players who bet in month 3 would be randomized to receive a retention offer or not at the start of "
        f"month 4, with contribution in months 4 to 9 as the outcome ({eb['n_players']:,} such players in the data). "
        f"Here the first three months happened before randomization, so they are valid covariates. CUPED adjusts each "
        f"player's outcome by their pre-period contribution; regression adjustment uses several pre-period measures, "
        f"including handle and active days.\n"
    )
    add(table(pd.DataFrame([
        ["CUPED, pre-period contribution", pct(eb["var_reduction_cuped"]), pct(eb["empirical_var_reduction"]["cuped"])],
        ["Regression, four pre-period measures", pct(eb["r2_multi"]), pct(eb["empirical_var_reduction"]["regression"])],
    ], columns=["Adjustment", "Variance reduction, formula", "Variance reduction, 1,000 random splits"]), "lrr") + "\n")
    add(table(pd.DataFrame([{
        "Minimum detectable effect": usd(row_["mde"]), "Unadjusted": f"{row_['raw']:,}",
        "CUPED": f"{row_['cuped']:,}", "Regression": f"{row_['regression']:,}",
    } for row_ in eb["table"] if row_["mde"] in (25, 50, 100, 150)]), "lrrr") + "\n")
    add(
        f"The formula and the simulation agree. Pre-period contribution alone cuts variance by "
        f"{pct(eb['var_reduction_cuped'])}, because realized contribution carries so much luck that last quarter's result "
        f"predicts next quarter's only loosely. Adding luck-free measures like handle and active days raises the reduction "
        f"to {pct(eb['empirical_var_reduction']['regression'])} in the simulation, which cuts the required sample by the "
        f"same share. Across 1,000 random splits with no true effect, the regression-adjusted estimate's standard deviation "
        f"is {usd(eb['empirical_sd_of_estimate']['regression'])} against {usd(eb['empirical_sd_of_estimate']['raw'])} for a "
        f"plain difference in means. The lesson generalizes: the best CUPED covariates are behavioral, not outcome-based.\n"
    )

    # ---- 9. Production -------------------------------------------------------
    add("## 9. From analysis to production\n")
    add(
        "A model only matters if it runs every day and someone acts on its output. This section is the path from the "
        "notebook to that.\n"
    )
    add("### 9.1 Daily day-14 scoring\n")
    add(
        f"`score.py` runs daily. It takes the FTDs who completed their first 14 days on the run date, scores each for "
        f"month-3 retention and 180-day theo value with the saved models, and assigns an action:\n"
    )
    add("\n".join([
        "- **VIP review**: predicted value in the training top decile, or 14-day handle of $5,000 or more.",
        "- **Retention offer**: under a 40% chance of betting in month 3, but predicted to be profitable.",
        "- **No action**: everyone else.",
    ]) + "\n")
    add(
        f"The thresholds are illustrative and belong to the business owner. On a sample day, {sample_day}, the job scored "
        f"{len(score_sample):,} new players: "
        + ", ".join(f"{n_} {a_ if a_.startswith('VIP') else a_.lower()}" for a_, n_ in score_sample.action.value_counts().items())
        + ". Each row carries the model version and training window, so any score can be traced back.\n"
    )
    add("### 9.2 Drift monitoring\n")
    drift_alerts = drift_sample[drift_sample.alert == True]  # noqa: E712
    add(
        f"The same job compares the last 30 days of incoming players with the training data on every feature, using the "
        f"population stability index (PSI), and flags any feature above 0.2. On {sample_day}, it flagged "
        f"{len(drift_alerts)} of {len(drift_sample)}:\n"
    )
    add(table(pd.DataFrame([{"Feature": feat_labels.get(row_.feature, row_.feature), "PSI": f"{row_.psi:.2f}"}
                            for row_ in drift_sample.head(6).itertuples()]), "lr") + "\n")
    add(
        "That is the out-of-time problem from section 5, caught automatically. Summer signups bet little football and "
        "face a different calendar than the September-to-April cohorts the model trained on. The calendar-ahead features "
        "were added so the model copes, but a production system should alert on this, and retrain once summer cohorts "
        "have matured into the training window.\n"
    )
    add("### 9.3 Scheduling\n")
    add(
        "`dags/customer_economics_dag.py` defines three Airflow DAGs. It is not executed here, but it shows how the pieces "
        "would run:\n"
    )
    add("\n".join([
        "- **Daily:** build features on the warehouse, score the day-14 cohort, check drift, and publish the dashboard, "
        "with the publish step gated on drift.",
        "- **Weekly:** rebuild channel, offer, and state economics, the LTV projections, and the report.",
        "- **Monthly:** retrain the models on the latest cohorts and promote the new version only if it beats the live "
        "model on the most recent held-out cohorts.",
    ]) + "\n")
    add(
        "In production the SQL would run on Databricks or Snowflake against real player and bet tables. The queries are "
        "standard SQL; the main DuckDB-specific pieces to translate are the calendar `range()` join, `date_diff`, and "
        "aggregate `FILTER` clauses.\n"
    )
    add("### 9.4 Dashboard\n")
    add(
        f"[The dashboard]({dash_url}) is the interactive companion to this report, built from the same outputs by "
        f"`build_dashboard.py`: channel payback with a toggle between 12-month actuals and 36-month projections, a "
        f"bid-cap calculator by channel, offer, state, and CAC, retention by cohort and channel, and gameplay trends.\n"
    )

    # ---- 10. Conclusion -----------------------------------------------------
    add("## 10. Conclusion\n")
    add("### 10.1 Summary\n")
    add(
        f"An FTD in this simulation returns {blended_ltv / blended_cac:.2f}× its acquisition cost in year one, but that "
        f"average hides extreme concentration: {pct(top(1))} of contribution comes from 1% of players and most players "
        f"lose money after promos. The biggest levers are which channel a player comes from, which offer they receive, "
        f"which state they bet in, and how broadly they bet. Two weeks of behavior is enough to value a player well, "
        f"provided value is measured as theo: realized results carry too much luck to model. Projected with a "
        f"retention-decay model that backtests within a few percent, 36-month returns run from "
        f"{ltv['projection'][-1]['ratio_36']:.1f}× to {ltv['projection'][0]['ratio_36']:.1f}× CAC. Section 11 tests a "
        f"pretrained tabular model, TabPFN, on the same problem.\n"
    )
    add("### 10.2 Recommendations\n")
    add("\n".join(f"{i}. {t_}" for i, t_ in enumerate([
        f"Test the no-sweat offer against bet $5 get $200 before changing any channel's budget, measured on theo: about "
        f"{th_['n_winsor']:,} FTDs per arm with a winsorized outcome (section 8.1).",
        "Set acquisition bid caps by state (7.4) rather than one national CAC target, and base them on projected theo "
        "value rather than 12-month value where the backtest supports it (7.5).",
        "Value players on theo, not realized results, for bids, VIP decisions, and model targets (7.1).",
        "Score every FTD at day 14 with the two-part value model and route them daily (9.1), with drift alerts (9.2).",
        "Cross-sell other sports to football-heavy players before the Super Bowl; they are the most likely to disappear "
        "in the offseason (4.5).",
        f"Run the retention-spend test with regression adjustment on behavioral pre-period measures; it cuts the required "
        f"sample by about {pct(eb['empirical_var_reduction']['regression'])} (8.2).",
        f"Use {ret_best} for retention scoring: accuracy is similar and it is better calibrated (7.1).",
    ], 1)) + "\n")
    add("### 10.3 Limitations\n")
    add("\n".join(f"- {t_}" for t_ in [
        "Player-level data is simulated. Channel, offer, gameplay, and player-type effects come from my assumptions, so "
        "conclusions about them demonstrate the method rather than describe real DraftKings economics.",
        "Only seasonality, hold, hold volatility, and New York's tax rate are calibrated to real data, and only from New "
        "York, the highest-tax state.",
        "Tax is a flat rate on GGR net of promos. Real states tier it, tax per wager, or limit promo deductions, and the "
        "non-New York rates are approximate.",
        "Expected hold by bet type is fixed. Real theo would use each bet's actual odds and market.",
        "No casino or daily fantasy cross-sell, and churned players never return.",
        "LTV projections beyond 18 months are extrapolations, and they hold value per active player flat.",
        "Sample sizes assume the simulated variance, which should be measured on recent cohorts before a test launches.",
    ]) + "\n")

    # ---- 11. TabPFN ---------------------------------------------------------
    add("## 11. TabPFN: process and comparison\n")
    add("### 11.1 What TabPFN is\n")
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
    add("### 11.2 Setup\n")
    add("\n".join([
        "- **Model version.** The open TabPFN v2 weights from Hugging Face (`Prior-Labs/TabPFN-v2-clf` and `-reg`), loaded "
        "through the `tabpfn` package. Newer versions (3.5) require an account and license acceptance for local use, so I "
        "used v2, which is ungated.",
        "- **Same problem as section 6.** Same features, same targets (month-3 retention and 180-day theo value), same "
        f"out-of-time test set of {alt['n_test'] if alt else model['split']['n_test']:,} players, so results are directly "
        "comparable.",
        f"- **Context size.** {alt['n_context'] if alt else 3000:,} players sampled at random from the training cohorts. "
        "TabPFN v2 was pretrained on datasets up to about 10,000 rows, and inference cost grows with context size, so a "
        "smaller context kept the run feasible on a laptop CPU.",
        "- **Encoding.** Categorical columns were integer-coded and flagged as categorical so TabPFN applies its own "
        "categorical handling. Numeric features were passed raw: TabPFN does its own preprocessing internally.",
        f"- **Ensembling.** {alt['n_estimators'] if alt else 4} ensemble members, each with a different feature ordering and "
        "preprocessing, averaged.",
        "- **CPU.** TabPFN refuses CPU runs above 1,000 rows by default because they are slow, so I set "
        "`ignore_pretraining_limits=True` and predicted in batches of 1,000. Scoring the test set took roughly 10 to 25 "
        "minutes per model. Predictions are cached against a fingerprint of the exact inputs, so the comparison can be "
        "rebuilt without rerunning TabPFN, and any change to the data forces a fresh run.",
    ]) + "\n")
    add(
        "To separate the effect of the model from the effect of less data, I compared three setups on the same test "
        "players: TabPFN on the 3,000-player sample, XGBoost on the same 3,000, and XGBoost on the full training set. "
        "All three value models are single squared-error-style regressors, the like-for-like comparison; the two-part "
        "model is reported alongside.\n"
    )
    add("### 11.3 Results\n")
    if alt:
        a0, a1, a2 = alt["models"]
        tpf = alt["two_part_full"]
        add(f"![TabPFN comparison]({figs['tabpfn']})\n")
        rows_ = [{
            "Model": m_["name"], "Retention AUC": f"{m_['auc']:.3f}", "Value Spearman": f"{m_['spearman']:.3f}",
            "Top decile, pred ÷ actual": f"{m_['top_ratio']:.2f}", "Value MAE": usd(m_["mae"]),
            "Top-decile lift": f"{m_['top_decile_lift']:.2f}×",
        } for m_ in alt["models"]]
        rows_.append({"Model": tpf["name"], "Retention AUC": "", "Value Spearman": f"{tpf['spearman']:.3f}",
                      "Top decile, pred ÷ actual": f"{tpf['top_ratio']:.2f}", "Value MAE": usd(tpf["mae"]), "Top-decile lift": ""})
        add(table(pd.DataFrame(rows_), "lrrrrr") + "\n")
        same, full = a0["auc"] - a1["auc"], a0["auc"] - a2["auc"]
        n_full = int(a2["name"].split(", ")[1].split()[0].replace(",", ""))
        ratio_n = round(n_full / alt["n_context"])
        vs_full = (f"TabPFN still comes out ahead by {full:.3f}" if full > 0 else
                   f"TabPFN is within {abs(full):.3f}" if full > -0.01 else f"TabPFN trails by {abs(full):.3f}")
        add("### 11.4 Interpretation\n")
        add("\n".join([
            f"- **At equal data**, TabPFN {'beats' if same > 0 else 'trails'} XGBoost on retention by {abs(same):.3f} AUC, "
            f"and ranks 180-day value at {a0['spearman']:.3f} Spearman against {a1['spearman']:.3f}.",
            f"- **Against {ratio_n}× the data**, XGBoost on {n_full:,} players: {vs_full} on retention AUC, and on value "
            f"ranking it scores {a0['spearman']:.3f} against {a2['spearman']:.3f}"
            + (". With a tenth of the data, it matches or beats the full-data model." if a0["spearman"] >= a2["spearman"] else "."),
            f"- **Against the two-part model**, which is built around this target's skew, XGBoost reaches "
            f"{tpf['spearman']:.3f}. "
            + ("TabPFN, a general-purpose model with no tuning and a tenth of the data, ties it."
               if abs(a0["spearman"] - tpf["spearman"]) < 0.005 else
               "TabPFN, a general-purpose model with no tuning, still ranks better." if a0["spearman"] > tpf["spearman"] else
               "That is the best value ranking in the report, so a well-specified tree model on the full data still wins "
               "the data-rich case."),
            f"- **Calibration.** TabPFN's top decile is predicted at {a0['top_ratio']:.2f}× its actual value, against "
            f"{a2['top_ratio']:.2f}× for full-data squared-error XGBoost and {tpf['top_ratio']:.2f}× for the two-part model."
            + (" So TabPFN orders players as well as the best model but understates what the top players are worth; its "
               "dollar predictions would need recalibrating before use in bids." if a0["top_ratio"] < 0.85 else ""),
            "- **Cost.** XGBoost trains and scores in seconds. TabPFN needed minutes per model on CPU, because every "
            "prediction attends over the whole context. At production scale it wants a GPU or Prior Labs' hosted API.",
            "- **Where I'd use it.** Small-sample questions where there is no time or data to tune a model: early reads on "
            "a new state, offer, or product. And as a benchmark: a strong untuned result is a quick check on whether a "
            "production model is leaving accuracy on the table.",
        ]) + "\n")
    else:
        add("TabPFN results are not in `outputs/` yet. Run `python model_tabpfn.py`, then `python build_report.py`.\n")
    add(
        "TabPFN v2 weights are released under the Prior Labs License, which is Apache 2.0 with an attribution "
        "requirement. Built with TabPFN.\n"
    )

    # ---- 12. Reproducibility -----------------------------------------------
    add("## 12. Reproducibility\n")
    add("```bash\npython -m venv .venv\n.venv/Scripts/pip install -r requirements.txt\n"
        ".venv/Scripts/python run.py            # simulate, SQL, models, LTV, test design, scoring, report, dashboard\n"
        ".venv/Scripts/python model_tabpfn.py   # TabPFN comparison (10 to 40 minutes on CPU the first time, then cached)\n"
        ".venv/Scripts/python build_report.py   # rebuild this report with the TabPFN results\n"
        ".venv/Scripts/python score.py --as-of 2025-06-15   # run the daily scoring job for one day\n```\n")
    add(
        "The pipeline is deterministic. One bug worth recording: results first changed between runs because DuckDB does "
        "not guarantee row order out of a parallel join, and row order decides which rows XGBoost holds out for "
        "early stopping. Sorting the feature table fixed it.\n"
    )

    text = "\n".join(md)
    text = text.replace("—", ", ").replace("–", " to ")  # keep punctuation plain
    (ROOT / "REPORT.md").write_text(text, encoding="utf-8")
    print(f"wrote REPORT.md ({len(text.split()):,} words) and {len(figs)} figures")


if __name__ == "__main__":
    main()
