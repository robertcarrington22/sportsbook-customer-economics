"""Build report/index.html from outputs/: findings, charts, and a CAC payback simulator."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
REPORT = ROOT / "report"


def weighted(df: pd.DataFrame, col: str, w: str = "ftds") -> float:
    return float((df[col] * df[w]).sum() / df[w].sum())


def main() -> None:
    ch = pd.read_csv(OUT / "04_channel_economics.csv")
    offers = pd.read_csv(OUT / "05_offer_economics.csv")
    states = pd.read_csv(OUT / "06_state_economics.csv")
    curves = pd.read_csv(OUT / "03_ltv_curves.csv")
    ret = pd.read_csv(OUT / "02_cohort_retention.csv")
    sim = pd.read_csv(OUT / "07_simulator_curves.csv")
    feats = pd.read_parquet(OUT / "features.parquet")
    model = json.loads((OUT / "model_results.json").read_text())

    blended_cac = weighted(ch, "cac")
    blended_ltv = weighted(ch, "ltv_12m")

    # Simulator curves, plus an "all offers" blend per channel.
    sim_all = (
        sim.assign(ngr_w=sim.ngr_per_ftd * sim.ftds, var_w=sim.variable_cost_per_ftd * sim.ftds, cac_w=sim.cac_per_ftd * sim.ftds)
        .groupby(["channel", "life_month"], as_index=False)
        .agg(ftds=("ftds", "sum"), ngr_w=("ngr_w", "sum"), var_w=("var_w", "sum"), cac_w=("cac_w", "sum"))
    )
    sim_all["offer"] = "all"
    sim_all["ngr_per_ftd"] = sim_all.ngr_w / sim_all.ftds
    sim_all["variable_cost_per_ftd"] = sim_all.var_w / sim_all.ftds
    sim_all["cac_per_ftd"] = sim_all.cac_w / sim_all.ftds
    sim_full = pd.concat([sim, sim_all[sim.columns]], ignore_index=True)
    sim_data: dict = {}
    for (c, o), g in sim_full.groupby(["channel", "offer"]):
        g = g.sort_values("life_month")
        sim_data.setdefault(c, {})[o] = {
            "ngr": g.ngr_per_ftd.round(2).tolist(),
            "var": g.variable_cost_per_ftd.round(2).tolist(),
            "cac": round(float(g.cac_per_ftd.iloc[0]), 0),
            "ftds": int(g.ftds.iloc[0]),
        }

    heat = ret.pivot(index="cohort_month", columns="life_month", values="active_rate")
    ny, mi = states.set_index("state").loc["NY"], states.set_index("state").loc["MI"]
    v, r = model["value"], model["retention"]
    best, worst = ch.iloc[0], ch.sort_values("ltv_to_cac").iloc[0]
    bet5 = offers.set_index("offer").loc["bet5_get200"]
    nosweat = offers.set_index("offer").loc["no_sweat_1000"]

    data = {
        "kpi": {
            "ftds": int(feats.shape[0]),
            "blended_cac": round(blended_cac),
            "blended_ltv": round(blended_ltv),
            "ltv_to_cac": round(blended_ltv / blended_cac, 2),
            "net_negative_180": round(float((feats.contribution_180 < 0).mean()), 3),
        },
        "channels": ch.fillna({"payback_month": -1}).to_dict("records"),
        "offers": offers.to_dict("records"),
        "states": states.to_dict("records"),
        "curves": {c: g.sort_values("life_month").cum_contribution_per_ftd.tolist() for c, g in curves.groupby("channel")},
        "curve_cac": {c: float(g.cac_per_ftd.iloc[0]) for c, g in curves.groupby("channel")},
        "heat": {
            "cohorts": [pd.Timestamp(x).strftime("%b %Y") for x in heat.index],
            "months": [int(m) + 1 for m in heat.columns],
            "z": [[None if pd.isna(x) else round(float(x), 3) for x in row] for row in heat.values],
        },
        "model": model,
        "sim": sim_data,
        "tax": {row.state: row.tax_rate for row in states.itertuples()},
        "findings": {
            "best": best.channel, "best_ratio": float(best.ltv_to_cac), "best_payback": int(best.payback_month),
            "worst": worst.channel, "worst_ratio": float(worst.ltv_to_cac),
            "ny_contrib": float(ny.contribution_per_ftd), "mi_contrib": float(mi.contribution_per_ftd),
            "ny_ratio": float(ny.ltv_to_cac), "mi_ratio": float(mi.ltv_to_cac),
            "bet5_cost": float(bet5.welcome_cost_per_ftd), "bet5_ltv": float(bet5.ltv_12m),
            "nosweat_cost": float(nosweat.welcome_cost_per_ftd), "nosweat_ltv": float(nosweat.ltv_12m),
        },
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    REPORT.mkdir(exist_ok=True)
    (REPORT / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {REPORT / 'index.html'}  ({len(html) / 1024:.0f} KB)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Sportsbook Customer Economics</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.3/plotly.min.js"></script>
<style>
:root{
  --ground:#F3F4F1; --card:#FFFFFF; --ink:#121A21; --ink-2:#48525C; --ink-3:#7D8790;
  --line:#D8DCD6; --line-2:#E8EBE6;
  --win:#16794C; --win-soft:#DDF0E6; --loss:#B23A2E; --loss-soft:#F7E0DC; --accent:#1D4E89;
  --mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
  --body:'Source Sans 3',system-ui,-apple-system,Segoe UI,sans-serif;
  --disp:'Archivo',var(--body);
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--body);font-size:17px;line-height:1.55}
.wrap{max-width:1080px;margin:0 auto;padding-inline:20px;padding-block:44px 80px}
header{border-bottom:2px solid var(--ink);padding-bottom:26px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
h1{font-family:var(--disp);font-weight:800;font-size:clamp(34px,5.4vw,58px);line-height:1;letter-spacing:-.02em;margin:10px 0 14px}
.lede{font-size:19px;color:var(--ink-2);max-width:66ch;margin:0}
.note{margin-top:16px;font-size:14.5px;color:var(--ink-2);background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);padding:10px 14px;max-width:80ch}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin:28px 0 8px}
.kpi{background:var(--card);padding:16px 18px}
.kpi .v{font-family:var(--disp);font-weight:800;font-size:34px;letter-spacing:-.02em;font-variant-numeric:tabular-nums;line-height:1.05}
.kpi .l{font-size:13.5px;color:var(--ink-2);margin-top:4px}
section{padding-block:38px;border-bottom:1px solid var(--line)}
h2{font-family:var(--disp);font-weight:800;font-size:clamp(24px,3vw,32px);letter-spacing:-.015em;margin:0 0 6px}
h2 small{display:block;font-family:var(--mono);font-size:12px;font-weight:400;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);margin-bottom:8px}
.sub{color:var(--ink-2);max-width:70ch;margin:0 0 18px}
.findings{list-style:none;padding:0;margin:0;display:grid;gap:12px}
.findings li{background:var(--card);border:1px solid var(--line);padding:14px 18px;display:grid;grid-template-columns:auto 1fr;gap:14px;align-items:start}
.findings .n{font-family:var(--mono);font-size:13px;color:var(--ink-3);padding-top:3px}
.findings b{font-weight:600}
.chart{background:var(--card);border:1px solid var(--line);padding:8px 6px 2px;margin:14px 0}
.tbl{overflow-x:auto;background:var(--card);border:1px solid var(--line);margin:14px 0}
table{border-collapse:collapse;width:100%;min-width:640px;font-size:15px}
th,td{padding:9px 14px;text-align:right;border-bottom:1px solid var(--line-2);font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}
th{font-family:var(--mono);font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);font-weight:500;background:#FAFBF9}
tr:last-child td{border-bottom:0}
.pos{color:var(--win)} .neg{color:var(--loss)}
.pill{display:inline-block;font-family:var(--mono);font-size:12px;padding:1px 8px;border-radius:999px}
.pill.pos{background:var(--win-soft)} .pill.neg{background:var(--loss-soft)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media (max-width:760px){.two{grid-template-columns:1fr}}
.sim{background:var(--card);border:1px solid var(--line);padding:18px}
.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:10px}
label{display:block;font-family:var(--mono);font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);margin-bottom:4px}
select,input[type=range]{width:100%;font:inherit;font-size:15px}
select{padding:6px 8px;border:1px solid var(--line);background:var(--ground);color:var(--ink)}
input[type=range]{accent-color:var(--accent)}
.out{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin:8px 0 4px}
.out div{background:var(--ground);padding:10px 14px}
.out .v{font-family:var(--disp);font-weight:800;font-size:26px;font-variant-numeric:tabular-nums}
.out .l{font-size:13px;color:var(--ink-2)}
.caveats{columns:2 320px;column-gap:28px;font-size:15px;color:var(--ink-2)}
.caveats p{break-inside:avoid;margin:0 0 12px}
code{font-family:var(--mono);font-size:.88em;background:var(--line-2);padding:1px 5px}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="eyebrow">Customer economics · simulated US sportsbook · Robert Carrington</div>
  <h1>What is a new depositor worth?</h1>
  <p class="lede">Acquisition cost, twelve-month contribution, payback by channel, the price of welcome offers, the effect of state tax, and a model that estimates a player's value from their first two weeks.</p>
  <p class="note"><b>The data is simulated.</b> No real operator data is used. Every assumption, from channel costs to hold rates, lives in one file, <code>assumptions.toml</code>. The SQL and models are written to run unchanged on real FTD and bet-level tables. The findings show the method, not DraftKings' actual economics.</p>
  <div class="kpis" id="kpis"></div>
</header>

<section>
  <h2><small>Memo</small>Four findings</h2>
  <ul class="findings" id="findings"></ul>
</section>

<section>
  <h2><small>Channels</small>Payback by acquisition channel</h2>
  <p class="sub">Cumulative contribution per FTD after promos, gaming tax, and variable cost. The dashed line is the channel's cost to acquire. Payback is where the solid line crosses it.</p>
  <div class="chart" id="c-curves"></div>
  <div class="tbl"><table id="t-channels"></table></div>
</section>

<section>
  <h2><small>Offers and tax</small>Two levers that move value before the player places a bet</h2>
  <div class="two">
    <div>
      <p class="sub">The richest welcome offer costs the most and brings the weakest players.</p>
      <div class="chart" id="c-offers"></div>
    </div>
    <div>
      <p class="sub">The same betting behavior is worth half as much in a 51% tax state.</p>
      <div class="chart" id="c-states"></div>
    </div>
  </div>
</section>

<section>
  <h2><small>Retention</small>Share of each cohort still betting</h2>
  <p class="sub">Each row is a monthly signup cohort. Each column is a 30-day month since first deposit. Blank cells have not happened yet.</p>
  <div class="chart" id="c-heat"></div>
</section>

<section>
  <h2><small>Model</small>Estimating value from the first 14 days</h2>
  <p class="sub" id="model-sub"></p>
  <div class="two">
    <div class="chart" id="c-deciles"></div>
    <div class="chart" id="c-importance"></div>
  </div>
  <div class="tbl"><table id="t-model"></table></div>
</section>

<section>
  <h2><small>Tool</small>CAC payback simulator</h2>
  <p class="sub">Pick a channel, offer, and state, then move the acquisition cost. Contribution applies that state's tax rate to the simulated net gaming revenue.</p>
  <div class="sim">
    <div class="controls">
      <div><label for="s-channel">Channel</label><select id="s-channel"></select></div>
      <div><label for="s-offer">Welcome offer</label><select id="s-offer"></select></div>
      <div><label for="s-state">State</label><select id="s-state"></select></div>
      <div><label for="s-cac">CAC per FTD: <span id="s-cac-v"></span></label><input type="range" id="s-cac" min="50" max="900" step="10"></div>
    </div>
    <div class="out" id="s-out"></div>
    <div id="c-sim"></div>
  </div>
</section>

<section>
  <h2><small>Method</small>How it's built, and what it doesn't know</h2>
  <div class="caveats">
    <p><b>Pipeline.</b> <code>simulate.py</code> generates 40,000 first-time depositors and every day they bet. <code>sql/</code> builds a player-month fact table in DuckDB and derives retention, LTV curves, and channel, offer, and state economics. <code>model.py</code> trains on Sep 2024 to Apr 2025 cohorts and tests on May to Aug 2025, an out-of-time split.</p>
    <p><b>Contribution</b> is gross gaming revenue, minus welcome and ongoing promos, minus gaming tax on the remainder, minus variable cost at 0.6% of handle. It excludes fixed costs, brand spend, and overhead.</p>
    <p><b>Censoring.</b> Every curve and 12-month figure uses only players with 12 full months of history, so younger cohorts never drag an average down.</p>
    <p><b>Simplifications.</b> Tax is a flat rate on net revenue, though several states tier it or tax per wager. No casino or daily fantasy cross-sell. Players never return after churning.</p>
    <p><b>Validation.</b> Because the data is simulated, each player's hidden type is known. The analysis never uses it. It is only used to check that the value model finds the players it should.</p>
    <p><b>Next.</b> Model log value to fix the low bias on the top decile. Add reactivation and cross-sell. Point the SQL at real tables.</p>
  </div>
</section>
</div>

<script>
const D = __DATA__;
const C = {ink:'#121A21', ink2:'#48525C', ink3:'#7D8790', line:'#D8DCD6', win:'#16794C', loss:'#B23A2E', accent:'#1D4E89'};
const CH_COLORS = {referral:'#16794C', tv_brand:'#1D4E89', search:'#7A5AA6', affiliate:'#C27C0E', paid_social:'#B23A2E'};
const NAMES = {referral:'Referral', tv_brand:'TV / brand', search:'Search', affiliate:'Affiliate', paid_social:'Paid social',
  bet5_get200:'Bet $5, get $200', no_sweat_1000:'No-sweat first bet', deposit_match_250:'Deposit match', all:'All offers'};
const nm = k => NAMES[k] || k;
const usd = x => (x < 0 ? '−$' : '$') + Math.abs(Math.round(x)).toLocaleString();
const pct = x => (x * 100).toFixed(0) + '%';
const base = {font:{family:'Source Sans 3, sans-serif', size:13, color:C.ink2}, paper_bgcolor:'#fff', plot_bgcolor:'#fff',
  margin:{l:62, r:18, t:18, b:48}, xaxis:{gridcolor:'#EEF0EC', zerolinecolor:C.line, linecolor:C.line},
  yaxis:{gridcolor:'#EEF0EC', zerolinecolor:'#B9BFB6', linecolor:C.line}, hoverlabel:{font:{family:'JetBrains Mono, monospace', size:12}},
  legend:{orientation:'h', y:-0.2}};
const cfg = {displayModeBar:false, responsive:true};
const L = extra => Object.assign({}, JSON.parse(JSON.stringify(base)), extra);

// KPIs
const k = D.kpi;
document.getElementById('kpis').innerHTML = [
  [k.ftds.toLocaleString(), 'first-time depositors simulated'],
  [usd(k.blended_cac), 'blended cost per FTD'],
  [usd(k.blended_ltv), '12-month contribution per FTD'],
  [k.ltv_to_cac.toFixed(2) + '×', 'blended LTV to CAC, year one'],
  [pct(k.net_negative_180), 'of FTDs net negative at 180 days'],
].map(([v,l]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');

// Findings
const f = D.findings, m = D.model, v = m.value, r = m.retention;
const items = [
  `<b>Channel mix matters more than channel cost.</b> ${nm(f.best)} returns ${f.best_ratio.toFixed(2)}× its CAC and pays back in month ${f.best_payback + 1}. ${nm(f.worst)} is one of the cheaper channels per FTD but returns ${f.worst_ratio.toFixed(2)}× and does not pay back inside a year. Cheap FTDs are not cheap if they never bet again.`,
  `<b>The richest welcome offer is the worst deal.</b> Bet $5, get $200 costs ${usd(f.bet5_cost)} per FTD and its players are worth ${usd(f.bet5_ltv)} over a year. The no-sweat first bet costs ${usd(f.nosweat_cost)} and brings players worth ${usd(f.nosweat_ltv)}. The generous offer attracts bonus hunters who leave once it's spent.`,
  `<b>Tax rate changes what a player is worth by 2×.</b> The same behavior yields ${usd(f.ny_contrib)} of 12-month contribution in New York at a 51% rate and ${usd(f.mi_contrib)} in Michigan. At a single national CAC, New York returns ${f.ny_ratio.toFixed(2)}× and Michigan ${f.mi_ratio.toFixed(2)}×. Bid caps should be set by state.`,
  `<b>Two weeks is enough to find most of the value.</b> ${pct(k.net_negative_180)} of depositors are net negative after 180 days, and the top decile carries nearly all of the profit. A model on the first 14 days ranks players better than sorting by handle (rank correlation ${v.spearman_model} vs ${v.spearman_heuristic}), though for picking out the top decile, simple handle does as well. The model's top decile is ${pct(v.hidden_high_value_share_top_decile)} true high-value players, against a ${pct(v.hidden_high_value_share_all)} base rate.`,
];
document.getElementById('findings').innerHTML = items.map((t,i) => `<li><span class="n">0${i+1}</span><span>${t}</span></li>`).join('');

// Channel curves
const months = [...Array(12).keys()].map(i => i + 1);
const tr = [];
for (const [c, ys] of Object.entries(D.curves)) {
  tr.push({x:months, y:ys, name:nm(c), mode:'lines', line:{color:CH_COLORS[c], width:3},
    hovertemplate:`${nm(c)} · month %{x}<br>%{y:$,.0f} per FTD<extra></extra>`});
  tr.push({x:[1,12], y:[D.curve_cac[c], D.curve_cac[c]], mode:'lines', line:{color:CH_COLORS[c], width:1.5, dash:'dash'},
    showlegend:false, hovertemplate:`${nm(c)} CAC %{y:$,.0f}<extra></extra>`});
}
Plotly.newPlot('c-curves', tr, L({height:420, xaxis:{...base.xaxis, title:'Month since first deposit', dtick:1},
  yaxis:{...base.yaxis, title:'Cumulative contribution per FTD', tickprefix:'$'}}), cfg);

document.getElementById('t-channels').innerHTML =
  '<tr><th>Channel</th><th>FTDs</th><th>CAC</th><th>12-mo LTV</th><th>Median LTV</th><th>LTV / CAC</th><th>Active month 3</th><th>Payback</th></tr>' +
  D.channels.map(c => `<tr><td>${nm(c.channel)}</td><td>${c.ftds.toLocaleString()}</td><td>${usd(c.cac)}</td><td>${usd(c.ltv_12m)}</td>
    <td class="${c.median_ltv_12m < 0 ? 'neg' : ''}">${usd(c.median_ltv_12m)}</td>
    <td><span class="pill ${c.ltv_to_cac >= 1 ? 'pos' : 'neg'}">${c.ltv_to_cac.toFixed(2)}×</span></td>
    <td>${pct(c.active_m3_rate)}</td><td>${c.payback_month < 0 ? '<span class="neg">not in 12 mo</span>' : 'month ' + (c.payback_month + 1)}</td></tr>`).join('');

// Offers
const off = D.offers;
Plotly.newPlot('c-offers', [
  {x:off.map(o => nm(o.offer)), y:off.map(o => o.welcome_cost_per_ftd), name:'Offer cost per FTD', type:'bar', marker:{color:C.loss},
   hovertemplate:'%{x}<br>cost %{y:$,.0f}<extra></extra>'},
  {x:off.map(o => nm(o.offer)), y:off.map(o => o.ltv_12m), name:'12-month contribution per FTD', type:'bar', marker:{color:C.win},
   hovertemplate:'%{x}<br>12-mo LTV %{y:$,.0f}<extra></extra>'},
], L({height:340, barmode:'group', yaxis:{...base.yaxis, tickprefix:'$'}}), cfg);

// States
const st = [...D.states].sort((a,b) => a.tax_rate - b.tax_rate);
Plotly.newPlot('c-states', [{
  x:st.map(s => s.state), y:st.map(s => s.contribution_per_ftd), type:'bar',
  marker:{color:st.map(s => s.tax_rate >= 0.3 ? C.loss : C.accent)},
  text:st.map(s => pct(s.tax_rate)), textposition:'outside', cliponaxis:false,
  hovertemplate:'%{x} · tax %{text}<br>%{y:$,.0f} per FTD<extra></extra>'}],
  L({height:340, showlegend:false, yaxis:{...base.yaxis, title:'12-mo contribution per FTD', tickprefix:'$'},
     xaxis:{...base.xaxis, title:'State, labeled with tax rate'}}), cfg);

// Heatmap
Plotly.newPlot('c-heat', [{z:D.heat.z, x:D.heat.months, y:D.heat.cohorts, type:'heatmap',
  colorscale:[[0,'#F4F6F2'],[0.5,'#7FB89A'],[1,'#0E5A36']], zmin:0, zmax:1, xgap:2, ygap:2,
  colorbar:{tickformat:'.0%', thickness:12, outlinewidth:0},
  hovertemplate:'%{y} cohort · month %{x}<br>%{z:.0%} active<extra></extra>'}],
  L({height:440, margin:{l:80, r:18, t:10, b:48}, xaxis:{...base.xaxis, title:'Month since first deposit', dtick:1},
     yaxis:{...base.yaxis, autorange:'reversed'}}), cfg);

// Model
document.getElementById('model-sub').textContent =
  `Trained on ${m.split.train_cohorts} signups (${m.split.n_train.toLocaleString()} players), tested on ${m.split.test_cohorts} (${m.split.n_test.toLocaleString()}). ` +
  `Each bar is a tenth of test players, ranked by predicted 180-day contribution.`;
const dec = v.deciles;
Plotly.newPlot('c-deciles', [
  {x:dec.map(d => d.decile), y:dec.map(d => d.actual), type:'bar', name:'Actual', marker:{color:dec.map(d => d.actual < 0 ? C.loss : C.win)},
   hovertemplate:'decile %{x}<br>actual %{y:$,.0f}<extra></extra>'},
  {x:dec.map(d => d.decile), y:dec.map(d => d.predicted), mode:'lines+markers', name:'Predicted', line:{color:C.ink, width:2},
   hovertemplate:'decile %{x}<br>predicted %{y:$,.0f}<extra></extra>'},
], L({height:360, xaxis:{...base.xaxis, title:'Predicted-value decile (1 = highest)', dtick:1},
     yaxis:{...base.yaxis, title:'180-day contribution per player', tickprefix:'$'}}), cfg);

const imp = [...v.importance].filter(d => d.importance > 0.001).reverse();
Plotly.newPlot('c-importance', [{y:imp.map(d => d.feature), x:imp.map(d => d.importance), type:'bar', orientation:'h',
  marker:{color:C.accent}, hovertemplate:'%{y}<br>%{x:.3f} drop in R²<extra></extra>'}],
  L({height:360, margin:{l:170, r:18, t:18, b:48}, showlegend:false, xaxis:{...base.xaxis, title:'Permutation importance (drop in R²)'}}), cfg);

const td = v.top_decile_model, th = v.top_decile_heuristic;
document.getElementById('t-model').innerHTML =
  '<tr><th>Question</th><th>Metric</th><th>Model</th><th>Baseline</th></tr>' +
  `<tr><td>Still betting in month 3?</td><td>AUC</td><td>${r.auc_gbm}</td><td>${r.auc_logistic} logistic</td></tr>` +
  `<tr><td>Flag the at-risk tenth</td><td>Lapse rate in bottom decile</td><td>${pct(r.lapse_rate_bottom_decile)}</td><td>${pct(r.lapse_rate_all)} overall</td></tr>` +
  `<tr><td>Rank everyone by 180-day value</td><td>Spearman correlation</td><td>${v.spearman_model}</td><td>${v.spearman_heuristic} handle only</td></tr>` +
  `<tr><td>Find the top tenth</td><td>Lift over average</td><td>${td.lift}×</td><td>${th.lift}× handle only</td></tr>` +
  `<tr><td>Predict dollar value</td><td>Mean absolute error</td><td>${usd(v.mae_model)}</td><td>${usd(v.mae_predict_mean)} predict the mean</td></tr>` +
  `<tr><td>Recover hidden whales</td><td>High-value share, top decile</td><td>${pct(v.hidden_high_value_share_top_decile)}</td><td>${pct(v.hidden_high_value_share_all)} base rate</td></tr>`;

// Simulator
const sc = document.getElementById('s-channel'), so = document.getElementById('s-offer'),
      ss = document.getElementById('s-state'), sr = document.getElementById('s-cac');
Object.keys(D.sim).forEach(c => sc.add(new Option(nm(c), c)));
['all','bet5_get200','no_sweat_1000','deposit_match_250'].forEach(o => so.add(new Option(nm(o), o)));
Object.entries(D.tax).sort((a,b) => a[1] - b[1]).forEach(([s,t]) => ss.add(new Option(`${s} · ${pct(t)} tax`, s)));
sc.value = 'paid_social'; so.value = 'all'; ss.value = 'NY';
function setDefaultCac(){ sr.value = D.sim[sc.value][so.value].cac; }
function runSim(){
  const d = D.sim[sc.value][so.value], tax = D.tax[ss.value], cac = +sr.value;
  document.getElementById('s-cac-v').textContent = usd(cac);
  let cum = 0, pay = -1; const ys = [];
  d.ngr.forEach((n, i) => { cum += n * (1 - tax) - d.var[i]; ys.push(cum); if (pay < 0 && cum >= cac) pay = i; });
  const ltv = ys[ys.length - 1];
  document.getElementById('s-out').innerHTML = [
    [usd(ltv), '12-month contribution per FTD'],
    [(ltv / cac).toFixed(2) + '×', 'LTV to CAC'],
    [pay < 0 ? 'none' : 'month ' + (pay + 1), 'payback'],
    [usd(Math.max(0, ltv)), 'break-even CAC'],
  ].map(([v,l]) => `<div><div class="v ${l === 'LTV to CAC' ? (ltv >= cac ? 'pos' : 'neg') : ''}">${v}</div><div class="l">${l}</div></div>`).join('');
  Plotly.react('c-sim', [
    {x:months, y:ys, mode:'lines+markers', name:'Cumulative contribution', line:{color:C.win, width:3},
     hovertemplate:'month %{x}<br>%{y:$,.0f}<extra></extra>'},
    {x:[1,12], y:[cac,cac], mode:'lines', name:'CAC', line:{color:C.loss, dash:'dash', width:2}, hovertemplate:'CAC %{y:$,.0f}<extra></extra>'},
  ], L({height:340, xaxis:{...base.xaxis, title:'Month since first deposit', dtick:1},
        yaxis:{...base.yaxis, tickprefix:'$'}}), cfg);
}
sc.onchange = so.onchange = () => { setDefaultCac(); runSim(); };
ss.onchange = sr.oninput = runSim;
setDefaultCac(); runSim();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
