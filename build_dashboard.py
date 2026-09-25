"""Build docs/index.html: an interactive dashboard for GitHub Pages, from outputs/."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"


def main() -> None:
    ch = pd.read_csv(OUT / "04_channel_economics.csv")
    curves = pd.read_csv(OUT / "03_ltv_curves.csv")
    ret = pd.read_csv(OUT / "02_cohort_retention.csv")
    ret_ch = pd.read_csv(OUT / "14_retention_by_channel.csv")
    sim = pd.read_csv(OUT / "07_simulator_curves.csv")
    states = pd.read_csv(OUT / "06_state_economics.csv")
    bt = pd.read_csv(OUT / "17_gameplay_bet_types.csv")
    sp = pd.read_csv(OUT / "18_gameplay_sports.csv")
    gseg = pd.read_csv(OUT / "19_gameplay_segments.csv")
    seg = pd.read_csv(OUT / "15_day14_segments.csv")
    ltv = json.loads((OUT / "ltv_forecast.json").read_text())
    model = json.loads((OUT / "model_results.json").read_text())
    feats = pd.read_parquet(OUT / "features.parquet", columns=["contribution_180"])

    w = ch.ftds
    kpi = {
        "ftds": int(len(feats)),
        "cac": round(float((ch.cac * w).sum() / w.sum())),
        "ltv12": round(float((ch.ltv_12m * w).sum() / w.sum())),
        "neg180": round(float((feats.contribution_180 < 0).mean()), 3),
        "auc": model["retention"]["auc_gbm"],
        "spearman": model["value"]["spearman_model"],
    }
    kpi["ratio"] = round(kpi["ltv12"] / kpi["cac"], 2)

    sim_data: dict = {}
    for (c, o), g in sim.groupby(["channel", "offer"]):
        g = g.sort_values("life_month")
        sim_data.setdefault(c, {})[o] = {"ngr": g.ngr_per_ftd.round(2).tolist(), "var": g.variable_cost_per_ftd.round(2).tolist(),
                                         "cac": round(float(g.cac_per_ftd.iloc[0])), "ftds": int(g.ftds.iloc[0])}
    for c, offers in sim_data.items():
        tot = sum(v["ftds"] for v in offers.values())
        offers["all"] = {
            k: [round(sum(v[k][i] * v["ftds"] for v in offers.values()) / tot, 2) for i in range(12)] for k in ("ngr", "var")
        } | {"cac": round(sum(v["cac"] * v["ftds"] for v in offers.values()) / tot), "ftds": tot}

    heat = ret.pivot(index="cohort_month", columns="life_month", values="active_rate")
    bt["month"] = pd.to_datetime(bt["month"]).dt.strftime("%Y-%m")
    sp["month"] = pd.to_datetime(sp["month"]).dt.strftime("%Y-%m")
    bt = bt[bt.month >= "2024-10"]
    sp = sp[sp.month >= "2024-10"]
    hold = bt.groupby("month").apply(lambda g: g.ggr.sum() / g.handle.sum(), include_groups=False)

    data = {
        "kpi": kpi,
        "channels": ch.fillna({"payback_month": -1}).to_dict("records"),
        "curves12": {c: g.sort_values("life_month").cum_contribution_per_ftd.tolist() for c, g in curves.groupby("channel")},
        "cac": {c: float(g.cac_per_ftd.iloc[0]) for c, g in curves.groupby("channel")},
        "proj": {p["channel"]: {"cum": p["cum_curve"], "lo": p["cum_lo"], "hi": p["cum_hi"], "ratio36": p["ratio_36"],
                                "ltv36": p["ltv_36"]} for p in ltv["projection"]},
        "heat": {"cohorts": [pd.Timestamp(x).strftime("%b %Y") for x in heat.index], "months": [int(m) + 1 for m in heat.columns],
                 "z": [[None if pd.isna(v) else round(float(v), 3) for v in row] for row in heat.values]},
        "ret_ch": {c: g.sort_values("life_month").active_rate.tolist() for c, g in ret_ch.groupby("channel")},
        "sim": sim_data,
        "tax": {r.state: r.tax_rate for r in states.itertuples()},
        "bt": {b: {"months": g.month.tolist(), "handle": g.handle.tolist()} for b, g in bt.groupby("bet_type")},
        "hold": {"months": hold.index.tolist(), "hold": [round(float(x), 4) for x in hold.values]},
        "sports": {s: {"months": g.month.tolist(), "handle": g.handle.tolist()} for s, g in sp.groupby("sport")},
        "gseg": gseg.to_dict("records"),
        "seg": seg.to_dict("records"),
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(TEMPLATE.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    (DOCS / ".nojekyll").write_text("")
    print(f"wrote {DOCS / 'index.html'}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sportsbook Customer Economics Dashboard</title>
<meta name="description" content="Channel payback, bid caps, retention, and gameplay trends for a simulated US sportsbook calibrated to DraftKings' public New York filings.">
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.3/plotly.min.js"></script>
<style>
:root{--bg:#F6F6F3;--card:#FFFFFF;--ink:#16202A;--ink2:#4A5560;--ink3:#7B858F;--line:#D9DCD6;--line2:#ECEEEA;
  --win:#1E7A4F;--loss:#B23A2E;--accent:#1D4E89;--font:'Times New Roman',Times,Georgia,serif;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);font-size:17px;line-height:1.5;padding:0 18px}
.wrap{max-width:1180px;margin:0 auto;padding:36px 0 64px}
h1{font-size:clamp(28px,4vw,40px);margin:0 0 6px;font-weight:bold}
.sub{color:var(--ink2);margin:0 0 6px;max-width:80ch}
.note{font-size:14.5px;color:var(--ink2);border-left:3px solid var(--accent);padding-left:12px;margin:14px 0 0;max-width:90ch}
.note a{color:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:var(--line);border:1px solid var(--line);margin:24px 0 8px}
@media(max-width:900px){.kpis{grid-template-columns:repeat(2,1fr)}}
.kpi{background:var(--card);padding:14px 16px}.kpi .v{font-size:28px;font-weight:bold;font-variant-numeric:tabular-nums}.kpi .l{font-size:14px;color:var(--ink2)}
h2{font-size:24px;margin:34px 0 4px;border-bottom:1px solid var(--line);padding-bottom:4px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);padding:12px 14px;margin-top:12px;overflow:hidden}
.card h3{font-size:17px;margin:0 0 2px}.card p{margin:0 0 6px;font-size:14.5px;color:var(--ink2)}
.toggle{display:inline-flex;border:1px solid var(--line);margin:6px 0}
.toggle button{font:inherit;font-size:14.5px;background:var(--card);border:0;padding:5px 12px;cursor:pointer;color:var(--ink2)}
.toggle button.on{background:var(--ink);color:#fff}
.controls{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:6px 0 10px}@media(max-width:760px){.controls{grid-template-columns:1fr 1fr}}
label{display:block;font-size:13.5px;color:var(--ink3);margin-bottom:3px}
select,input[type=range]{width:100%;font:inherit;font-size:15px}select{padding:5px;border:1px solid var(--line);background:#fff}
.out{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line)}@media(max-width:760px){.out{grid-template-columns:1fr 1fr}}
.out div{background:var(--bg);padding:10px 12px}.out .v{font-size:22px;font-weight:bold;font-variant-numeric:tabular-nums}.out .l{font-size:13.5px;color:var(--ink2)}
.pos{color:var(--win)}.neg{color:var(--loss)}
table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:6px 9px;border-bottom:1px solid var(--line2);text-align:right;font-variant-numeric:tabular-nums}
th:first-child,td:first-child{text-align:left}th{font-size:13px;color:var(--ink3);font-weight:normal;border-bottom:1px solid var(--ink)}
.tbl{overflow-x:auto}
footer{margin-top:40px;font-size:13.5px;color:var(--ink3)}
</style>
</head>
<body><div class="wrap">
<h1>Sportsbook Customer Economics</h1>
<p class="sub">Robert Carrington · Interactive companion to the report</p>
<p class="note">Simulated player data, with seasonality, hold, and hold volatility calibrated to DraftKings' monthly New York filings. None of this is DraftKings internal data. Full method in the <a href="https://github.com/robertcarrington22/sportsbook-customer-economics/blob/main/REPORT.pdf">report</a> and <a href="https://github.com/robertcarrington22/sportsbook-customer-economics">code</a>.</p>
<div class="kpis" id="kpis"></div>

<h2>Channel payback</h2>
<div class="card">
  <div class="toggle"><button id="t12" class="on">First 12 months, actual</button><button id="t36">Projected to 36 months</button></div>
  <p id="payback-note"></p>
  <div id="c-payback"></div>
</div>

<h2>Bid-cap calculator</h2>
<div class="card">
  <p>Pick a channel, welcome offer, and state, then move the CAC. Contribution applies that state's tax rate to the channel's monthly net gaming revenue.</p>
  <div class="controls">
    <div><label for="s-ch">Channel</label><select id="s-ch"></select></div>
    <div><label for="s-of">Welcome offer</label><select id="s-of"></select></div>
    <div><label for="s-st">State</label><select id="s-st"></select></div>
    <div><label for="s-cac">CAC per FTD: <b id="s-cac-v"></b></label><input type="range" id="s-cac" min="50" max="900" step="10"></div>
  </div>
  <div class="out" id="s-out"></div>
  <div id="c-sim"></div>
</div>

<h2>Retention</h2>
<div class="grid">
  <div class="card"><h3>By signup cohort</h3><p>Share still betting, by month since first deposit.</p><div id="c-heat"></div></div>
  <div class="card"><h3>By channel</h3><p>Share still betting, first 12 months.</p><div id="c-retch"></div></div>
</div>

<h2>Gameplay</h2>
<div class="grid">
  <div class="card"><h3>Handle by bet type</h3><p>Monthly handle (bars) and the book's hold (line).</p><div id="c-bt"></div></div>
  <div class="card"><h3>Handle by sport</h3><p>Share of monthly handle, following the sports calendar.</p><div id="c-sp"></div></div>
</div>
<div class="card"><h3>What first-two-week gameplay says</h3><p>Month-3 retention, 180-day contribution, and, for fall signups, the share still betting in the following offseason (March to August).</p><div class="tbl"><table id="t-gseg"></table></div></div>

<h2>Early value</h2>
<div class="card"><h3>Handle in the first 14 days against the next 180</h3><div class="tbl"><table id="t-seg"></table></div></div>

<footer>Built from the same outputs as the report by build_dashboard.py. Model performance on the out-of-time test set: retention AUC <span id="f-auc"></span>, value rank correlation <span id="f-sp"></span> (theoretical contribution).</footer>
</div>
<script>
const D = __DATA__;
const FONT = "'Times New Roman', Times, serif";
const COL = {referral:'#1E7A4F', tv_brand:'#1D4E89', search:'#7A5AA6', affiliate:'#C27C0E', paid_social:'#B23A2E'};
const BTC = {straight:'#1D4E89', live:'#7FA3C9', parlay:'#C27C0E', sgp:'#B23A2E'};
const NM = {referral:'Referral', tv_brand:'TV and brand', search:'Search', affiliate:'Affiliate', paid_social:'Paid social',
  bet5_get200:'Bet $5, get $200', no_sweat_1000:'No-sweat first bet', deposit_match_250:'Deposit match', all:'All offers',
  straight:'Straight', live:'Live', parlay:'Parlay', sgp:'Same-game parlay', nfl:'NFL', college_football:'College football',
  nba:'NBA', college_basketball:'College basketball', mlb:'MLB', nhl:'NHL', soccer:'Soccer', other:'Other'};
const nm = k => NM[k] || k;
const usd = x => (x < 0 ? '-$' : '$') + Math.abs(Math.round(x)).toLocaleString();
const pct = (x, d = 0) => (x * 100).toFixed(d) + '%';
const cfg = {displayModeBar:false, responsive:true};
function L(o = {}) {
  const ax = {gridcolor:'#ECEEEA', linecolor:'#D9DCD6', zerolinecolor:'#B9BFB6', tickfont:{size:13}, title:{font:{size:14}}, automargin:true};
  return Object.assign({font:{family:FONT, size:14, color:'#4A5560'}, paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    margin:{l:10, r:14, t:30, b:10}, legend:{orientation:'h', y:1.08, x:0, font:{size:13}}, hoverlabel:{font:{family:FONT}}},
    o, {xaxis:Object.assign({}, ax, o.xaxis || {}), yaxis:Object.assign({}, ax, o.yaxis || {})});
}
const k = D.kpi;
document.getElementById('kpis').innerHTML = [
  [k.ftds.toLocaleString(), 'simulated FTDs'], [usd(k.cac), 'blended CAC'], [usd(k.ltv12), '12-month contribution per FTD'],
  [k.ratio.toFixed(2) + 'x', '12-month LTV / CAC'], [pct(k.neg180), 'net negative at day 180'], [k.spearman.toFixed(2), 'value model rank correlation'],
].map(([v, l]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');
document.getElementById('f-auc').textContent = k.auc.toFixed(3);
document.getElementById('f-sp').textContent = k.spearman.toFixed(3);

// Payback: 12 months actual, or 36 months projected with bands
const order = D.channels.map(c => c.channel);
function drawPayback(h36) {
  const tr = [];
  order.forEach(c => {
    const col = COL[c];
    if (h36) {
      const p = D.proj[c], x = [...Array(36).keys()].map(i => i + 1);
      tr.push({x, y:p.hi, mode:'lines', line:{width:0}, showlegend:false, hoverinfo:'skip'});
      tr.push({x, y:p.lo, mode:'lines', line:{width:0}, fill:'tonexty', fillcolor:col + '22', showlegend:false, hoverinfo:'skip'});
      tr.push({x, y:p.cum, mode:'lines', name:`${nm(c)} (${p.ratio36.toFixed(1)}x at 36 mo)`, line:{color:col, width:2.5},
        hovertemplate:`${nm(c)} month %{x}: %{y:$,.0f}<extra></extra>`});
    } else {
      const y = D.curves12[c], x = [...Array(12).keys()].map(i => i + 1);
      tr.push({x, y, mode:'lines', name:nm(c), line:{color:col, width:2.5}, hovertemplate:`${nm(c)} month %{x}: %{y:$,.0f}<extra></extra>`});
    }
    tr.push({x:[1, h36 ? 36 : 12], y:[D.cac[c], D.cac[c]], mode:'lines', line:{color:col, width:1, dash:'dot'}, showlegend:false,
      hovertemplate:`${nm(c)} CAC %{y:$,.0f}<extra></extra>`});
  });
  document.getElementById('payback-note').textContent = h36
    ? 'Projected theoretical contribution per FTD with 95% bootstrap bands, from a retention-decay model backtested on held-out months. Dotted lines are CAC.'
    : 'Cumulative contribution per FTD after promos, tax, and variable cost, for players with 12 full months. Dotted lines are CAC.';
  Plotly.react('c-payback', tr, L({height:430, xaxis:{title:{text:'Month since first deposit'}, dtick:h36 ? 6 : 1},
    yaxis:{title:{text:'Cumulative contribution per FTD'}, tickformat:'$,.0f'}}), cfg);
}
document.getElementById('t12').onclick = () => { t12.classList.add('on'); t36.classList.remove('on'); drawPayback(false); };
document.getElementById('t36').onclick = () => { t36.classList.add('on'); t12.classList.remove('on'); drawPayback(true); };
drawPayback(false);

// Calculator
const sc = document.getElementById('s-ch'), so = document.getElementById('s-of'), ss = document.getElementById('s-st'), sr = document.getElementById('s-cac');
order.forEach(c => sc.add(new Option(nm(c), c)));
['all', 'no_sweat_1000', 'deposit_match_250', 'bet5_get200'].forEach(o => so.add(new Option(nm(o), o)));
Object.entries(D.tax).sort((a, b) => b[1] - a[1]).forEach(([s, t]) => ss.add(new Option(`${s}, ${(t * 100).toFixed(t * 100 % 1 ? 1 : 0)}% tax`, s)));
sc.value = order[order.length - 1]; so.value = 'all'; ss.value = 'NY';
const setCac = () => { sr.value = D.sim[sc.value][so.value].cac; };
function runSim() {
  const d = D.sim[sc.value][so.value], tax = D.tax[ss.value], cac = +sr.value;
  document.getElementById('s-cac-v').textContent = usd(cac);
  let cum = 0, pay = -1; const ys = [];
  d.ngr.forEach((n, i) => { cum += n * (1 - tax) - d.var[i]; ys.push(cum); if (pay < 0 && cum >= cac) pay = i; });
  const ltv = ys[11];
  document.getElementById('s-out').innerHTML = [
    [usd(ltv), '12-month contribution per FTD', ''], [(ltv / cac).toFixed(2) + 'x', '12-month return on CAC', ltv >= cac ? 'pos' : 'neg'],
    [pay < 0 ? 'not in 12 mo' : 'month ' + (pay + 1), 'payback', pay < 0 ? 'neg' : ''], [usd(ltv), 'break-even CAC, 12 months', ''],
  ].map(([v, l, c]) => `<div><div class="v ${c}">${v}</div><div class="l">${l}</div></div>`).join('');
  const x = [...Array(12).keys()].map(i => i + 1);
  Plotly.react('c-sim', [
    {x, y:ys, mode:'lines+markers', name:'Cumulative contribution', line:{color:'#1E7A4F', width:2.5}, hovertemplate:'month %{x}: %{y:$,.0f}<extra></extra>'},
    {x:[1, 12], y:[cac, cac], mode:'lines', name:'CAC', line:{color:'#B23A2E', dash:'dot'}, hovertemplate:'CAC %{y:$,.0f}<extra></extra>'},
  ], L({height:320, xaxis:{title:{text:'Month since first deposit'}, dtick:1}, yaxis:{tickformat:'$,.0f', rangemode:'tozero'}}), cfg);
}
sc.onchange = so.onchange = () => { setCac(); runSim(); }; ss.onchange = runSim; sr.oninput = runSim; setCac(); runSim();

// Retention
Plotly.newPlot('c-heat', [{z:D.heat.z, x:D.heat.months, y:D.heat.cohorts, type:'heatmap', zmin:0, zmax:1,
  colorscale:[[0, '#F4F6F2'], [0.5, '#7FB89A'], [1, '#0E5A36']], xgap:1, ygap:1, colorbar:{tickformat:'.0%', thickness:10},
  hovertemplate:'%{y} · month %{x}: %{z:.0%}<extra></extra>'}],
  L({height:420, xaxis:{title:{text:'Month since first deposit'}}, yaxis:{autorange:'reversed'}}), cfg);
Plotly.newPlot('c-retch', order.map(c => ({x:[...Array(12).keys()].map(i => i + 1), y:D.ret_ch[c], mode:'lines', name:nm(c),
  line:{color:COL[c], width:2.2}, hovertemplate:`${nm(c)} month %{x}: %{y:.0%}<extra></extra>`})),
  L({height:420, xaxis:{title:{text:'Month since first deposit'}, dtick:1}, yaxis:{tickformat:'.0%', range:[0, 1.02]}}), cfg);

// Gameplay
const btOrder = ['straight', 'live', 'parlay', 'sgp'];
const btTr = btOrder.map(b => ({x:D.bt[b].months, y:D.bt[b].handle.map(v => v / 1e6), type:'bar', name:nm(b), marker:{color:BTC[b]},
  hovertemplate:`${nm(b)} %{x}: $%{y:.1f}M<extra></extra>`}));
btTr.push({x:D.hold.months, y:D.hold.hold, yaxis:'y2', mode:'lines+markers', name:'Hold', line:{color:'#16202A', width:2}, marker:{size:5},
  hovertemplate:'hold %{x}: %{y:.1%}<extra></extra>'});
Plotly.newPlot('c-bt', btTr, L({height:400, barmode:'stack', yaxis:{title:{text:'Handle, $M'}},
  yaxis2:{overlaying:'y', side:'right', tickformat:'.0%', range:[0, 0.16], showgrid:false, tickfont:{size:13}}}), cfg);
const months = D.sports.nfl ? D.sports.nfl.months : Object.values(D.sports)[0].months;
const tot = months.map((m, i) => Object.values(D.sports).reduce((s, v) => s + (v.handle[v.months.indexOf(m)] || 0), 0));
Plotly.newPlot('c-sp', Object.keys(D.sports).map(s => ({x:months, y:months.map((m, i) => (D.sports[s].handle[D.sports[s].months.indexOf(m)] || 0) / tot[i]),
  stackgroup:'one', name:nm(s), mode:'lines', line:{width:0.5}, hovertemplate:`${nm(s)} %{x}: %{y:.0%}<extra></extra>`})),
  L({height:400, yaxis:{tickformat:'.0%', range:[0, 1]}}), cfg);
document.getElementById('t-gseg').innerHTML = '<tr><th>Cut</th><th>Group</th><th>FTDs</th><th>Active month 3</th><th>180-day contribution</th><th>Fall signups still betting in offseason</th></tr>' +
  D.gseg.map(r => `<tr><td>${r.cut}</td><td>${r.band.slice(3)}</td><td>${r.ftds.toLocaleString()}</td><td>${pct(r.active_m3_rate)}</td><td>${usd(r.mean_contribution_180)}</td><td>${pct(r.offseason_active_rate_fall_signups)}</td></tr>`).join('');
document.getElementById('t-seg').innerHTML = '<tr><th>Handle, days 0 to 13</th><th>Share of FTDs</th><th>Active month 3</th><th>Mean 180-day contribution</th><th>Share profitable</th><th>Share of total 180-day</th></tr>' +
  D.seg.map(r => `<tr><td>${r.handle_band.slice(3)}</td><td>${pct(r.share_of_ftds, 1)}</td><td>${pct(r.active_m3_rate)}</td><td>${usd(r.mean_contribution_180)}</td><td>${pct(r.share_profitable_180)}</td><td>${pct(r.share_of_total_180)}</td></tr>`).join('');
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
