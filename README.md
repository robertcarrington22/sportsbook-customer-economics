# What is a new depositor worth?

A customer economics model for a simulated US online sportsbook: acquisition cost, twelve-month contribution, payback by channel, the cost of welcome offers, the effect of state tax, and a model that estimates a player's value from their first two weeks.

**The data is simulated.** No real operator data is used. Every assumption lives in [`assumptions.toml`](assumptions.toml). The SQL and models are written to run unchanged against real first-time-depositor and bet-level tables. The findings demonstrate the method, not any real operator's economics.

Open `report/index.html` for the charts and a CAC payback simulator.

---

## Memo

**To:** Customer Economics
**Re:** Where the next acquisition dollar should go

40,000 simulated first-time depositors (FTDs) from September 2024 to August 2025, followed through June 2026. Blended cost per FTD is **$323**. Blended twelve-month contribution per FTD is **$423**, after promos, gaming tax, and variable cost. That is **1.31×** in year one.

### 1. Channel mix matters more than channel cost

| Channel | CAC | 12-mo LTV | Median LTV | LTV / CAC | Payback |
|---|---:|---:|---:|---:|---|
| Referral | $160 | $541 | −$28 | 3.37× | month 4 |
| TV / brand | $220 | $372 | −$37 | 1.69× | month 8 |
| Search | $390 | $518 | −$35 | 1.33× | month 9 |
| Affiliate | $452 | $486 | −$55 | 1.08× | month 11 |
| Paid social | $310 | $285 | −$52 | 0.92× | not in 12 months |

Paid social is cheaper per FTD than search or affiliate and is still the only channel that loses money in year one. Its depositors are the least likely to still be betting in month three: 50%, against 62% for referral. A cheap depositor who never bets again is not cheap.

### 2. The richest welcome offer is the worst deal

| Offer | Cost per FTD | Active in month 3 | 12-mo LTV | LTV / CAC |
|---|---:|---:|---:|---:|
| No-sweat first bet | $90 | 58% | $514 | 1.59× |
| Deposit match | $65 | 55% | $458 | 1.42× |
| Bet $5, get $200 | $144 | 53% | $353 | 1.09× |

"Bet $5, get $200" costs the most and brings the weakest players, because a no-strings offer attracts bonus hunters who leave once it is spent.

### 3. State tax changes what a player is worth by 2×

At a 51% rate, a New York FTD contributes **$242** in year one and returns **0.75×** a national-average CAC. The same behavior in Michigan at 8.4% contributes **$520** and returns **1.62×**. A single national bid cap overpays in high-tax states. Caps should be set by state.

### 4. Two weeks is enough to find most of the value

**67%** of FTDs are net negative at 180 days once promos are counted, and the top tenth of players carries essentially all of the profit. So the question that matters is who those players are, early.

Trained on September 2024 to April 2025 signups, tested on May to August 2025:

| Question | Metric | Model | Baseline |
|---|---|---:|---:|
| Still betting in month 3? | AUC | 0.741 | 0.739 logistic |
| Flag the at-risk tenth | Lapse rate in bottom decile | 75% | 45% overall |
| Rank everyone by 180-day value | Spearman | 0.48 | 0.35 handle only |
| Find the top tenth | Lift over average | 9.8× | 9.8× handle only |
| Predict dollar value | Mean absolute error | $382 | $567 predict the mean |

The honest read: **to find the top tenth, sorting by two-week handle is as good as the model.** The model earns its keep on everyone else, ranking the middle and bottom of the distribution well enough to set per-player bid caps and retention spend. Because the data is simulated, each player's hidden type is known, so the model can be checked against it. **62%** of its top decile are true high-value players, against a 10% base rate.

The model is biased low on the top decile. It predicts about $1,400 where the actual average is about $2,300, because squared-error loss shrinks a heavy right tail. Anything that sets a bid cap from predicted value should model log value or recalibrate the top first.

### Recommendations

1. Shift budget from paid social toward referral and brand until paid social's month-three retention improves. Test a smaller, stickier offer on paid social before cutting it.
2. Retire "Bet $5, get $200" as the default. Test the no-sweat offer as default in two states against a holdout.
3. Set acquisition bid caps by state tax rate, not nationally.
4. Score every FTD at day 14. Use the model to set retention spend across the whole base, and simple handle to flag likely high-value players for VIP outreach.

---

## How it's built

```
assumptions.toml     every number the simulation uses
simulate.py          40,000 FTDs and every day they bet  ->  data/
sql/01..08           DuckDB: player-month fact table, retention, LTV curves,
                     channel / offer / state economics, modeling features
analyze.py           runs sql/ in order, writes outputs/
model.py             month-3 retention classifier and 180-day value regressor
build_report.py      report/index.html with charts and the payback simulator
run.py               all of the above, in order
```

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python run.py
```

The full pipeline runs in about fifteen seconds and is deterministic.

**Contribution** is gross gaming revenue, minus welcome and ongoing promos, minus gaming tax on the remainder, minus variable cost at 0.6% of handle. It excludes fixed costs and overhead.

**Censoring.** Every twelve-month figure and curve uses only players with twelve full months of history, so younger cohorts never drag an average down.

**Out-of-time split.** Models train on earlier cohorts and test on later ones, which is how they would actually be used.

**Simplifications.** Tax is a flat rate on net revenue, though several states tier it or tax per wager. The rates are illustrative. There is no casino or fantasy cross-sell. Players never return after churning.

**Next.** Model log value to fix the top-decile bias. Add reactivation and cross-sell. Point the SQL at real tables.
