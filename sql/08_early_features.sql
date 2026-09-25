-- Modeling table: what we know about a player after their first 14 days,
-- and the outcomes we want to predict.
--   active_m3         placed a bet in life month 3 (days 60-89)
--   contribution_180       contribution over the first 180 days
--   theo_contribution_180  the same on expected rather than realized GGR (removes bet-outcome luck)

CREATE OR REPLACE TABLE features AS
WITH early AS (
    SELECT
        a.player_id,
        date_diff('day', p.signup_date, a.activity_date) AS d,
        a.bets,
        a.handle,
        a.ggr
    FROM activity AS a
    JOIN players AS p USING (player_id)
    WHERE date_diff('day', p.signup_date, a.activity_date) <= 13
),
early_agg AS (
    SELECT
        player_id,
        count(*)                                 AS active_days_14,
        count(*) FILTER (WHERE d BETWEEN 7 AND 13) AS active_days_wk2,
        sum(bets)                                AS bets_14,
        sum(handle)                              AS handle_14,
        sum(ggr)                                 AS ggr_14,
        max(handle)                              AS max_day_handle_14,
        sum(handle) / sum(bets)                  AS avg_stake_14,
        13 - max(d)                              AS days_idle_at_d14
    FROM early
    GROUP BY player_id
),
early_bets AS (
    -- gameplay in the first 14 days, from the bet-level table
    SELECT
        b.player_id,
        sum(b.stake) FILTER (WHERE b.bet_type IN ('parlay', 'sgp')) / sum(b.stake)          AS parlay_share_14,
        sum(b.stake) FILTER (WHERE b.bet_type = 'live') / sum(b.stake)                      AS live_share_14,
        count(DISTINCT b.sport)                                                             AS n_sports_14,
        sum(b.stake) FILTER (WHERE b.sport IN ('nfl', 'college_football')) / sum(b.stake)   AS football_share_14
    FROM bets AS b
    JOIN players AS p USING (player_id)
    WHERE date_diff('day', p.signup_date, b.activity_date) <= 13
    GROUP BY b.player_id
),
calendar AS (
    -- How busy the calendar ahead is, from DraftKings' real NY seasonality index.
    -- Known on day 14 (it depends only on the signup date), so it is not leakage,
    -- and unlike signup month it generalizes to months the training data never saw.
    SELECT
        p.player_id,
        avg(s.handle_index) FILTER (WHERE r.d BETWEEN 14 AND 179) AS season_ahead_180,
        avg(s.handle_index) FILTER (WHERE r.d BETWEEN 60 AND 89)  AS season_m3
    FROM players AS p
    CROSS JOIN range(14, 180) AS r(d)
    JOIN seasonality AS s ON s.month_of_year = month(CAST(p.signup_date AS DATE) + CAST(r.d AS INTEGER))
    GROUP BY p.player_id
),
outcomes AS (
    SELECT
        player_id,
        max(CASE WHEN life_month = 2 THEN active_days > 0 END) AS active_m3,
        sum(contribution) FILTER (WHERE life_month <= 5)      AS contribution_180,
        sum(theo_contribution) FILTER (WHERE life_month <= 5) AS theo_contribution_180
    FROM player_month
    GROUP BY player_id
)
SELECT
    p.player_id,
    p.cohort_month,
    p.channel,
    p.offer,
    p.state,
    CAST(month(p.signup_date) AS VARCHAR)  AS signup_month,
    p.welcome_promo_cost,
    e.active_days_14,
    e.active_days_wk2,
    e.bets_14,
    e.handle_14,
    e.ggr_14,
    e.max_day_handle_14,
    e.avg_stake_14,
    e.days_idle_at_d14,
    coalesce(g.parlay_share_14, 0)         AS parlay_share_14,
    coalesce(g.live_share_14, 0)           AS live_share_14,
    g.n_sports_14,
    coalesce(g.football_share_14, 0)       AS football_share_14,
    c.season_ahead_180,
    c.season_m3,
    CAST(o.active_m3 AS INTEGER)           AS active_m3,
    o.contribution_180,
    o.theo_contribution_180
FROM players AS p
JOIN early_agg AS e USING (player_id)
JOIN early_bets AS g USING (player_id)
JOIN calendar AS c USING (player_id)
JOIN outcomes AS o USING (player_id)
ORDER BY p.player_id;

SELECT
    count(*)                              AS rows,
    round(avg(active_m3), 3)              AS active_m3_rate,
    round(avg(contribution_180), 0)       AS mean_contribution_180
FROM features;
