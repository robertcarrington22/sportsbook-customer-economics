-- One row per first-time depositor per 30-day "life month" since first deposit.
-- Zero-activity months are included, so averages are per FTD, not per active player.
-- Each player is truncated at their last fully observed month, so no curve is
-- biased by cohorts that have not had time to mature.

CREATE OR REPLACE TABLE player_month AS
WITH p AS (
    SELECT
        pl.*,
        s.tax_rate,
        CAST(floor((date_diff('day', pl.signup_date, (SELECT observation_end FROM params)) + 1) / 30) AS INTEGER) - 1
            AS last_full_month
    FROM players AS pl
    JOIN states AS s USING (state)
),
months AS (
    SELECT p.player_id, m.life_month
    FROM p
    CROSS JOIN range(0, 24) AS m(life_month)
    WHERE m.life_month <= p.last_full_month
),
act AS (
    SELECT
        a.player_id,
        CAST(floor(date_diff('day', pl.signup_date, a.activity_date) / 30) AS INTEGER) AS life_month,
        count(*)        AS active_days,
        sum(a.bets)     AS bets,
        sum(a.handle)   AS handle,
        sum(a.ggr)      AS ggr,
        sum(a.theo_ggr) AS theo_ggr,
        sum(a.reinvest) AS reinvest
    FROM activity AS a
    JOIN players AS pl USING (player_id)
    GROUP BY ALL
),
base AS (
    SELECT
        m.player_id,
        m.life_month,
        p.last_full_month,
        p.signup_date,
        p.cohort_month,
        p.channel,
        p.offer,
        p.state,
        p.cac,
        p.tax_rate,
        coalesce(a.active_days, 0) AS active_days,
        coalesce(a.bets, 0)        AS bets,
        coalesce(a.handle, 0)      AS handle,
        coalesce(a.ggr, 0)         AS ggr,
        coalesce(a.theo_ggr, 0)    AS theo_ggr,
        coalesce(a.reinvest, 0)    AS reinvest,
        CASE WHEN m.life_month = 0 THEN p.welcome_promo_cost ELSE 0 END AS welcome_promo
    FROM months AS m
    JOIN p USING (player_id)
    LEFT JOIN act AS a USING (player_id, life_month)
),
econ AS (
    SELECT
        *,
        ggr - reinvest - welcome_promo AS ngr,
        theo_ggr - reinvest - welcome_promo AS theo_ngr,
        handle * (SELECT variable_cost_pct_handle FROM params) AS variable_cost
    FROM base
)
SELECT
    *,
    ngr * tax_rate                         AS gaming_tax,
    ngr * (1 - tax_rate) - variable_cost   AS contribution,
    -- Theoretical contribution: the same calculation on expected rather than realized GGR,
    -- which removes short-run luck in bet outcomes. Sportsbooks value players on this basis ("theo").
    theo_ngr * (1 - tax_rate) - variable_cost AS theo_contribution
FROM econ;

SELECT
    count(DISTINCT player_id)            AS ftds,
    count(*)                             AS player_months,
    round(sum(ggr))                      AS ggr,
    round(sum(welcome_promo + reinvest)) AS promo,
    round(sum(gaming_tax))               AS gaming_tax,
    round(sum(contribution))             AS contribution
FROM player_month;
