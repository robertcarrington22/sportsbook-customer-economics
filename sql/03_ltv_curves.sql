-- Cumulative contribution per FTD by acquisition channel, months 0 to 11.
-- Fixed population: only FTDs with at least 12 fully observed months, so every
-- point on a curve averages over the same players.

WITH base AS (
    SELECT *
    FROM player_month
    WHERE last_full_month >= 11
      AND life_month <= 11
),
monthly AS (
    SELECT
        channel,
        life_month,
        count(*)          AS ftds,
        sum(contribution) AS contribution,
        sum(ngr)          AS ngr,
        avg(cac)          AS cac
    FROM base
    GROUP BY ALL
)
SELECT
    channel,
    life_month,
    ftds,
    round(avg(cac) OVER w_all, 2)                        AS cac_per_ftd,
    round(sum(contribution) OVER w_cum / ftds, 2)        AS cum_contribution_per_ftd,
    round(sum(ngr) OVER w_cum / ftds, 2)                 AS cum_ngr_per_ftd
FROM monthly
WINDOW
    w_cum AS (PARTITION BY channel ORDER BY life_month),
    w_all AS (PARTITION BY channel)
ORDER BY channel, life_month;
