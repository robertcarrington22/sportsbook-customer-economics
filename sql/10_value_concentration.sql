-- How concentrated year-one value is. Players are ranked by 12-month
-- contribution; each row is a percentile of players (1 = most valuable) with
-- the running share of total contribution.
-- The running share goes above 1.0 and comes back down, because most players
-- are net negative.

WITH per_player AS (
    SELECT player_id, sum(contribution) AS c12
    FROM player_month
    WHERE last_full_month >= 11
      AND life_month <= 11
    GROUP BY player_id
),
ranked AS (
    SELECT
        c12,
        ntile(100) OVER (ORDER BY c12 DESC) AS pctile
    FROM per_player
),
by_pct AS (
    SELECT pctile, sum(c12) AS c, avg(c12) AS mean_c12, count(*) AS n
    FROM ranked
    GROUP BY pctile
)
SELECT
    pctile,
    n,
    round(mean_c12, 2)                                                     AS mean_contribution,
    round(sum(c) OVER (ORDER BY pctile) / sum(c) OVER (), 4)               AS cum_share_of_total
FROM by_pct
ORDER BY pctile;
