-- Where year-one revenue goes, per FTD: GGR down to contribution.
-- Players with 12 full months observed.

WITH base AS (
    SELECT *
    FROM player_month
    WHERE last_full_month >= 11
      AND life_month <= 11
),
n AS (SELECT count(DISTINCT player_id) AS ftds FROM base)
SELECT step, round(amount / (SELECT ftds FROM n), 2) AS per_ftd, sort
FROM (
    SELECT 'Gross gaming revenue' AS step, sum(ggr)                 AS amount, 1 AS sort FROM base
    UNION ALL SELECT 'Welcome offer',       -sum(welcome_promo),               2 FROM base
    UNION ALL SELECT 'Ongoing promos',      -sum(reinvest),                    3 FROM base
    UNION ALL SELECT 'Gaming tax',          -sum(gaming_tax),                  4 FROM base
    UNION ALL SELECT 'Variable cost',       -sum(variable_cost),               5 FROM base
    UNION ALL SELECT 'Contribution',         sum(contribution),                6 FROM base
)
ORDER BY sort;
