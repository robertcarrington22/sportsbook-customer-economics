-- Signup-month cohorts: volume, cost, and 6-month value.
-- Every cohort has at least 6 full months observed, so all 12 are comparable.

WITH per_player AS (
    SELECT
        player_id,
        cohort_month,
        any_value(cac)                                          AS cac,
        sum(contribution) FILTER (WHERE life_month <= 5)        AS c6,
        max(CASE WHEN life_month = 2 THEN active_days > 0 END)  AS active_m3
    FROM player_month
    GROUP BY player_id, cohort_month
)
SELECT
    cohort_month,
    count(*)                                   AS ftds,
    round(avg(cac), 0)                         AS cac,
    round(avg(c6), 0)                          AS ltv_6m,
    round(avg(CAST(active_m3 AS INTEGER)), 3)  AS active_m3_rate
FROM per_player
GROUP BY cohort_month
ORDER BY cohort_month;
