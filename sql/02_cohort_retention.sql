-- Share of each signup cohort that placed at least one bet in each life month.
-- Only months every player in the row has fully lived through are counted.

SELECT
    cohort_month,
    life_month,
    count(*)                                      AS ftds_observed,
    round(avg(CAST(active_days > 0 AS INTEGER)), 4) AS active_rate
FROM player_month
GROUP BY ALL
ORDER BY cohort_month, life_month;
