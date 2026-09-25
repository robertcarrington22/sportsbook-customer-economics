-- What the first 14 days say about the next 6 months.
-- Segments by handle in days 0-13. Uses the features table from 08.

WITH seg AS (
    SELECT
        *,
        CASE
            WHEN handle_14 < 50    THEN '1  under $50'
            WHEN handle_14 < 250   THEN '2  $50 to $250'
            WHEN handle_14 < 1000  THEN '3  $250 to $1k'
            WHEN handle_14 < 5000  THEN '4  $1k to $5k'
            ELSE                        '5  $5k and up'
        END AS handle_band
    FROM features
)
SELECT
    handle_band,
    count(*)                                              AS ftds,
    round(count(*) / sum(count(*)) OVER (), 3)            AS share_of_ftds,
    round(avg(active_days_14), 1)                         AS avg_active_days_14,
    round(avg(active_m3), 3)                              AS active_m3_rate,
    round(avg(contribution_180), 0)                       AS mean_contribution_180,
    round(quantile_cont(contribution_180, 0.5), 0)        AS median_contribution_180,
    round(avg(CAST(contribution_180 > 0 AS INTEGER)), 3)  AS share_profitable_180,
    round(sum(contribution_180) / sum(sum(contribution_180)) OVER (), 3) AS share_of_total_180
FROM seg
GROUP BY handle_band
ORDER BY handle_band;
