-- What first-two-week gameplay says about retention and value.
-- Three cuts of the same FTDs: parlay share of handle, number of sports bet,
-- and football concentration. The last cut uses football-season signups
-- (Sep to Dec 2024) and asks who is still betting in the following offseason
-- (March to August 2025).

WITH f AS (SELECT * FROM features),
offseason AS (
    SELECT DISTINCT a.player_id
    FROM activity AS a
    WHERE a.activity_date BETWEEN DATE '2025-03-01' AND DATE '2025-08-31'
),
cuts AS (
    SELECT 'Parlay share of handle' AS cut,
           CASE WHEN parlay_share_14 < 0.25 THEN '1  under 25%'
                WHEN parlay_share_14 < 0.50 THEN '2  25% to 50%'
                WHEN parlay_share_14 < 0.75 THEN '3  50% to 75%'
                ELSE '4  75% and up' END AS band,
           f.*
    FROM f
    UNION ALL
    SELECT 'Sports bet in first 14 days',
           CASE WHEN n_sports_14 = 1 THEN '1  one sport'
                WHEN n_sports_14 = 2 THEN '2  two sports'
                ELSE '3  three or more' END,
           f.*
    FROM f
    UNION ALL
    SELECT 'Football share of handle',
           CASE WHEN football_share_14 >= 0.8 THEN '1  football-heavy (80%+)'
                WHEN football_share_14 >= 0.4 THEN '2  mixed (40% to 80%)'
                ELSE '3  mostly other sports' END,
           f.*
    FROM f
)
SELECT
    c.cut,
    c.band,
    count(*)                                                        AS ftds,
    round(avg(c.handle_14), 0)                                      AS avg_handle_14,
    round(avg(c.active_m3), 3)                                      AS active_m3_rate,
    round(avg(c.contribution_180), 0)                               AS mean_contribution_180,
    round(avg(CAST(o.player_id IS NOT NULL AS INTEGER))
          FILTER (WHERE c.cohort_month BETWEEN DATE '2024-09-01' AND DATE '2024-12-01'), 3)
                                                                    AS offseason_active_rate_fall_signups
FROM cuts AS c
LEFT JOIN offseason AS o USING (player_id)
GROUP BY c.cut, c.band
ORDER BY c.cut, c.band;
