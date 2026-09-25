-- Gameplay trend: handle by sport and calendar month.

SELECT
    CAST(date_trunc('month', activity_date) AS DATE) AS month,
    CAST(sport AS VARCHAR)                           AS sport,
    round(sum(stake), 0)                             AS handle,
    round(sum(ggr), 0)                               AS ggr
FROM bets
GROUP BY ALL
ORDER BY month, sport;
