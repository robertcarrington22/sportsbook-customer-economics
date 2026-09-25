-- Gameplay trend: handle, GGR, and hold by bet type and calendar month.

SELECT
    CAST(date_trunc('month', activity_date) AS DATE) AS month,
    CAST(bet_type AS VARCHAR)                        AS bet_type,
    count(*)                                         AS bets,
    round(sum(stake), 0)                             AS handle,
    round(sum(ggr), 0)                               AS ggr,
    round(sum(ggr) / sum(stake), 4)                  AS hold
FROM bets
GROUP BY ALL
ORDER BY month, bet_type;
