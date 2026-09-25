-- Share of each channel's FTDs still betting in each life month, first year.

SELECT
    channel,
    life_month,
    count(*)                                         AS ftds_observed,
    round(avg(CAST(active_days > 0 AS INTEGER)), 4)  AS active_rate
FROM player_month
WHERE last_full_month >= 11
  AND life_month <= 11
GROUP BY channel, life_month
ORDER BY channel, life_month;
