-- Headline unit economics by acquisition channel over the first 12 months.
-- Payback month is the first life month where cumulative contribution per FTD
-- covers the channel's average CAC. NULL means it did not pay back inside a year.
-- top_decile_share_of_value can exceed 1.0: after welcome promos most players
-- are net negative, so the top 10% earn more than the whole channel's total.

WITH base AS (
    SELECT *
    FROM player_month
    WHERE last_full_month >= 11
      AND life_month <= 11
),
per_player AS (
    SELECT
        player_id,
        channel,
        any_value(cac)                                         AS cac,
        sum(contribution)                                      AS contribution_12m,
        max(CASE WHEN life_month = 2 THEN active_days > 0 END) AS active_m3
    FROM base
    GROUP BY player_id, channel
),
curve AS (
    SELECT
        channel,
        life_month,
        sum(sum(contribution)) OVER (PARTITION BY channel ORDER BY life_month)
            / count(DISTINCT player_id) AS cum_per_ftd
    FROM base
    GROUP BY channel, life_month
),
summary AS (
    SELECT
        channel,
        count(*)                                     AS ftds,
        avg(cac)                                     AS cac,
        avg(contribution_12m)                        AS ltv_12m,
        avg(CAST(active_m3 AS INTEGER))              AS active_m3_rate,
        quantile_cont(contribution_12m, 0.5)         AS median_ltv_12m,
        sum(contribution_12m) FILTER (WHERE contribution_12m >= quantile_12m) / sum(contribution_12m)
                                                      AS top_decile_share
    FROM per_player
    JOIN (
        SELECT channel AS ch, quantile_cont(contribution_12m, 0.9) AS quantile_12m
        FROM per_player GROUP BY channel
    ) q ON q.ch = per_player.channel
    GROUP BY channel
)
SELECT
    s.channel,
    s.ftds,
    round(s.cac, 0)                  AS cac,
    round(s.ltv_12m, 0)              AS ltv_12m,
    round(s.median_ltv_12m, 0)       AS median_ltv_12m,
    round(s.ltv_12m / s.cac, 2)      AS ltv_to_cac,
    round(s.active_m3_rate, 3)       AS active_m3_rate,
    round(s.top_decile_share, 3)     AS top_decile_share_of_value,
    min(c.life_month) FILTER (WHERE c.cum_per_ftd >= s.cac) AS payback_month
FROM summary AS s
JOIN curve AS c USING (channel)
GROUP BY ALL
ORDER BY ltv_to_cac DESC;
