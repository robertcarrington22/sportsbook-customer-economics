-- Channel x offer: does the offer that works in one channel work in another?

WITH per_player AS (
    SELECT
        player_id,
        channel,
        offer,
        any_value(cac)    AS cac,
        sum(contribution) AS c12
    FROM player_month
    WHERE last_full_month >= 11
      AND life_month <= 11
    GROUP BY player_id, channel, offer
)
SELECT
    channel,
    offer,
    count(*)                          AS ftds,
    round(avg(cac), 0)                AS cac,
    round(avg(c12), 0)                AS ltv_12m,
    round(avg(c12) / avg(cac), 2)     AS ltv_to_cac
FROM per_player
GROUP BY channel, offer
ORDER BY channel, offer;
