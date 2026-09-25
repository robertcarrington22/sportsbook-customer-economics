-- Welcome offers: what each costs up front and what the players it brings are worth.

WITH per_player AS (
    SELECT
        player_id,
        offer,
        any_value(cac)                                          AS cac,
        sum(welcome_promo)                                      AS welcome_cost,
        sum(contribution)                                       AS contribution_12m,
        max(CASE WHEN life_month = 2 THEN active_days > 0 END)  AS active_m3
    FROM player_month
    WHERE last_full_month >= 11
      AND life_month <= 11
    GROUP BY player_id, offer
)
SELECT
    offer,
    count(*)                                              AS ftds,
    round(avg(welcome_cost), 0)                           AS welcome_cost_per_ftd,
    round(avg(CAST(active_m3 AS INTEGER)), 3)             AS active_m3_rate,
    round(avg(contribution_12m), 0)                       AS ltv_12m,
    round(avg(contribution_12m) / avg(cac), 2)            AS ltv_to_cac
FROM per_player
GROUP BY offer
ORDER BY ltv_to_cac DESC;
