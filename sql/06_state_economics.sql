-- The same player is worth very different amounts depending on the state's tax rate.

WITH per_player AS (
    SELECT
        player_id,
        state,
        any_value(tax_rate) AS tax_rate,
        any_value(cac)      AS cac,
        sum(ggr)            AS ggr_12m,
        sum(ngr)            AS ngr_12m,
        sum(gaming_tax)     AS tax_12m,
        sum(variable_cost)  AS variable_cost_12m,
        sum(contribution)   AS contribution_12m
    FROM player_month
    WHERE last_full_month >= 11
      AND life_month <= 11
    GROUP BY player_id, state
)
SELECT
    state,
    any_value(tax_rate)                            AS tax_rate,
    count(*)                                       AS ftds,
    round(avg(ggr_12m), 0)                         AS ggr_per_ftd,
    round(avg(ngr_12m), 0)                         AS ngr_per_ftd,
    round(avg(tax_12m), 0)                         AS tax_per_ftd,
    round(avg(contribution_12m), 0)                AS contribution_per_ftd,
    round(avg(contribution_12m) / avg(cac), 2)     AS ltv_to_cac,
    -- NGR per FTD needed in year one to cover CAC and variable cost after tax
    round((avg(cac) + avg(variable_cost_12m)) / (1 - any_value(tax_rate)), 0)
                                                   AS breakeven_ngr_per_ftd
FROM per_player
GROUP BY state
ORDER BY tax_rate DESC;
