-- Pre-tax monthly economics per FTD by channel and offer, months 0 to 11.
-- The payback simulator in the report applies a state tax rate and a CAC on top.

WITH base AS (
    SELECT *
    FROM player_month
    WHERE last_full_month >= 11
      AND life_month <= 11
)
SELECT
    channel,
    offer,
    life_month,
    count(*)                                   AS ftds,
    round(avg(cac), 2)                         AS cac_per_ftd,
    round(sum(ngr) / count(*), 2)              AS ngr_per_ftd,
    round(sum(variable_cost) / count(*), 2)    AS variable_cost_per_ftd
FROM base
GROUP BY ALL
ORDER BY channel, offer, life_month;
