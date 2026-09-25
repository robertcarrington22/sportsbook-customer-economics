-- Maximum CAC that still pays back inside 12 months, by channel and state.
-- This is 12-month contribution per FTD for each cell, i.e. a bid cap.

WITH per_player AS (
    SELECT
        player_id,
        channel,
        state,
        any_value(cac)    AS cac,
        sum(contribution) AS c12
    FROM player_month
    WHERE last_full_month >= 11
      AND life_month <= 11
    GROUP BY player_id, channel, state
)
SELECT
    channel,
    state,
    count(*)             AS ftds,
    round(avg(cac), 0)   AS current_cac,
    round(avg(c12), 0)   AS breakeven_cac
FROM per_player
GROUP BY channel, state
ORDER BY channel, state;
