select
*,
actualEarningResult/estimatedEarning -1 as pct_surprise
from
	stocks_data.surprise_earnings_annual as t1
where
	t1.Symbol =:symbol
order by
	t1.date desc
limit 8