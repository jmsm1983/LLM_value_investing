select
*
from
	stocks_data.ratios_annual as t1
where
	t1.Symbol =:symbol
order by
	t1.date desc
limit 10