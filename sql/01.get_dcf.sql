select
*
from
	stocks_data.dcf as t1
where
	t1.Symbol =:symbol
order by
	t1.date desc
