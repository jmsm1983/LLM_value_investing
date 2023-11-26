
select
	t2.*
from
	(
	select
		*,
		row_number() over(partition by t1.Symbol
	order by
		t1.date desc) as rn
	from
		stocks_data.ratios_annual as t1
	where
		t1.Symbol in :symbol
	order by
		t1.Symbol,
		rn) as t2
where
	rn = 1
	
