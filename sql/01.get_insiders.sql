select
*
from
	stocks_data.insider_trading as t1
where
	t1.Symbol =:symbol