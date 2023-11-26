select
date,adjClose, symbol
from
	stocks_data.sp500_index
WHERE date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR)	
order by
	date desc
