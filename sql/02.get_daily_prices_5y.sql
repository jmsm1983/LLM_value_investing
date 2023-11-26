
select
	date, symbol, adjClose 
from
	stocks_data.historical_stock_prices
where symbol in :symbol
and  date >= DATE_SUB(CURDATE(), INTERVAL 5 YEAR)
order by symbol, date desc
