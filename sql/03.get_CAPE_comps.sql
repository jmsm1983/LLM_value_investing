
select
	Date(Date) as Date, SP500, CAPE
FROM macro_data.sp500_comp
where `Date` >='2000-01-01'

