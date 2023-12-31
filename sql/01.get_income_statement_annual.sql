select
t1.`date` as `date`,
t1.symbol as symbol,
t1.reportedCurrency as reportedCurrency,
t1.cik as cik,
t1.fillingDate as fillingDate,
t1.acceptedDate as acceptedDate,
t1.calendarYear as calendarYear,
t1.period as period,
t1.revenue as revenue,
t1.costOfRevenue as costOfRevenue,
t1.grossProfit as grossProfit,
t1.grossProfitRatio as grossProfitRatio,
t1.ResearchAndDevelopmentExpenses as ResearchAndDevelopmentExpenses,
t1.GeneralAndAdministrativeExpenses as GeneralAndAdministrativeExpenses,
t1.SellingAndMarketingExpenses as SellingAndMarketingExpenses,
t1.otherExpenses as otherExpenses,
t1.operatingExpenses as operatingExpenses,
t1.costAndExpenses as costAndExpenses,
t1.interestExpense as interestExpense,
t1.depreciationAndAmortization as depreciationAndAmortization,
t1.EBITDA as EBITDA,
t1.EBITDARatio as EBITDARatio,
t1.operatingIncome as operatingIncome,
t1.operatingIncomeRatio as operatingIncomeRatio,
t1.totalOtherIncomeExpensesNet as totalOtherIncomeExpensesNet,
t1.incomeBeforeTax as incomeBeforeTax,
t1.incomeBeforeTaxRatio as incomeBeforeTaxRatio,
t1.incomeTaxExpense as incomeTaxExpense,
t1.netIncome as netIncome,
t1.netIncomeRatio as netIncomeRatio,
t1.EPS as EPS,
t1.EPSDiluted as EPSDiluted,
t1.weightedAverageShsOut as weightedAverageShsOut,
t1.weightedAverageShsOutDil as weightedAverageShsOutDil,
t1.link as link,
t1.finalLink as finalLink,
t1.interestIncome as interestIncome,
t2.growthRevenue as growthRevenue,
t2.growthCostOfRevenue as growthCostOfRevenue,
t2.growthGrossProfit as growthGrossProfit,
t2.growthGrossProfitRatio as growthGrossProfitRatio,
t2.growthResearchAndDevelopmentExpenses as growthResearchAndDevelopmentExpenses,
t2.growthGeneralAndAdministrativeExpenses as growthGeneralAndAdministrativeExpenses,
t2.growthSellingAndMarketingExpenses as growthSellingAndMarketingExpenses,
t2.growthOtherExpenses as growthOtherExpenses,
t2.growthOperatingExpenses as growthOperatingExpenses,
t2.growthCostAndExpenses as growthCostAndExpenses,
t2.growthInterestExpense as growthInterestExpense,
t2.growthDepreciationAndAmortization as growthDepreciationAndAmortization,
t2.growthEBITDA as growthEBITDA,
t2.growthEBITDARatio as growthEBITDARatio,
t2.growthOperatingIncome as growthOperatingIncome,
t2.growthOperatingIncomeRatio as growthOperatingIncomeRatio,
t2.growthTotalOtherIncomeExpensesNet as growthTotalOtherIncomeExpensesNet,
t2.growthIncomeBeforeTax as growthIncomeBeforeTax,
t2.growthIncomeBeforeTaxRatio as growthIncomeBeforeTaxRatio,
t2.growthIncomeTaxExpense as growthIncomeTaxExpense,
t2.growthNetIncome as growthNetIncome,
t2.growthNetIncomeRatio as growthNetIncomeRatio,
t2.growthEPS as growthEPS,
t2.growthEPSDiluted as growthEPSDiluted,
t2.growthWeightedAverageShsOut as growthWeightedAverageShsOut,
t2.growthWeightedAverageShsOutDil as growthWeightedAverageShsOutDil
from
	stocks_data.income_statement_annual as t1
left join stocks_data.income_statement_annual_growth as t2 on
	(t1.symbol = t2.symbol
		and t1.`date` = t2.`date` )
where
	t1.Symbol = :symbol
order by
	t1.date desc
limit 10