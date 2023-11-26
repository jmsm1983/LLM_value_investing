select
	t1.symbol as symbol,
	t1.`date` as `date`,
	t1.reportedCurrency as reportedCurrency,
	t1.period as period,
	t1.cashAndCashEquivalents as cashAndCashEquivalents,
	t1.cashAndShortTermInvestments as cashAndShortTermInvestments,
	t1.netReceivables as netReceivables,
	t1.inventory as inventory,
	t1.otherCurrentAssets as otherCurrentAssets,
	t1.totalCurrentAssets as totalCurrentAssets,
	t1.propertyPlantEquipmentNet as propertyPlantEquipmentNet,
	t1.goodwill as goodwill,
	t1.intangibleAssets as intangibleAssets,
	t1.goodwillAndIntangibleAssets as goodwillAndIntangibleAssets,
	t1.longTermInvestments as longTermInvestments,
	t1.taxAssets as taxAssets,
	t1.otherNonCurrentAssets as otherNonCurrentAssets,
	t1.totalNonCurrentAssets as totalNonCurrentAssets,
	t1.otherAssets as otherAssets,
	t1.totalAssets as totalAssets,
	t1.accountPayables as accountPayables,
	t1.shortTermDebt as shortTermDebt,
	t1.taxPayables as taxPayables,
	t1.deferredRevenue as deferredRevenue,
	t1.otherCurrentLiabilities as otherCurrentLiabilities,
	t1.totalCurrentLiabilities as totalCurrentLiabilities,
	t1.longTermDebt as longTermDebt,
	t1.deferredRevenueNonCurrent as deferredRevenueNonCurrent,
	t1.deferrredTaxLiabilitiesNonCurrent as deferrredTaxLiabilitiesNonCurrent,
	t1.otherNonCurrentLiabilities as otherNonCurrentLiabilities,
	t1.totalNonCurrentLiabilities as totalNonCurrentLiabilities,
	t1.otherLiabilities as otherLiabilities,
	t1.totalLiabilities as totalLiabilities,
	t1.preferredStock as preferredStock,
	t1.commonStock as commonStock,
	t1.retainedEarnings as retainedEarnings,
	t1.accumulatedOtherComprehensiveIncomeLoss as accumulatedOtherComprehensiveIncomeLoss,
	t1.othertotalStockholdersEquity as othertotalStockholdersEquity,
	t1.totalStockholdersEquity as totalStockholdersEquity,
	t1.totalLiabilitiesAndStockholdersEquity as totalLiabilitiesAndStockholdersEquity,
	t1.totalInvestments as totalInvestments,
	t1.totalDebt as totalDebt,
	t1.netDebt as netDebt,
	t1.minorityInterest as minorityInterest,
	t1.capitalLeaseObligations as capitalLeaseObligations,
	t2.growthCashAndCashEquivalents as growthCashAndCashEquivalents,
	t2.growthTotalCurrentAssets as growthTotalCurrentAssets,
	t2.growthPropertyPlantEquipmentNet as growthPropertyPlantEquipmentNet,
	t2.growthGoodwillAndIntangibleAssets as growthGoodwillAndIntangibleAssets,
	t2.growthShortTermDebt as growthShortTermDebt,
	t2.growthLongTermDebt as growthLongTermDebt,
	t2.growthTotalLiabilities as growthTotalLiabilities,
	t2.growthTotalLiabilitiesAndStockholdersEquity as growthTotalLiabilitiesAndStockholdersEquity,
	t2.growthTotalDebt as growthTotalDebt,
	t2.growthNetDebt as growthNetDebt
from
	stocks_data.balance_sheet_annual as t1
left join stocks_data.balance_sheet_annual_growth as t2 on
	(t1.symbol = t2.symbol
		and t1.`date` = t2.`date` )
where
	t1.Symbol = :symbol
order by
	t1.date desc
limit 10




