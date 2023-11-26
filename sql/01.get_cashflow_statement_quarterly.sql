select
t1.`date` as `date`,
t1.symbol as symbol,
t1.reportedCurrency as reportedCurrency,
t1.period as period,
t1.netIncome as netIncome,
t1.depreciationAndAmortization as depreciationAndAmortization,
t1.deferredIncomeTax as deferredIncomeTax,
t1.stockBasedCompensation as stockBasedCompensation,
t1.changeInWorkingCapital as changeInWorkingCapital,
t1.accountsReceivables as accountsReceivables,
t1.inventory as inventory,
t1.accountsPayables as accountsPayables,
t1.otherWorkingCapital as otherWorkingCapital,
t1.otherNonCashItems as otherNonCashItems,
t1.netCashProvidedByOperatingActivites as netCashProvidedByOperatingActivites,
t1.investmentsInPropertyPlantAndEquipment as investmentsInPropertyPlantAndEquipment,
t1.acquisitionsNet as acquisitionsNet,
t1.purchasesOfInvestments as purchasesOfInvestments,
t1.salesMaturitiesOfInvestments as salesMaturitiesOfInvestments,
t1.otherInvestingActivites as otherInvestingActivites,
t1.netCashUsedForInvestingActivites as netCashUsedForInvestingActivites,
t1.debtRepayment as debtRepayment,
t1.commonStockIssued as commonStockIssued,
t1.commonStockRepurchased as commonStockRepurchased,
t1.dividendsPaid as dividendsPaid,
t1.otherFinancingActivites as otherFinancingActivites,
t1.netCashUsedProvidedByFinancingActivities as netCashUsedProvidedByFinancingActivities,
t1.effectOfForexChangesOnCash as effectOfForexChangesOnCash,
t1.netChangeInCash as netChangeInCash,
t1.cashAtEndOfPeriod as cashAtEndOfPeriod,
t1.cashAtBeginningOfPeriod as cashAtBeginningOfPeriod,
t1.operatingCashFlow as operatingCashFlow,
t1.capitalExpenditure as capitalExpenditure,
t1.freeCashFlow as freeCashFlow,
t2.growthDepreciationAndAmortization as growthDepreciationAndAmortization,
t2.growthStockBasedCompensation as growthStockBasedCompensation,
t2.growthChangeInWorkingCapital as growthChangeInWorkingCapital,
t2.growthNetCashProvidedByOperatingActivites as growthNetCashProvidedByOperatingActivites,
t2.growthNetCashUsedForInvestingActivites as growthNetCashUsedForInvestingActivites,
t2.growthDebtRepayment as growthDebtRepayment,
t2.growthCommonStockIssued as growthCommonStockIssued,
t2.growthCommonStockRepurchased as growthCommonStockRepurchased,
t2.growthDividendsPaid as growthDividendsPaid,
t2.growthOtherFinancingActivites as growthOtherFinancingActivites,
t2.growthNetCashUsedProvidedByFinancingActivities as growthNetCashUsedProvidedByFinancingActivities,
t2.growthOperatingCashFlow as growthOperatingCashFlow,
t2.growthCapitalExpenditure as growthCapitalExpenditure,
t2.growthFreeCashFlow as growthFreeCashFlow
from
        stocks_data.cash_flow_statement_quarter as t1
left join stocks_data.cash_flow_statement_quarter_growth as t2 on
        (t1.symbol = t2.symbol
                and t1.`date` = t2.`date` )
where
	t1.Symbol = :symbol
order by
	t1.date desc
limit 10
