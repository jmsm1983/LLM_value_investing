# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:39:38 2020

@author: jmsm
"""
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
# Import SQL Aclhemy libs
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy import func
import configparser
import certifi
import json

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

config=configparser.ConfigParser()
config.read('c:\Conf\conf.ini')
FMP_API_KEY = config.get('FMP', 'API_KEY')

def open_connection(schema):
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    Connections to database                                                   
    1. Stablish connection
    2. Check if database is empty
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    config_file_path = ('C:\Conf\conf.ini')
    config = configparser.ConfigParser()
    config.read(config_file_path)

    username = config.get('mysql', 'username')
    password = config.get('mysql', 'password')
    hostname = config.get('mysql', 'hostname')

    connection_string = f"mysql://{username}:{password}@{hostname}/"

    engine = create_engine(connection_string+schema,echo=True)

    try:
        conn = engine.connect()
    except:
        print("Error opening Database")

    return conn

def execute_query_from_file(filepath, conn, **bindparams):
    with open(filepath, 'r') as query_file:
        query_str = text(query_file.read()).bindparams(**bindparams)
        result = conn.execute(query_str)

        # Check if the executed statement is a SELECT statement
        if result.returns_rows:
            return pd.DataFrame(result.fetchall(), columns=result.keys())

def read_file (file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def load_stock_data(stock_symbol):
    conn_ml_algo = open_connection('stocks_data')

    sql_files = {
        'income_annual': '../sql/01.get_income_statement_annual.sql',
        'income_quarter': '../sql/01.get_income_statement_quarterly.sql',
        'bs_annual': '../sql/01.get_balance_sheet_annual.sql',
        'cfs_annual': '../sql/01.get_cashflow_statement_annual.sql',
        'cfs_quarter': '../sql/01.get_cashflow_statement_quarterly.sql',
        'ratios_annual': '../sql/01.get_ratios_annual.sql',
        'sp500_1y': '../sql/01.get_sp500_1y.sql',
        'sp500_5y': '../sql/01.get_sp500_5y.sql',
        'earnings_surprise': '../sql/01.get_earningsurprises.sql',
        'profiles': '../sql/01.get_profiles.sql',
        'insiders': '../sql/01.get_insiders.sql',
        'dcf': '../sql/01.get_dcf.sql',
        'cape_stats':'../sql/03.get_CAPE_stats.sql',
        'cape':'../sql/03.get_CAPE_comps.sql'
    }

    dataframes = {}
    for key, path in sql_files.items():
        if key in ["sp500_1y", "sp500_5y","cape","cape_stats"]:  # If the key is one of these, don't pass the symbol
            df = execute_query_from_file(path, conn_ml_algo)
        else:
            df = execute_query_from_file(path, conn_ml_algo, symbol=stock_symbol)
        try:
            df['date'] = df['date'].astype(str)
        except:
            pass
        dataframes[key] = df

    return dataframes

def load_comps(stock_list):
    conn_ml_algo = open_connection('stocks_data')

    sql_files = {
        'ratios_annual_comps': '../sql/02.get_ratios_annual_comps.sql',
        'daily_prices_5y': '../sql/02.get_daily_prices_5y.sql',
        'daily_prices_1y': '../sql/02.get_daily_prices_1y.sql'
    }

    dataframes = {}
    for key, path in sql_files.items():
        df = execute_query_from_file(path, conn_ml_algo, symbol=stock_list)
        df['date'] = df['date'].astype(str)
        dataframes[key] = df

    return dataframes

def format_df_is(columns_selected, df):
    df = df[columns_selected]
    for col in ['grossProfitRatio', 'EBITDARatio', 'netIncomeRatio']:
        df[col] = df[col].apply(lambda x: "{:.1%}".format(x))

    numerical_cols = ['revenue', 'grossProfit', 'EBITDA', 'netIncome']
    df[numerical_cols] = df[numerical_cols].applymap(lambda x: "{:,.0f}".format(x / 1000000))

    return df

def format_df_earnings(df):
    df["pct_surprise"] = df["pct_surprise"].apply(lambda x: f'{x:.1%}')
    return df

def format_df_ratios(df):

    selected_columns=['date','currentRatio', 'quickRatio', 'cashRatio',
                    'daysOfSalesOutstanding','daysOfInventoryOutstanding', 'operatingCycle', 'daysOfPayablesOutstanding',
                    'cashConversionCycle', 'receivablesTurnover', 'payablesTurnover','inventoryTurnover', 'fixedAssetTurnover', 'assetTurnover',
                    'debtRatio', 'debtEquityRatio', 'longTermDebtToCapitalization', 'totalDebtToCapitalization', 'interestCoverage', 'shortTermCoverageRatios', 'capitalExpenditureCoverageRatio',
                    'cashFlowToDebtRatio','cashFlowCoverageRatios','dividendPaidAndCapexCoverageRatio','companyEquityMultiplier',
                    'grossProfitMargin', 'operatingProfitMargin','netProfitMargin', 'returnOnAssets', 'returnOnEquity','returnOnCapitalEmployed', 'operatingCashFlowSalesRatio',
                    'freeCashFlowOperatingCashFlowRatio', 'payoutRatio','dividendYield']
    df=df[selected_columns]

    # List of columns for each format type
    one_decimal_cols = ['currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding',
                        'daysOfInventoryOutstanding', 'operatingCycle', 'daysOfPayablesOutstanding',
                        'cashConversionCycle', 'receivablesTurnover', 'payablesTurnover',
                        'inventoryTurnover', 'fixedAssetTurnover', 'assetTurnover',
                        'debtRatio', 'debtEquityRatio', 'longTermDebtToCapitalization',
                        'totalDebtToCapitalization', 'interestCoverage', 'shortTermCoverageRatios',
                        'capitalExpenditureCoverageRatio', 'cashFlowToDebtRatio',
                        'cashFlowCoverageRatios', 'dividendPaidAndCapexCoverageRatio',
                        'companyEquityMultiplier']

    percentage_cols = ['grossProfitMargin', 'operatingProfitMargin', 'netProfitMargin',
                       'returnOnAssets', 'returnOnEquity', 'returnOnCapitalEmployed',
                       'operatingCashFlowSalesRatio', 'freeCashFlowOperatingCashFlowRatio',
                       'payoutRatio', 'dividendYield']

    # Apply formatting
    for col in one_decimal_cols:
        df[col] = df[col].apply(lambda x: "{:.1f}".format(x))

    for col in percentage_cols:
        df[col] = df[col].apply(lambda x: "{:.1%}".format(x))

    return df


def get_jsonparsed_data(ticker):
    url="https://financialmodelingprep.com/api/v3/stock_news?tickers="+ticker+"&page=0&apikey="+FMP_API_KEY
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)


