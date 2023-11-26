"""
Import libraries
"""

from yahooquery import Ticker
import pandas as pd
import numpy as np

from scipy import stats

def get_value(data, key, default=0):
    if not isinstance(data, dict):
        return default
    try:
        return data.get(key, default)
    except Exception as e:
        print(f"Error accessing key '{key}' in data: {e}")
        return default

def get_multiples_from_yahoo(stock, comps):
    # Combine the symbols into a single list
    all_symbols = [stock] + comps
    data = []

    for ticker in all_symbols:
        symbol = Ticker(ticker)

        # Ensure the data structures are dictionaries before accessing
        key_stats = symbol.key_stats.get(ticker, {})
        price = symbol.price.get(ticker, {})
        summary_detail = symbol.summary_detail.get(ticker, {})

        name = get_value(price, "shortName", "N/A")
        price_to_book = get_value(key_stats, "priceToBook")
        peg_ratio = get_value(key_stats, "pegRatio")
        trailing_pe = get_value(summary_detail, "trailingPE")
        forward_pe = get_value(key_stats, "forwardPE")
        price_to_sales = get_value(summary_detail, "priceToSalesTrailing12Months")
        enterprise_to_ebitda = get_value(key_stats, "enterpriseToEbitda")

        data.append([ticker,name,  price_to_sales, price_to_book, trailing_pe, forward_pe, peg_ratio,enterprise_to_ebitda])


    columns = ['symbol','name',  'Price to Sales', 'Price to Book Value', 'Trailing PE', 'Forward PE', 'PEG Ratio','EV/EBITDA']
    df = pd.DataFrame(data, columns=columns)

    return df


def calculate_valuation_stats(df, stock_symbol):
    # Create an empty list to store dictionaries for each row
    result_data = []

    # Iterate over each multiple column
    for column in df.columns[2:]:
        # Exclude the stock being analyzed
        data = df[df['symbol'] != stock_symbol][column]

        # Remove negative values
        data = data[data >= 0]

        # Calculate z-scores to identify outliers
        z_scores = np.abs(stats.zscore(data))

        # Define a threshold (e.g., z-score < 3 to exclude outliers)
        threshold = 3

        # Exclude outliers based on z-scores
        data = data[z_scores < threshold]

        # Calculate min, max, and average values
        min_val = data.min()
        max_val = data.max()
        avg_val = data.mean()

        # Append the results to the result_data list
        result_data.append({'Multiple': column, 'Minimum': min_val, 'Maximum': max_val, 'Average': avg_val})

    # Create a DataFrame from the list of dictionaries
    result_df = pd.DataFrame(result_data)
    print (result_df)

    sales, bookvalue, ebitda,netincome, shares= get_ttm_from_yahoo(stock_symbol)
    print ("Aqui el statistic value para bookvale",bookvalue)
    mapping = {
        'Price to Sales': sales,
        'Price to Book Value': bookvalue*shares,
        'Trailing PE': netincome,
        'Forward PE': netincome,
        'PEG Ratio': netincome,
        'EV/EBITDA': ebitda
    }
    print (bookvalue)
    # Add a new 'statistic_value' column based on the mapping
    result_df['statistic_value'] = result_df['Multiple'].map(mapping)
    result_df['shares']=shares

    result_df['min_value']=result_df['Minimum']*result_df['statistic_value']/result_df['shares']
    result_df['max_value'] = result_df['Maximum'] * result_df['statistic_value'] / result_df['shares']

    # Replace min_value if it's less than or equal to zero
    for index, row in result_df.iterrows():
        if row['min_value'] <= 0:
            # Sort data to find the next smallest positive value
            sorted_data = sorted(df[df['symbol'] != stock_symbol][row['Multiple']])
            next_min_val = next((x for x in sorted_data if x > 0), None)
            if next_min_val is not None:
                result_df.at[index, 'min_value'] = next_min_val * row['statistic_value'] / row['shares']
            else:
                result_df.at[index, 'min_value'] = np.nan  # or some default value if no positive value is found

    print ("Here the multiples")
    print (result_df)
    return result_df

def get_ttm_from_yahoo(ticker):
    symbol = Ticker(ticker)
    df = symbol.income_statement(frequency='q')
    mask = df['periodType'] == '3M'
    filtered_df = df[mask].head(4)
    print (symbol.key_stats[ticker])

    bookvalue=symbol.key_stats[ticker]["bookValue"]
    sales=filtered_df["TotalRevenue"].sum()
    ebitda = filtered_df["EBITDA"].sum()
    netincome=filtered_df["NetIncome"].sum()
    sharesOutstanding=symbol.key_stats[ticker]["sharesOutstanding"]

    return sales, bookvalue, ebitda,netincome, sharesOutstanding

def combine_dataframes(df, df_profiles, df_dcf, low_target, high_target):
    # Extract the "Range" values and split them into min and max
    df_profiles['Range'] = df_profiles['Range'].str.split('-')
    df_profiles['Min_52week'] = df_profiles['Range'].apply(lambda x: float(x[0]))
    df_profiles['Max_52week'] = df_profiles['Range'].apply(lambda x: float(x[1]))

    dcf_min = df_dcf['DCF'] * 0.9
    dcf_max = df_dcf['DCF'] * 1.1

    # Filter the original DataFrame to include only specific metrics
    selected_metrics = ['Price to Sales', 'PE Ratio', 'Price to Book Value', 'Forward PE']
    filtered_df = df[df['Multiple'].isin(selected_metrics)]

    metrics = filtered_df['Multiple'].tolist()
    min_values = filtered_df['min_value'].tolist()
    max_values = filtered_df['max_value'].tolist()


    # Create a new DataFrame that combines the selected metrics, 52-week range, and DCF
    combined_df = pd.DataFrame({
        'Multiple': metrics + ['52 Week Range',"Analyst Target Price", 'DCF'],
        'Minimum': min_values + [df_profiles['Min_52week'].iloc[0],low_target, dcf_min.iloc[0]],
        'Maximum': max_values + [df_profiles['Max_52week'].iloc[0], high_target, dcf_max.iloc[0]]
    })

    print(combined_df)
    return combined_df

def get_stock_price(ticker):
    # Create a Ticker object
    symbol = Ticker(ticker)
    stock_price = symbol.financial_data[ticker]["currentPrice"]
    print (stock_price)
    return stock_price

def get_stock_target(ticker):
    symbol = Ticker(ticker)
    target_low = symbol.financial_data[ticker]["targetLowPrice"]
    target_high = symbol.financial_data[ticker]["targetHighPrice"]
    return target_low,target_high





