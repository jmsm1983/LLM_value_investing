"""
Import libraries
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from datetime import datetime
from matplotlib.lines import Line2D
import matplotlib.patches as patches


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_income(df, data_type):
    fig, axs = plt.subplots(3, 1, figsize=(20, 15))  # Adjust the subplot layout

    # Sort the dataframe
    df = df.sort_values(by="date", ascending=True)

    metrics = [
        ("revenue", "grossProfitRatio", "growthRevenue"),
        ("EBITDA", "EBITDARatio", "growthEBITDA"),
        ("netIncome", "netIncomeRatio", "growthNetIncome")
    ]

    if data_type == 'annual':
        titles = [
            "Annual Revenues with Gross Profit Margin and YoY Growth",
            "Annual EBITDA with EBITDA Margin and YoY Growth",
            "Annual Net Income with Net Income Margin and YoY Growth"
        ]
    elif data_type == 'quarterly':
        titles = [
            "Quarterly Revenues with Gross Profit Margin",
            "Quarterly EBITDA with EBITDA Margin",
            "Quarterly Net Income with Net Income Margin"
        ]

    for ax, (metric, margin, growth), title in zip(axs, metrics, titles):
        # Plot the bars for the metric
        bars = ax.bar(df['date'], df[metric], color='skyblue', label=f'{metric.capitalize()}')

        # Positions for margin and growth texts
        y_position_margin = df[metric].max() * 1.15
        y_position_growth = df[metric].max() * 1.25
        min_value = min(df[metric].min(), 0)
        ax.set_ylim(min_value, y_position_growth * 1.20)

        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2, y_position_margin,
                    f"{df[margin].iloc[i] * 100:.1f}%",
                    ha='center', va='center', color='black', fontsize=9)

            ax.text(bar.get_x() + bar.get_width() / 2, y_position_growth,
                    f"{df[growth].iloc[i] * 100:.1f}%",
                    ha='center', va='center', color='blue', fontsize=9)

        # Additional logic for annual data
        if data_type == 'annual':
            beginning_value = df[metric].iloc[0]
            ending_value = df[metric].iloc[-1]
            n_years = len(df[metric])

            cagr = ((ending_value / beginning_value) ** (1 / n_years)) - 1
            cagr_percentage = cagr * 100

            projected_values = [beginning_value]
            for i in range(1, n_years):
                projected_values.append(projected_values[-1] * (1 + cagr))

            ax.plot(df['date'], projected_values, color='red', marker='o', linestyle='--',
                    label=f'CAGR {cagr_percentage:.1f}%')

            # CAGR annotation logic

            start_x_position = bars[0].get_x() + bars[0].get_width() / 2
            start_y_position = df[metric].iloc[0]
            end_x_position = bars[-1].get_x() + bars[-1].get_width() / 2
            end_y_position = df[metric].iloc[-1]

            ax.annotate('',
                        xy=(end_x_position, end_y_position),
                        xytext=(start_x_position, start_y_position),
                        arrowprops=dict(facecolor='black', arrowstyle='->'))

            label_x_position = start_x_position + (end_x_position - start_x_position) / 2
            label_y_position = start_y_position + (end_y_position - start_y_position) / 2
            ax.text(label_x_position, label_y_position, f'CAGR {cagr_percentage:.1f}%',
                    ha='center', va='center', color='black', fontsize=9, backgroundcolor='white', rotation=25)

        # Set legend, title, labels
        custom_entries = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10,
                                 label=f'{metric} Margin (%)'),
                          Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10,
                                 label=f'{metric} Growth (%)')]
        ax.legend(handles=[*ax.get_legend_handles_labels()[0], *custom_entries], loc='upper left', ncol=2)

        ax.set_title(title)
        ax.set_xlabel('Period')
        ax.set_ylabel(metric.capitalize())
        ax.set_xticks(df['date'])
        ax.set_xticklabels(df['date'], rotation=45)

    plt.tight_layout()
    return fig



def plot_cfs(df, data_type):
    fig, axs = plt.subplots(2, 1, figsize=(20, 9))  # Adjusted subplot layout for either annual or quarterly data

    # Sort the dataframe
    df = df.sort_values(by="date", ascending=True)

    metrics = [
        ("netCashProvidedByOperatingActivites", "growthNetCashProvidedByOperatingActivites"),
        ("freeCashFlow", "growthFreeCashFlow"),
    ]

    if data_type == 'annual':
        titles = [
            "Annual Operating Cashflow and YoY Growth",
            "Annual Free Cashflow and YoY Growth",
        ]
    elif data_type == 'quarterly':
        titles = [
            "Quarterly Operating Cashflow and YoY Growth",
            "Quarterly Free Cashflow and YoY Growth",
        ]

    for ax, (metric, growth), title in zip(axs, metrics, titles):
        bars = ax.bar(df['date'], df[metric], color='skyblue', label=f'{metric.capitalize()}')

        # Additional logic for annual data
        if data_type == 'annual':
            # CAGR calculations
            beginning_value = df[metric].iloc[0]
            ending_value = df[metric].iloc[-1]
            n_years = len(df[metric])

            cagr = ((ending_value / beginning_value) ** (1 / n_years)) - 1
            cagr_percentage = cagr * 100

            projected_values = [beginning_value]
            for i in range(1, n_years):
                projected_values.append(projected_values[-1] * (1 + cagr))

            ax.plot(df['date'], projected_values, color='red', marker='o', linestyle='--',
                    label=f'CAGR {cagr_percentage:.1f}%')

        # Positioning and labeling for growth
        y_position_growth = df[metric].max() * 1.25
        min_value = min(df[metric].min(), 0)
        ax.set_ylim(min_value, y_position_growth * 1.20)

        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2, y_position_growth,
                    f"{df[growth].iloc[i] * 100:.1f}%",
                    ha='center', va='center', color='blue', fontsize=9)

        # Set legend, title, labels
        custom_entries = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10,
                                 label=f'{metric} Growth (%)')]
        ax.legend(handles=[*ax.get_legend_handles_labels()[0], *custom_entries], loc='upper left', ncol=2)

        ax.set_title(title)
        ax.set_xlabel('Period')
        ax.set_ylabel(metric.capitalize())
        ax.set_xticks(df['date'])
        ax.set_xticklabels(df['date'], rotation=45)

    plt.tight_layout()
    return fig


def plot_ratio(df):
    df = df.sort_values(by="date", ascending=True)

    fig, axs = plt.subplots(4, 2, figsize=(15, 20), tight_layout=True)

    metrics = ["currentRatio", "cashConversionCycle", "returnOnEquity", "returnOnAssets",
               "interestCoverage", "debtEquityRatio", "dividendYield", "dividendPayoutRatio"]
    titles = ["Current Ratio", "Cash Conversion Cycle", "ROE", "ROA",
              "Coverage Ratio", "debt-equity Ratio", "Dividend Yield", "Payout Ratio"]

    axs = axs.ravel()  # Flatten the axs for easy indexing

    for ax, metric, title in zip(axs, metrics, titles):
        ax.bar(df['date'], df[metric], color='lightblue', label=f'{title}')

        # Calculating and plotting the historical average
        historical_avg = df[metric].mean()
        ax.axhline(y=historical_avg, color='red', linestyle='--', label='Historical Average')

        ax.set_title(title)
        ax.set_xlabel('Period')
        ax.set_ylabel('Value')
        ax.legend()

        # Set x-ticks for better visibility
        ax.set_xticks(df['date'])
        ax.set_xticklabels(df['date'], rotation=45)

    return fig

def plot_ratios_comps(df,stock):
    df = df.set_index('symbol').reindex([stock] + [s for s in df['symbol'] if s != stock]).reset_index()
    fig, axs = plt.subplots(4, 2, figsize=(15, 20), tight_layout=True)

    metrics = ["currentRatio", "cashConversionCycle", "returnOnEquity", "returnOnAssets",
               "interestCoverage", "debtEquityRatio", "dividendYield", "dividendPayoutRatio"]
    titles = ["Current Ratio", "Cash Conversion Cycle", "ROE", "ROA",
              "Coverage Ratio", "debt-equity Ratio", "Dividend Yield", "Payout Ratio"]

    axs = axs.ravel()  # Flatten the axs for easy indexing

    for ax, metric, title in zip(axs, metrics, titles):
        # Create a custom color list
        colors = ['#2F5597' if s == stock else 'lightblue' for s in df['symbol']]

        ax.bar(df['symbol'], df[metric], color=colors, label=f'{title}')

        # Calculating and plotting the historical average
        historical_avg = df[metric].mean()
        ax.axhline(y=historical_avg, color='red', linestyle='--', label='Average')

        ax.set_title(title)
        ax.set_xlabel('Stocks')
        ax.set_ylabel('Value')
        ax.legend()

        # Set x-ticks for better visibility
        ax.set_xticks(df['symbol'])
        ax.set_xticklabels(df['symbol'], rotation=45)

    return fig


def plot_valuation(df,df_last_multiples,stock_to_analyze):
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    rename_dict = {
        'Price to Sales': 'priceSalesRatio',
        'Price to Book Value': 'priceToBookRatio',
        'Trailing PE': 'priceEarningsRatio',
        'Forward PE': 'forwardPE',  # If you want to rename it differently, change the value here
        'PEG Ratio': 'priceEarningsToGrowthRatio',
        'EV/EBITDA': 'enterpriseValueMultiple'
    }

    # Use the rename method to update the column names
    df_last_multiples = df_last_multiples.rename(columns=rename_dict)
    df_last_multiples['date'] = today_date_str
    df_last_multiples=df_last_multiples[df_last_multiples["symbol"]==stock_to_analyze]
    print (df_last_multiples)

    df = df.sort_values(by="date", ascending=True)



    fig, axs = plt.subplots(6, 1, figsize=(10, 20))



    # Filter the DataFrame to include only the columns from the list

    metrics = ["priceSalesRatio", "priceEarningsRatio", "priceToBookRatio", "priceToFreeCashFlowsRatio", "priceEarningsToGrowthRatio", "enterpriseValueMultiple"]
    selected_colums=metrics+['date']
    df = df[selected_colums]

    new_row_data = {
        "priceSalesRatio": [df_last_multiples.iloc[0]["priceSalesRatio"]],
        "priceEarningsRatio": [df_last_multiples.iloc[0]["priceEarningsRatio"]],
        "priceToBookRatio": [df_last_multiples.iloc[0]["priceToBookRatio"]],
        "priceToFreeCashFlowsRatio": [0],  # Assuming you want this to be zero
        "priceEarningsToGrowthRatio": [df_last_multiples.iloc[0]["priceEarningsToGrowthRatio"]],
        "enterpriseValueMultiple": [df_last_multiples.iloc[0]["enterpriseValueMultiple"]],
        "date": [df_last_multiples.iloc[0]["date"]]
    }

    # Create a DataFrame for the new row with an arbitrary index

    new_row_df = pd.DataFrame(new_row_data)

    # Use concat to append the new row to the DataFrame, reset the index
    df = pd.concat([df, new_row_df], ignore_index=True)

    print (df.columns)
    titles = ["Price to Sales", "PE Ratio", "Price to Book Value", "Price to FCF", "PEG", "EBITDA multiple"]

    # Assuming df_last_multiples only has one row of data for the last values
    last_values = df_last_multiples.iloc[0] if not df_last_multiples.empty else None

    print (last_values)

    for ax, metric, title in zip(axs, metrics, titles):
        bars=ax.bar(df['date'], df[metric], color='skyblue', label=f'{title}')
        # Change the color of the last bar
        bars[-1].set_color('#2F5597')

        # Calculating and plotting the historical average
        historical_avg = df[metric].mean()
        ax.axhline(y=historical_avg, color='red', linestyle='--', label='Historical Average')

        # Calculating and plotting the 3-year moving average
        moving_avg = df[metric].rolling(window=3).mean()
        ax.plot(df['date'], moving_avg, color='green', linestyle='--', label='3-Year Moving Average')

        ax.set_title(title)
        ax.set_xlabel('Period')
        ax.set_ylabel('Multiple (x)')
        ax.legend()

        # Set x-ticks for better visibility
        ax.set_xticks(df['date'])
        ax.set_xticklabels(df['date'], rotation=45)

    plt.tight_layout()
    return fig


def plot_valuation_comps(df,stock):
    df = df.set_index('symbol').reindex([stock] + [s for s in df['symbol'] if s != stock]).reset_index()
    fig, axs = plt.subplots(6, 1, figsize=(10, 20))


    metrics = ['Price to Sales', 'Price to Book Value', 'Trailing PE', 'Forward PE', 'PEG Ratio', 'EV/EBITDA']
    titles = ['Price to Sales', 'Price to Book Value', 'Trailing PE', 'Forward PE', 'PEG Ratio', 'EV/EBITDA']

    for ax, metric, title in zip(axs, metrics, titles):

        colors = ['#2F5597' if s == stock else 'lightblue' for s in df['symbol']]
        ax.bar(df['symbol'], df[metric], color=colors, label=f'{title}')

        # Calculating and plotting the historical average
        historical_avg = df[metric].mean()
        ax.axhline(y=historical_avg, color='red', linestyle='--', label='Average')

        ax.set_title(title)
        ax.set_xlabel('Period')
        ax.set_ylabel('Multiple (x)')
        ax.legend()

        # Set x-ticks for better visibility
        ax.set_xticks(df['symbol'])
        ax.set_xticklabels(df['symbol'], rotation=45)

    plt.tight_layout()
    return fig


def waterfall_plot_pnl(df):
    # Extract data for the latest year
    latest_data = df.sort_values(by="date", ascending=False).iloc[0]

    # Define the sequence of line items for the waterfall chart
    items_sequence = [
        ("Revenue", latest_data["revenue"]),
        ("Cost of Revenue", -latest_data["costOfRevenue"]),
        ("Gross Profit", latest_data["grossProfit"]),
        ("R&D Expenses", -latest_data["ResearchAndDevelopmentExpenses"]),
        ("G&A Expenses", -latest_data["GeneralAndAdministrativeExpenses"]+latest_data["depreciationAndAmortization"]),
        ("S&M Expenses", -latest_data["SellingAndMarketingExpenses"]),
        ("EBITDA", latest_data["EBITDA"]),
        ("D&A", -latest_data["depreciationAndAmortization"]),
        ("Other Expenses", +latest_data["otherExpenses"]),
        ("Income Tax Expense", -latest_data["incomeTaxExpense"]),
        ("Net Income", latest_data["netIncome"])
    ]

    # Calculate the starting positions for each step of the waterfall
    starting_positions = [0]
    for i, (label, value) in enumerate(items_sequence[:-1]):
        if label in ["Revenue", "Gross Profit", "EBITDA","Net Income"]:
            starting_positions.append(value)
        else:
            starting_positions.append(starting_positions[-1] + value)

    # Adjust the starting position for Net Income
    starting_positions[-1] = starting_positions[-2] + items_sequence[-2][1]

    # Create the waterfall plot
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plotting each bar
    for i, (label, value) in enumerate(items_sequence):
        if label in ["Revenue","Net Income"]:  # Total bars
            ax.bar(label, value, color='#2F5597')
        elif label in ["Gross Profit","EBITDA"]:  # Final bar
            ax.bar(label, value, color='#87CEEB')
        else:  # Deduction bars
            ax.bar(label, value, bottom=starting_positions[i], color='#FF9E9E' if value < 0 else '#CDE5CD')

    # Formatting
    ax.set_title('P&L Waterfall Chart with GP and EBITDA for the Latest Year')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))  # Format y-axis as currency
    ax.set_ylabel('Amount ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def normalized_dataframes(df,sp500_df):
    # Check for duplicate date-symbol combinations in both dataframes
    duplicates_df = df[df.duplicated(subset=['date', 'symbol'], keep=False)]
    duplicates_sp500 = sp500_df[sp500_df.duplicated(subset=['date', 'symbol'], keep=False)]

    # If there are duplicates, print them out (you can remove this section later)
    if not duplicates_df.empty:
        print("Duplicates in stock data:")
        print(duplicates_df)

    if not duplicates_sp500.empty:
        print("Duplicates in S&P 500 data:")
        print(duplicates_sp500)

    # Remove duplicates for now (you might want to inspect them as printed above)
    df = df.drop_duplicates(subset=['date', 'symbol'])
    sp500_df = sp500_df.drop_duplicates(subset=['date', 'symbol'])

    # Concatenate and pivot as before
    combined_df = pd.concat([df, sp500_df])
    df_pivot = combined_df.pivot(index='date', columns='symbol', values='adjClose')

    # Rest of your function remains unchanged
    df_normalized = df_pivot / df_pivot.iloc[0]

    return df_normalized



def plot_normalized_performance(df_normalized):
    """
    Plots the normalized performance of multiple stocks and the S&P 500 in a dataframe.

    """

    fig, ax = plt.subplots(figsize=(12, 6))
    df_normalized.plot(ax=ax)

    ax.set_title('Stock and S&P 500 Performance (Normalized)')
    ax.set_ylabel('Cumulative Returns (Starting from 100%)')
    ax.set_xlabel('Date')
    plt.legend(title='Symbols')
    plt.tight_layout()

    return fig


def plot_stock_evolution(df, symbol):
    df = df.drop_duplicates()
    # Ensure the date column is of datetime type
    df['date'] = pd.to_datetime(df['date'])
    # Filter by symbol
    df = df[df['symbol'] == symbol]

    # Current year
    current_year = datetime.now().year

    # Define the time frames
    last_5_years = df[df['date'].dt.year >= current_year - 5]
    last_year = df[df['date'].dt.year == current_year - 1]
    ytd = df[df['date'].dt.year == current_year]

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot data for each time frame
    timeframes = [last_5_years, last_year, ytd]
    titles = ['Last 5 Years', 'Last Year', 'Year-to-Date']

    for ax, timeframe, title in zip(axs, timeframes, titles):
        ax.plot(timeframe['date'], timeframe['adjClose'], label=f'{symbol} Price')
        ax.set_title(f'{symbol} Price Evolution: {title}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    return fig


def plot_football_field(df,stock_price):
    # Filter the DataFrame to include only specific metrics
    selected_metrics = ['52 Week Range','Analyst Target Price', 'Price to Sales', 'Forward PE', 'Price to Book Value','DCF']
    filtered_df = df[df['Multiple'].isin(selected_metrics)]

    metrics = filtered_df['Multiple'].tolist()
    min_values = filtered_df['Minimum'].tolist()
    max_values = filtered_df['Maximum'].tolist()

    num_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust figsize for vertical display

    for i in range(num_metrics):
        y = [i, i]
        x = [min_values[i], max_values[i]]
        ax.plot(x, y, marker='o', linestyle='-', markersize=8)
        ax.annotate(f'{min_values[i]:.2f}', (x[0], y[0]), textcoords="offset points", xytext=(-15,-5), ha='center')
        ax.annotate(f'{max_values[i]:.2f}', (x[1], y[1]), textcoords="offset points", xytext=(15,5), ha='center')

    ax.axvline(x=stock_price, color='red', linestyle='--', label='Current Price')


    ax.set_yticks(range(num_metrics))
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Values')  # Adjust the label for the x-axis
    ax.set_title('Vertical Football Field Graph with Selected Metrics')
    ax.legend()

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    return fig



def plotting_sp500(df_cape, df_cape_stats):
    # Sort df_cape by Date in ascending order
    df_cape = df_cape.sort_values(by='Date')

    # Extract data from df_cape
    dates = df_cape['Date']
    cape = df_cape['CAPE']
    sp500 = df_cape['SP500']  # Add SP500 data

    # Extract data from df_cape_stats
    avg = df_cape_stats["AVERAGE"][0]
    std1 = df_cape_stats["1STD"][0]
    std2 = df_cape_stats["2STD"][0]
    neg1std = df_cape_stats["NEG1STD"][0]
    neg2std = df_cape_stats["NEG2STD"][0]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot CAPE on the primary y-axis (left)
    ax1.plot(dates, cape, label='CAPE', color='black')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('CAPE', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a secondary y-axis (right) for SP500
    ax2 = ax1.twinx()
    ax2.plot(dates, sp500, label='SP500', color='blue')
    ax2.set_ylabel('SP500', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Add horizontal lines
    ax1.axhline(avg, color='yellow', linestyle='-', label='Average')
    ax1.axhline(std1, color='red', linestyle='--', label='1 Std Dev')
    ax1.axhline(std2, color='red', linestyle='--', label='2 Std Dev')
    ax1.axhline(neg1std, color='green', linestyle='--', label='-1 Std Dev')
    ax1.axhline(neg2std, color='green', linestyle='--', label='-2 Std Dev')

    # Customize the plot
    ax1.set_title('Historical SP500 and CAPE')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Show the plot
    plt.tight_layout()

    return fig
