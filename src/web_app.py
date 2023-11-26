"""
Import libraries
"""

import streamlit as st

from lib.load_data import *
from lib.plotting_functions import *
from lib.agent_functions import *
from lib.pdf_functions import *
from lib.get_multiples import *
import os
import configparser
import Inv_analysis
import io

from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,AIMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

config=configparser.ConfigParser()
config.read('c:\Conf\conf.ini')
OPENAI_API_KEY = config.get('OPENAI_API', 'OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

keys_defaults = {
    'count':0,
    'inv_analysis': '',
    'symbol': ''
}

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10= st.tabs(["Industry","Annual","Quarterly","Earnings Surp","Ratios","Valuation","Performance","Insiders","News","SP500"])

for key, default in keys_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default

def increment():
    st.session_state.count += 1

def step_one():
    original_title = '<p style="font-size: 25px;position: fixed;top:40px;">Value investing Assistant</p>'
    st.sidebar.markdown(original_title, unsafe_allow_html=True)
    st.sidebar.markdown('<p style="font-size: 15px;font-weight:bold">Step 1: Selecting stocks</p>',unsafe_allow_html=True)
    symbol = st.sidebar.text_area('Please specify the stock you would like to analyze first, followed by the companies (comps) you want to include in the analysis. Press the Proceed button to continue.',height=250)
    st.session_state.symbol = symbol
    st.sidebar.button('Proceed', on_click=increment)

def step_two():
    stocks_list = [stock.strip() for stock in st.session_state.symbol.split(',')]
    stock_to_analyze = stocks_list[0]
    comps = stocks_list[1:]

    with st.spinner('It seems like a very interesting stock to analyze. Let me see what I can do based on the data we have about this stock. Please wait for a few seconds.'):
        original_title = '<p style="font-size: 25px;position: fixed;top:40px;">Value investing Assistant</p>'
        st.sidebar.markdown(original_title, unsafe_allow_html=True)
        st.sidebar.markdown('<p style="font-size: 15px;color:grey">Step 1: Selecting the stock</p>', unsafe_allow_html=True)
        st.sidebar.markdown('<p style="font-size: 15px;font-weight:bold">Step 2: Analyzing the stock </p>',unsafe_allow_html=True)

        inv_analysis = Inv_analysis.chain_one.run(stock_to_analyze)
        st.session_state.inv_analysis = inv_analysis

        # Analyzing Income Statement

        dataframes = load_stock_data(stock_to_analyze)
        df_income_annual = dataframes['income_annual']
        df_income_quarter = dataframes['income_quarter']
        df_bs_annual = dataframes['bs_annual']
        df_cfs_annual = dataframes['cfs_annual']
        df_cfs_quarter = dataframes['cfs_quarter']
        df_ratios = dataframes['ratios_annual']
        df_sp500_5y = dataframes['sp500_5y']
        df_sp500_1y = dataframes['sp500_1y']
        df_earnings_surprise = dataframes['earnings_surprise']
        df_profiles = dataframes['profiles']
        df_insiders = dataframes['insiders']
        df_dcf = dataframes['dcf']
        df_CAPE= dataframes['cape']
        df_CAPE_stats=dataframes['cape_stats']


        dataframes = load_comps(stocks_list)
        df_ratios_comps = dataframes['ratios_annual_comps']

        df_daily_5y = dataframes['daily_prices_5y']
        df_daily_1y = dataframes['daily_prices_1y']

        industry_to_analyze = df_profiles['industry'][0]

    with tab1:

        url_image=df_profiles['image'][0]
        st.image(url_image, width=50)

        llm = ChatOpenAI(temperature=0)
        # List of templates
        templates = [
            "You are a value investor looking to better understand the industry and the competitive landscape of the following industry {industry_to_analyze} where the stock {stock_to_analyze} operates. Please analyze the current state of the industry, including key trends and barriers of entry.",
            "You are a value investor looking to better understand the industry and the competitive landscape of the following industry {industry_to_analyze} where the stock {stock_to_analyze} operates. Please analyze the competitive position of the stock, but avoid talking about key trends and barriers of entry.",
            "You are a value investor looking to better understand the industry and the competitive landscape of the following industry where the stock {stock_to_analyze} operates. Please analyze potential growth drivers, and any recent developments that might impact the stock price. Additionally, please offer an assessment of the stock's overall investment attractiveness based on your analysis."
        ]

        for template in templates:
            prompt = create_prompt(template, stock_to_analyze, industry_to_analyze)
            chain_industry = LLMChain(llm=llm, prompt=prompt)
            response = chain_industry.run(
                {'stock_to_analyze': stock_to_analyze, 'industry_to_analyze': industry_to_analyze})
            st.markdown(response, unsafe_allow_html=True)
            print (response)

        downloadPdf_from_link(df_income_annual['finalLink'][0])
        delete_create_index_pinecone()
        docsearch=load_embeddings()
        results=query_report(docsearch, '../txt/annual_report_industry.txt')

        st.write(
            '<p style="font-weight: bold; text-decoration: underline">Let´s analyze the annual report and see what is going on </p>',
            unsafe_allow_html=True)

        for result in results:
            st.markdown(f'<p style="font-weight: bold">{result["question"]}</p>', unsafe_allow_html=True)
            st.markdown(result['answer'], unsafe_allow_html=True)

    with tab2:
        #Plotting income Statement
        fig_income = plot_income(df_income_annual,"annual")
        fig_waterfall = waterfall_plot_pnl(df_income_annual)
        st.pyplot(fig_income)
        st.pyplot(fig_waterfall)

        #Presenting info in taBle format
        st.write('<p>Annual Income Statement (In millions): </p>', unsafe_allow_html=True)
        columns_selected = ['date','revenue', 'grossProfit', 'EBITDA', 'netIncome','grossProfitRatio', 'EBITDARatio', 'netIncomeRatio']
        df_is_limited=format_df_is(columns_selected,df_income_annual)
        st.dataframe(df_is_limited.sort_values(by='date', ascending=True).transpose())


        #Querying 10K filing
        results = query_report(docsearch, '../txt/annual_report_industry2.txt')
        for result in results:
            st.markdown(f'<p style="font-weight: bold">{result["question"]}</p>', unsafe_allow_html=True)
            st.markdown(result['answer'], unsafe_allow_html=True)

        #Querying dataframe using pandas agent
        prompt_isa = read_file('../txt/income_statement_prompt.txt')
        answer=execute_agent(df_income_annual, prompt_isa)
        st.markdown(answer, unsafe_allow_html=True)


        fig = plot_cfs(df_cfs_annual,"annual")
        st.pyplot(fig)


        results = query_report(docsearch, '../txt/annual_report_industry3.txt')
        for result in results:
            st.markdown(f'<p style="font-weight: bold">{result["question"]}</p>', unsafe_allow_html=True)
            st.markdown(result['answer'], unsafe_allow_html=True)

        prompt_bs=read_file('../txt/balance_sheet_prompt.txt')
        answer = execute_agent(df_bs_annual, prompt_bs)
        st.markdown(answer, unsafe_allow_html=True)

        prompt_cfs=read_file('../txt/cashflow_prompt.txt')
        answer = execute_agent(df_cfs_annual, prompt_cfs)
        st.markdown(answer, unsafe_allow_html=True)

    with tab3:

        #Plotting income Statement

        fig_income = plot_income(df_income_quarter,"quarterly")
        st.pyplot(fig_income)

        #Presenting info in taBle format
        st.write('<p>Quarterly Income Statement(In millions):</p>', unsafe_allow_html=True)
        df_is_limited = format_df_is(columns_selected, df_income_quarter)
        st.dataframe(df_is_limited.sort_values(by='date', ascending=True).transpose())

        downloadPdf_from_link(df_income_quarter['finalLink'][0])
        delete_create_index_pinecone()
        docsearch=load_embeddings()
        results=query_report(docsearch, '../txt/annual_report_industry4.txt')

        st.write(
            '<p style="font-weight: bold; text-decoration: underline">Let´s analyze the quarterly report and see what is going on </p>',
            unsafe_allow_html=True)

        for result in results:
            st.markdown(f'<p style="font-weight: bold">{result["question"]}</p>', unsafe_allow_html=True)
            st.markdown(result['answer'], unsafe_allow_html=True)

        fig = plot_cfs(df_cfs_quarter,"quarterly")
        st.pyplot(fig)

        results = query_report(docsearch, '../txt/annual_report_industry3.txt')
        for result in results:
            st.markdown(f'<p style="font-weight: bold">{result["question"]}</p>', unsafe_allow_html=True)
            st.markdown(result['answer'], unsafe_allow_html=True)

    with tab4:
        prompt_earnings=read_file('../txt/earnings_surprises.txt')
        answer = execute_agent(df_earnings_surprise, prompt_earnings)
        st.markdown(answer, unsafe_allow_html=True)
        df_earnings_surprise = format_df_earnings(df_earnings_surprise)
        st.dataframe(df_earnings_surprise)

    with tab5:

        fig = plot_ratio(df_ratios)
        st.pyplot(fig)

        prompt_ratios=read_file('../txt/ratios.txt')
        answer = execute_agent(df_ratios, prompt_ratios)
        st.markdown(answer, unsafe_allow_html=True)
        df_ratios_formatted=format_df_ratios(df_ratios)
        st.dataframe(df_ratios_formatted.sort_values(by='date', ascending=True).transpose())

        st.write(
            '<p style="font-weight: bold; text-decoration: underline">Let´s analyze ratios vs competition. </p>',
            unsafe_allow_html=True)

        fig = plot_ratios_comps(df_ratios_comps, stock_to_analyze)
        st.pyplot(fig)

        prompt_ratios=read_file('../txt/ratios_comps.txt')
        prompt_ratios=prompt_ratios.format(stock_to_analyze=stock_to_analyze)
        answer = execute_agent(df_ratios_comps, prompt_ratios)
        st.markdown(answer, unsafe_allow_html=True)

    with tab6:

        df_multiples = get_multiples_from_yahoo(stock_to_analyze, comps)
        fig = plot_valuation(df_ratios,  df_multiples, stock_to_analyze)
        st.pyplot(fig)

        prompt_hist_multiples=read_file('../txt/historical_multiples.txt')
        answer = execute_agent(df_ratios, prompt_hist_multiples)
        st.markdown(answer, unsafe_allow_html=True)


        fig = plot_valuation_comps(df_multiples, stock_to_analyze)
        st.pyplot(fig)
        prompt_comps_multiples=read_file('../txt/multiples_comps.txt')
        prompt_comps_multiples=prompt_comps_multiples.format(stock_to_analyze=stock_to_analyze)
        answer = execute_agent(df_multiples, prompt_comps_multiples)
        st.markdown(answer, unsafe_allow_html=True)

        df_results=calculate_valuation_stats(df_multiples, stock_to_analyze)
        low_target,high_target= get_stock_target(stock_to_analyze)
        df_results= combine_dataframes(df_results,df_profiles, df_dcf, low_target,high_target)
        stock_price=get_stock_price(stock_to_analyze)

        fig = plot_football_field(df_results,stock_price)
        st.pyplot(fig)

        prompt_footballfield = read_file('../txt/football_field_valuation.txt')
        prompt_footballfield = prompt_footballfield.format(stock_to_analyze=stock_to_analyze, stock_price=stock_price)
        answer = execute_agent(df_results, prompt_footballfield)
        st.markdown(answer, unsafe_allow_html=True)

    with tab7:

        df_normalized=normalized_dataframes(df_daily_5y, df_sp500_5y)
        fig = plot_normalized_performance(df_normalized)
        st.pyplot(fig)

        prompt_stock_price = read_file('../txt/stock_price.txt')
        prompt_stock_price = prompt_stock_price.format(stock_to_analyze=stock_to_analyze, stock_price=stock_price)
        answer = execute_agent(df_normalized, prompt_stock_price)
        st.markdown(answer, unsafe_allow_html=True)

        df_normalized = normalized_dataframes(df_daily_1y, df_sp500_1y)
        fig = plot_normalized_performance(df_normalized)
        st.pyplot(fig)

        prompt_stock_price = read_file('../txt/stock_price.txt')
        prompt_stock_price = prompt_stock_price.format(stock_to_analyze=stock_to_analyze, stock_price=stock_price)
        answer = execute_agent(df_normalized, prompt_stock_price)
        st.markdown(answer, unsafe_allow_html=True)

        fig = plot_stock_evolution(df_daily_5y, stock_to_analyze)
        st.pyplot(fig)
    
    with tab8:

        prompt_insiders=read_file('../txt/insiders.txt')
        answer = execute_agent(df_insiders, prompt_insiders)
        st.markdown(answer, unsafe_allow_html=True)

    with tab9:
        news = get_jsonparsed_data(stock_to_analyze)
        for i in range(5):
            st.markdown(news[i]["publishedDate"], unsafe_allow_html=True)
            st.markdown(news[i]["title"], unsafe_allow_html=True)
            st.markdown(news[i]["text"], unsafe_allow_html=True)

    with tab10:
        fig=plotting_sp500(df_CAPE,df_CAPE_stats)
        st.pyplot(fig)

        df_CAPE["1STD"]=df_CAPE_stats["1STD"][0]
        df_CAPE["2STD"] = df_CAPE_stats["2STD"][0]
        df_CAPE["AVERAGE"] = df_CAPE_stats["AVERAGE"][0]
        df_CAPE["NEG1STD"] = df_CAPE_stats["NEG1STD"][0]
        df_CAPE["NEG2STD"] = df_CAPE_stats["NEG2STD"][0]

def main():
    if 'count' not in st.session_state:
        st.session_state.count = 0

    if st.session_state.count == 0:
        step_one()
    elif st.session_state.count == 1:
        step_two()

if __name__ == "__main__":
    main()

