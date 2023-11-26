"""
Import libraries
"""
import os
import configparser
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import SequentialChain
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,AIMessagePromptTemplate,HumanMessagePromptTemplate

"""
read config.ini and get info
"""
config=configparser.ConfigParser()
config.read('C:/Conf/conf.ini')
OPENAI_API_KEY = config.get('OPENAI_API', 'OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


chat = ChatOpenAI(temperature=0)

system_template = """
        You are a Value Investing Consultant. You will be asked to analyze a stock: {symbol}.
        In the first step, I want you to clearly state the type of analysis you are going to do over the stock mentioned.
        """
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

example_human = HumanMessagePromptTemplate.from_template("Given the current market conditions\
 I am contemplating to invest in the {symbol}. In order to do that, I need to perform a detail analysis on the {symbol}. \
 Should the i invest in this company?")

example_ai = AIMessagePromptTemplate.from_template("1.Is there a reason to invest in Madrid.?\
            2.Does the Company have specific Real Estate assets typologies in mind for this market entry?")

human_template="{symbol}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

memory = ConversationBufferMemory()
inv_analysis_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai,human_message_prompt])
chain_one = LLMChain(llm=chat, prompt=inv_analysis_prompt, memory=memory)


template_industry="You are a value investor looking to better understand the industry and the competitive landscape of the following industry {industry_to_analyze} where the stock {stock_to_analyze} operates. Please analyze the current state of the industry, including key trends and barriers of entry."
system_message_prompt = SystemMessagePromptTemplate.from_template(template_industry)
first_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

chain_industry = LLMChain(llm=chat, prompt=first_prompt)

template_industry="You are a value investor looking to better understand the industry and the competitive landscape of the following industry {industry_to_analyze} where the stock {stock_to_analyze} operates. Please analyze the competitive position of the stock.."
system_message_prompt = SystemMessagePromptTemplate.from_template(template_industry)
first_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

chain_competitive_pos = LLMChain(llm=chat, prompt=first_prompt)

template_industry="You are a value investor looking to better understand the industry and the competitive landscape of the following industry where the stock {stock_to_analyze} operates. Please analyze potential growth drivers, and any recent developments that might impact the stock price. Additionally, please offer an assessment of the stock's overall investment attractiveness based on your analysis."
system_message_prompt = SystemMessagePromptTemplate.from_template(template_industry)
first_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

chain_growth_drivers = LLMChain(llm=chat, prompt=first_prompt)

