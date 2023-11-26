
"""Import libraries"""

import configparser

from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,AIMessagePromptTemplate,HumanMessagePromptTemplate

"""
read config.ini and get info
"""
config=configparser.ConfigParser()
config.read('c:\Conf\conf.ini')

OPENAI_API_KEY = config.get('OPENAI_API', 'OPENAI_API_KEY')
pinecone_api=config.get('pinecone', 'API_KEY')
pinecone_env=config.get('pinecone', 'ENVIRONMENT')

"""
Load Data
"""


def create_pandas_agent(df):
    """
    Create an agent that can access and use a large language model (LLM).

    Args:
        df:dataframe with information

    Returns:
        An agent that can access and use the LLM.
    """
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-4"),
        [df],
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    return agent

def create_prompt(template, stock_to_analyze, industry_to_analyze):
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    return ChatPromptTemplate.from_messages([system_message_prompt])


def execute_agent(df, prompt):
    agent=create_pandas_agent(df)
    answer = agent.run(prompt)
    return answer




