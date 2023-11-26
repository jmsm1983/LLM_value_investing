"""
Import libraries
"""

import urllib.request
import pdfkit
import pinecone
import configparser
import langchain.chains.question_answering.map_reduce_prompt
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import pinecone
from time import sleep

config_file_path = ('C:\Conf\conf.ini')
config = configparser.ConfigParser()
config.read(config_file_path)

PINECONE_API_KEY = config.get('pinecone', 'API_KEY')
PINECONE_ENV = config.get('pinecone', 'ENVIRONMENT')
OPENAI_API_KEY= config.get('OPENAI_API', 'OPENAI_API_KEY')

def downloadPdf_from_link(annual_link):
    path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
    output_path='../input/10k.pdf'
    pdfkit.from_url(annual_link, output_path, configuration=config)

def wait_on_index(index: str):
  """
  Takes the name of the index to wait for and blocks until it's available and ready.
  """
  ready = False
  while not ready:
    try:
      desc = pinecone.describe_index(index)
      if desc[7]['ready']:
        return True
    except pinecone.core.client.exceptions.NotFoundException:
      # NotFoundException means the index is created yet.
      pass
    sleep(5)

def delete_create_index_pinecone():

    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV# next to api key in console
    )
    index_name = "10k"
    namespace = "Starter"
    dimension = 1536

    try:
        print ("There is an index already created but i must delete it and then create a new one")
        active_index = pinecone.list_indexes()[0]
        pinecone.delete_index(active_index)
        pinecone.create_index(index_name, dimension=dimension, metric='cosine', pods=1, replicas=1, pod_type='p1.x1')
        wait_on_index(index_name)
    except:
        print ("There is no previous index created, so i must create one")
        pinecone.create_index(index_name, dimension=dimension, metric='cosine', pods=1, replicas=1, pod_type='p1.x1')
        wait_on_index(index_name)

    print("Creating embeddings...be patient")
    loader = PyPDFLoader(r'../input/10k.pdf')
    doc_10k = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(doc_10k)
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings,
        index_name="10k", namespace="Starter")

    print ("Ok,embeddings created succesfully")


def load_embeddings():
    print ("loading_embeddings")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Pinecone.from_existing_index(index_name="10k", embedding=embeddings, namespace="Starter")
    return docsearch


def query_report(docsearch, file_path):
    results=[]
    with open(file_path, 'r') as file:
        queries = [line.strip() for line in file.readlines()]

    for query in queries:
        docs = docsearch.similarity_search(query)

        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=query)

        results.append({
            'question': query,
            'answer': answer
        })

    return results


