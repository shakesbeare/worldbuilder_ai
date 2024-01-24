import os
import openai
from langchain import hub
from langchain.agents import AgentExecutor, AgentType, load_tools, initialize_agent, create_react_agent
from langchain_community.chat_models import ChatAnyscale
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

#RAG dependencies?
from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from dotenv import dotenv_values

config = dotenv_values(".env")

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


chain.invoke("where did harrison work?")
