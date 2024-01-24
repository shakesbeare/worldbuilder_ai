
# RAG dependencies?
from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import format_document
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

vectorstore = FAISS.from_texts(
    [
        "harrison worked at Safeway",
        "harrison is blond",
        "barrison lives in colorado",
        "harrison is a gay clown"
    ],
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()
model = ChatOpenAI()


def make_basic_context():
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain


def make_language_context():

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}

    Answer in the following language: {language}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "language": itemgetter("language"),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain


def make_history_chain():

    condense_question_template = """

    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. Answer in English.
    You're allowed to make stuff up if you don't have the context to answer the question. 
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:

    """

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
        condense_question_template)

    answer_template = """
    You are a creative genius who has been tasked with creating a legendary world. Given your context provided, come up with answers for the following question:
    {context}

    Question: {question}
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_template)
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
        template="{page_content}")

    def combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    inputs = RunnableParallel(
        standalone_question=RunnablePassthrough.assign(
            chat_history=lambda x: get_buffer_string(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0.7)
        | StrOutputParser(),
    )

    context = {
        "context": itemgetter("standalone_question") | retriever | combine_documents,
        "question": lambda x: x["standalone_question"],
    }

    conversational_qa_chain = inputs | context | ANSWER_PROMPT | ChatOpenAI()

    return conversational_qa_chain
