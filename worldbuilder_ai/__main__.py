from langchain_core.messages import HumanMessage, AIMessage

# env variables, only for api key right now
from dotenv import load_dotenv
load_dotenv()

from models import basic_chain, language_chain, history_chain

# print(basic_context.invoke("where did harrison work?"))
# print(language_context.invoke({"question": "what is kensho?", "language": "english"}))

###### history stuff: #####


log = []

log.append(
    history_chain.invoke(
        {
            "question": "where did harrison work?",
            "chat_history": [],
        }
    )
)

log.append(
    history_chain.invoke(
        {
            "question": "where did he work?",
            "chat_history": [
                HumanMessage(content="Who wrote this notebook?"),
                AIMessage(content="Harrison"),
            ],
        }
    )
)

print(log)
