from langchain_core.messages import HumanMessage, AIMessage

# env variables, only for api key right now
from dotenv import load_dotenv
load_dotenv()

from models import make_history_chain

# print(basic_context.invoke("where did harrison work?"))
# print(language_context.invoke({"question": "what is kensho?", "language": "english"}))

###### history stuff: #####


log = []
history_chain = make_history_chain()

log.append(
    history_chain.invoke(
        {
            "question": "where did harrison work?",
            "chat_history": [],
            "language": "english",
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
            "language": "english",
        }
    )
)

print(log)
