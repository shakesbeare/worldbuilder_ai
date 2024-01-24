from langchain_core.messages import HumanMessage, AIMessage
from models import make_history_chain
import datetime

# env variables, only for api key right now
from dotenv import load_dotenv
load_dotenv()

def main():
    log = []
    history = []
    chain = make_history_chain()
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    while (user_input := input("User > ")) != "exit":
        log.append("User> " + user_input + "\n")
        response = chain.invoke(
            {
                "question": user_input,
                "chat_history": history,
            }
        )
        log.append("Agent> " + response.content + "\n")
        print(response.content)

        human_message = HumanMessage(content=user_input)
        ai_message = response

        history.append(human_message)
        history.append(ai_message)

    with open(f"logs/{now}.txt", "w") as f:
        f.writelines(log)

if __name__ == "__main__":
    main()
