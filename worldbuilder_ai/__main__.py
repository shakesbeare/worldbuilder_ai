from langchain_core.messages import HumanMessage, AIMessage
from models import make_history_chain
import datetime
from rich.console import Console
from dotenv import load_dotenv
load_dotenv()
console = Console()

def main():
    log = []
    history = []
    qdrant, chain = make_history_chain()
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
        console.print("AI >\n" + response.content, style="bold yellow")

        human_message = HumanMessage(content=user_input)
        ai_message = response

        history.append(human_message)
        history.append(ai_message)
        qdrant.add_texts([human_message.content, ai_message.content])

    with open(f"logs/{now}.txt", "w") as f:
        f.writelines(log)

if __name__ == "__main__":
    main()
