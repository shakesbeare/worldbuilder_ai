from langchain_core.messages import HumanMessage, AIMessage
from models import make_history_chain
import datetime
from rich.console import Console
from dotenv import load_dotenv
import json
import os

load_dotenv()

console = Console()

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def ask_preliminary_questions():
    details = []
    while (
        user_input := input(
            "\nProvide any details about the world, or type 'done' to be done:\n"
        )
    ) != "done":
        details.append(user_input)

    return details


def main():
    if not os.path.exists("worlds.json"):
        with open("worlds.json", "w") as f:
            json.dump({}, f)

    with open("worlds.json", "r") as f:
        worlds = json.load(f)

    world_collection = input("Enter the name of the project to open: ").strip().lower()
    if world_collection not in worlds:
        details = ask_preliminary_questions()
        worlds[world_collection] = details

        with open("worlds.json", "w") as f:
            json.dump(worlds, f)
    else:
        details = worlds[world_collection]

    clear()

    log = []
    history = []
    qdrant, chain = make_history_chain(details, world_collection)
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
