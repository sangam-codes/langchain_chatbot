import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in a .env file before running.")
        sys.exit(1)

    user_input = input("You: ").strip()
    if not user_input:
        print("Bot: Type a message and run again.")
        return

    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke(
        [
            SystemMessage(content="Reply in one short sentence."),
            HumanMessage(content=user_input),
        ]
    )
    print(f"Bot: {response.content}")


if __name__ == "__main__":
    main()
