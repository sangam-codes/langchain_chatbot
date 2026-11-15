# [imports]
# Loads the env helper and the pieces that talk to Groq.
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# [main]
# Runs the terminal chat until the user quits.
# 1. Load the API key from .env.
# 2. Stop if the API key is missing.
# 3. Create the Groq model.
# 4. Start the conversation with the system instruction.
# 5. Show the welcome line.
# 6. Read the next user message.
# 7. Skip a blank message.
# 8. Quit when the user asks to leave.
# 9. Add the user message to the conversation.
# 10. Ask Groq for a reply.
# 11. Drop the last user message and show the error if the call fails.
# 12. Turn a non-text reply into text.
# 13. Keep the reply and print it.
def main() -> None:
    # [main-1] Load the API key from .env.
    load_dotenv()

    # [main-2] Stop if the API key is missing.
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        print("Set GROQ_API_KEY in a .env file before running.")
        sys.exit(1)

    # [main-3] Create the Groq model.
    model = ChatOpenAI(
        model="openai/gpt-oss-20b",
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
    )

    # [main-4] Start the conversation with the system instruction.
    history = [
        SystemMessage(content="You are a helpful assistant. Answer in 2-3 sentences.")
    ]

    # [main-5] Show the welcome line.
    print()
    print("LangChain Chatbot (terminal). Type quit to exit.")
    print()

    while True:
        # [main-6] Read the next user message.
        user_input = input("[You] ").strip()

        # [main-7] Skip a blank message.
        if not user_input:
            continue

        # [main-8] Quit when the user asks to leave.
        if user_input.lower() in {"quit", "exit", "q"}:
            break

        # [main-9] Add the user message to the conversation.
        history.append(HumanMessage(content=user_input))

        try:
            # [main-10] Ask Groq for a reply.
            reply = model.invoke(history).content
        except Exception as error:
            # [main-11] Drop the last user message and show the error if the call fails.
            print(f"Error: {error}")
            history.pop()
            continue

        # [main-12] Turn a non-text reply into text.
        if not isinstance(reply, str):
            reply = str(reply)

        # [main-13] Keep the reply and print it.
        history.append(AIMessage(content=reply))
        print(f"[Bot] {reply}")
        print()


# [entrypoint]
# Starts the chat when this file is run directly.
if __name__ == "__main__":
    main()
