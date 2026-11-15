# [imports]
# Loads the env helper, chat messages, shared tools, and the agent builder.
import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from tools import TOOLS

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Use get_current_time for the current date or time. "
    "Use calculate for math. "
    "Answer in 2-3 sentences."
)


# [main]
# Runs the terminal agent chat until the user quits.
# 1. Load the API key from .env.
# 2. Stop if the API key is missing.
# 3. Create the Groq model.
# 4. Build the agent with the shared tools.
# 5. Start an empty history so the agent can remember earlier turns.
# 6. Show the welcome line.
# 7. Read the next user message.
# 8. Skip a blank message.
# 9. Quit when the user asks to leave.
# 10. Add the user message to the conversation.
# 11. Ask the agent for a reply (it may call tools first).
# 12. Drop the last user message and show the error if the call fails.
# 13. Keep the full reply history, including any tool results.
# 14. Print each tool result from this turn.
# 15. Print the final assistant reply.
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

    # [main-4] Build the agent with the shared tools.
    agent = create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    # [main-5] Start an empty history so the agent can remember earlier turns.
    history: list = []

    # [main-6] Show the welcome line.
    print()
    print("LangChain Chatbot (terminal + tools). Type quit to exit.")
    print("Tools: get_current_time, calculate")
    print()

    while True:
        # [main-7] Read the next user message.
        user_input = input("[You] ").strip()

        # [main-8] Skip a blank message.
        if not user_input:
            continue

        # [main-9] Quit when the user asks to leave.
        if user_input.lower() in {"quit", "exit", "q"}:
            break

        # [main-10] Add the user message to the conversation.
        prior_len = len(history)
        history.append(HumanMessage(content=user_input))

        try:
            # [main-11] Ask the agent for a reply (it may call tools first).
            result = agent.invoke({"messages": history})
        except Exception as error:
            # [main-12] Drop the last user message and show the error if the call fails.
            print(f"Error: {error}")
            history = history[:prior_len]
            continue

        # [main-13] Keep the full reply history, including any tool results.
        history = list(result["messages"])

        # [main-14] Print each tool result from this turn.
        for message in history[prior_len:]:
            if isinstance(message, ToolMessage):
                tool_name = message.name or "tool"
                print(f"[Tool] {tool_name}: {message.content}")

        # [main-15] Print the final assistant reply.
        reply = history[-1].content
        if not isinstance(reply, str):
            reply = str(reply)
        print(f"[Bot] {reply}")
        print()


# [entrypoint]
# Starts the chat when this file is run directly.
if __name__ == "__main__":
    main()
