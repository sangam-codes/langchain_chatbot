# [imports]
# Loads the env helper, chat messages, shared agent helpers, and HITL resume types.
import os
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent_common import (
    action_requests_from_value,
    build_agent,
    decision_from_answer,
    has_interrupts,
    messages_from_result,
    reply_text_from_messages,
    tool_lines_from_messages,
)


# [ask_about_interrupts]
# Asks y/n for each paused tool and builds the resume decisions.
# 1. Walk each paused interrupt from LangGraph.
# 2. Ask about every tool call in that interrupt.
# 3. Return the decisions list for Command(resume=...).
def ask_about_interrupts(result) -> dict:
    decisions = []

    # [ask_about_interrupts-1] Walk each paused interrupt from LangGraph.
    for interrupt in result.interrupts:
        value = interrupt.value or {}

        # [ask_about_interrupts-2] Ask about every tool call in that interrupt.
        for action in action_requests_from_value(value):
            name = action.get("name", "tool")
            args = action.get("args", {})
            print(f"[Approval needed] {name}: {args}")
            answer = input("Approve this tool? [y/n] ").strip().lower()
            decisions.append(decision_from_answer(answer))

    # [ask_about_interrupts-3] Return the decisions list for Command(resume=...).
    return {"decisions": decisions}


# [run_agent_turn]
# Runs the agent until it finishes, pausing for approval when needed.
# 1. Start or resume the agent for this turn.
# 2. If tools need approval, ask the user and resume.
# 3. Return the finished result.
def run_agent_turn(agent, payload, config):
    # [run_agent_turn-1] Start or resume the agent for this turn.
    result = agent.invoke(payload, config=config, version="v2")

    # [run_agent_turn-2] If tools need approval, ask the user and resume.
    while has_interrupts(result):
        resume_payload = ask_about_interrupts(result)
        result = agent.invoke(
            Command(resume=resume_payload),
            config=config,
            version="v2",
        )

    # [run_agent_turn-3] Return the finished result.
    return result


# [print_turn]
# Prints new tool results and the final assistant reply for this turn.
# 1. Print each new tool result from this turn.
# 2. Print the final assistant reply and return the new count.
def print_turn(messages: list, seen_count: int) -> int:
    # [print_turn-1] Print each new tool result from this turn.
    for line in tool_lines_from_messages(messages, seen_count, markdown=False):
        print(line)

    # [print_turn-2] Print the final assistant reply and return the new count.
    print(f"[Bot] {reply_text_from_messages(messages)}")
    print()
    if not messages:
        return seen_count
    return len(messages)


# [wants_to_quit]
# True when the user typed a quit command.
def wants_to_quit(user_input: str) -> bool:
    return user_input.lower() in {"quit", "exit", "q"}


# [print_welcome]
# Shows the startup lines for the terminal chat.
def print_welcome() -> None:
    print()
    print("LangChain Chatbot (terminal + memory). Type quit to exit.")
    print("Tools: get_current_time, calculate, remember_fact, recall_facts, delete_note")
    print("delete_note asks for y/n. Try saving hobby / favorite color in .memory.json.")
    print()


# [main]
# Runs the terminal agent chat until the user quits.
# 1. Load the API key from .env.
# 2. Stop if the API key is missing.
# 3. Build the agent with tools, saved state, and delete approval.
# 4. Start a thread id and message counter for this chat.
# 5. Show the welcome line.
# 6. Read the next user message.
# 7. Skip a blank message.
# 8. Quit when the user asks to leave.
# 9. Ask the agent for a reply (it may pause for approval).
# 10. Show the error if the call fails.
# 11. Print tools and the final assistant reply.
def main() -> None:
    # [main-1] Load the API key from .env.
    load_dotenv()

    # [main-2] Stop if the API key is missing.
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        print("Set GROQ_API_KEY in a .env file before running.")
        sys.exit(1)

    # [main-3] Build the agent with tools, saved state, and delete approval.
    agent = build_agent(key, InMemorySaver())

    # [main-4] Start a thread id and message counter for this chat.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    seen_message_count = 0

    # [main-5] Show the welcome line.
    print_welcome()

    while True:
        # [main-6] Read the next user message.
        user_input = input("[You] ").strip()

        # [main-7] Skip a blank message.
        if not user_input:
            continue

        # [main-8] Quit when the user asks to leave.
        if wants_to_quit(user_input):
            break

        try:
            # [main-9] Ask the agent for a reply (it may pause for approval).
            result = run_agent_turn(
                agent,
                {"messages": [HumanMessage(content=user_input)]},
                config,
            )
        except Exception as error:
            # [main-10] Show the error if the call fails.
            print(f"Error: {error}")
            continue

        # [main-11] Print tools and the final assistant reply.
        messages = messages_from_result(result)
        seen_message_count = print_turn(messages, seen_message_count)


# [entrypoint]
# Starts the chat when this file is run directly.
if __name__ == "__main__":
    main()
