# [imports]
# Loads shared agent helpers used by the terminal and Streamlit apps.
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

from tools import TOOLS

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Use get_current_time for the current date or time. "
    "Use calculate for math. "
    "Use delete_note only when the user asks to delete a note. "
    "If delete_note was rejected or blocked, say the note was not deleted. "
    "Do not claim a delete succeeded unless a tool result shows it was pretend-deleted. "
    "If the user asks to delete again later, call delete_note again. "
    "Answer in 2-3 sentences."
)

HITL_INTERRUPT_ON = {
    "get_current_time": False,
    "calculate": False,
    "delete_note": {"allowed_decisions": ["approve", "reject"]},
}

REJECT_MESSAGE = (
    "User rejected this delete. Do not retry unless the user asks again."
)


# [messages_from_result]
# Reads the message list from a normal result or a version=v2 graph result.
# 1. Prefer messages from a version=v2 graph value.
# 2. Fall back to a plain dict result.
# 3. Return an empty list when neither shape matches.
def messages_from_result(result) -> list:
    # [messages_from_result-1] Prefer messages from a version=v2 graph value.
    if hasattr(result, "value") and isinstance(result.value, dict):
        return list(result.value.get("messages", []))

    # [messages_from_result-2] Fall back to a plain dict result.
    if isinstance(result, dict):
        return list(result.get("messages", []))

    # [messages_from_result-3] Return an empty list when neither shape matches.
    return []


# [has_interrupts]
# True when LangGraph paused for human approval.
def has_interrupts(result) -> bool:
    return bool(getattr(result, "interrupts", None))


# [action_requests_from_value]
# Normalizes one interrupt payload into a list of tool calls waiting for approval.
# 1. Use the batched action_requests list when it is present.
# 2. Treat a single named action as one request.
# 3. Return an empty list when nothing is waiting.
def action_requests_from_value(value: dict) -> list:
    # [action_requests_from_value-1] Use the batched action_requests list when it is present.
    action_requests = value.get("action_requests", [])
    if action_requests:
        return action_requests

    # [action_requests_from_value-2] Treat a single named action as one request.
    if value.get("name"):
        return [value]

    # [action_requests_from_value-3] Return an empty list when nothing is waiting.
    return []


# [first_action_from_interrupts]
# Picks the first paused tool call so the UI can show Approve / Reject.
# 1. Walk each interrupt until one has an action.
# 2. Return None when nothing is waiting.
def first_action_from_interrupts(result) -> dict | None:
    # [first_action_from_interrupts-1] Walk each interrupt until one has an action.
    for interrupt in result.interrupts:
        actions = action_requests_from_value(interrupt.value or {})
        if actions:
            return actions[0]

    # [first_action_from_interrupts-2] Return None when nothing is waiting.
    return None


# [approval_decision]
# Builds an approve or reject resume decision.
# 1. Approve when the user said yes.
# 2. Otherwise reject with a short message for the agent.
def approval_decision(approved: bool) -> dict:
    # [approval_decision-1] Approve when the user said yes.
    if approved:
        return {"type": "approve"}

    # [approval_decision-2] Otherwise reject with a short message for the agent.
    return {"type": "reject", "message": REJECT_MESSAGE}


# [decision_from_answer]
# Turns a y/n answer into an approve or reject decision.
def decision_from_answer(answer: str) -> dict:
    return approval_decision(answer in {"y", "yes"})


# [build_agent]
# Builds the Groq agent with tools, saved state, and delete approval.
# 1. Create the Groq chat model.
# 2. Wrap it in create_agent with tools, checkpointer, and HITL middleware.
def build_agent(api_key: str, checkpointer):
    # [build_agent-1] Create the Groq chat model.
    model = ChatOpenAI(
        model="openai/gpt-oss-20b",
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    # [build_agent-2] Wrap it in create_agent with tools, checkpointer, and HITL middleware.
    return create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[HumanInTheLoopMiddleware(interrupt_on=HITL_INTERRUPT_ON)],
    )


# [tool_lines_from_messages]
# Builds display lines for new tool results after seen_count.
# 1. Walk messages that appeared after the last shown count.
# 2. Format each tool result for Streamlit or the terminal.
def tool_lines_from_messages(
    messages: list,
    seen_count: int,
    *,
    markdown: bool = True,
) -> list[str]:
    tool_lines = []

    # [tool_lines_from_messages-1] Walk messages that appeared after the last shown count.
    for message in messages[seen_count:]:
        if not isinstance(message, ToolMessage):
            continue
        tool_name = message.name or "tool"

        # [tool_lines_from_messages-2] Format each tool result for Streamlit or the terminal.
        if markdown:
            tool_lines.append(f"Tool `{tool_name}`: {message.content}")
        else:
            tool_lines.append(f"[Tool] {tool_name}: {message.content}")
    return tool_lines


# [reply_text_from_messages]
# Reads the final assistant reply as plain text.
# 1. Return a fallback when no messages came back.
# 2. Return the last message content as text.
def reply_text_from_messages(messages: list) -> str:
    # [reply_text_from_messages-1] Return a fallback when no messages came back.
    if not messages:
        return "No reply came back."

    # [reply_text_from_messages-2] Return the last message content as text.
    reply = messages[-1].content
    if not isinstance(reply, str):
        return str(reply)
    return reply
