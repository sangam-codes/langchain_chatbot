# [imports]
# Loads Streamlit, env helper, shared agent helpers, and HITL resume types.
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent_common import (
    approval_decision,
    build_agent as build_hitl_agent,
    first_action_from_interrupts,
    has_interrupts,
    messages_from_result,
    reply_text_from_messages,
    tool_lines_from_messages,
)
from memory_store import get_facts


# [build_agent]
# Builds the Groq agent using the session checkpointer so pauses can resume.
def build_agent(api_key: str):
    return build_hitl_agent(api_key, st.session_state.checkpointer)


# [finish_turn]
# Collects new tool captions and the final assistant reply from a finished result.
# 1. Collect new tool captions from this turn.
# 2. Remember how many messages have been shown.
# 3. Return the captions and the final assistant reply.
def finish_turn(result) -> tuple[list[str], str]:
    messages = messages_from_result(result)
    prior = st.session_state.seen_message_count

    # [finish_turn-1] Collect new tool captions from this turn.
    tool_lines = tool_lines_from_messages(messages, prior)

    # [finish_turn-2] Remember how many messages have been shown.
    st.session_state.seen_message_count = len(messages)

    # [finish_turn-3] Return the captions and the final assistant reply.
    return tool_lines, reply_text_from_messages(messages)


# [init_session]
# Keeps chat history, thread id, checkpointer, and pending approval across reruns.
# 1. Start an empty chat history.
# 2. Create one thread id for this browser session.
# 3. Keep one checkpointer so paused graphs can resume.
# 4. Track whether a tool is waiting for Approve / Reject.
# 5. Track how many agent messages have already been shown.
def init_session() -> None:
    # [init_session-1] Start an empty chat history.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # [init_session-2] Create one thread id for this browser session.
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # [init_session-3] Keep one checkpointer so paused graphs can resume.
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = InMemorySaver()

    # [init_session-4] Track whether a tool is waiting for Approve / Reject.
    if "pending_approval" not in st.session_state:
        st.session_state.pending_approval = None

    # [init_session-5] Track how many agent messages have already been shown.
    if "seen_message_count" not in st.session_state:
        st.session_state.seen_message_count = 0


# [render_sidebar]
# Reads the Groq key without using a password field, so Chrome does not offer to save it.
# 1. Read the key, starting from the .env value.
# 2. Let the user reveal the key.
# 3. Mask the characters unless the key is revealed.
def render_sidebar() -> str:
    with st.sidebar:
        # [render_sidebar-1] Read the key, starting from the .env value.
        api_key = st.text_input(
            "Groq API key",
            value=os.getenv("GROQ_API_KEY", ""),
            key="groq_api_key",
            autocomplete="off",
        )

        # [render_sidebar-2] Let the user reveal the key.
        show_key = st.checkbox("Show API key")

        # [render_sidebar-3] Mask the characters unless the key is revealed.
        text_security = "none" if show_key else "disc"
        st.markdown(
            f"""
            <style>
            .st-key-groq_api_key input {{
                -webkit-text-security: {text_security};
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Risky tools like delete_note ask for Approve / Reject before they run."
        )
        st.caption("Lasting facts are stored in `.memory.json` and survive restarts.")
        st.caption(f"Saved memory: {get_facts()}")
        st.caption(
            "Ask about the cafe, library, or park notes in `docs/` "
            "(TF-IDF vector search)."
        )
    return api_key


# [store_pending]
# Saves the paused tool name and args until the user clicks Approve or Reject.
def store_pending(action: dict) -> None:
    st.session_state.pending_approval = {
        "name": action.get("name", "tool"),
        "args": action.get("args", {}),
    }


# [show_assistant_reply]
# Shows tool captions and the assistant text in the chat bubble.
# 1. Show each tool caption from this turn.
# 2. Show the assistant reply text.
def show_assistant_reply(tool_lines: list[str], reply: str) -> None:
    # [show_assistant_reply-1] Show each tool caption from this turn.
    for line in tool_lines:
        st.caption(line)

    # [show_assistant_reply-2] Show the assistant reply text.
    st.markdown(reply)


# [save_assistant_message]
# Keeps the reply and tool captions in history so they survive Streamlit reruns.
def save_assistant_message(tool_lines: list[str], reply: str) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "tool_lines": list(tool_lines),
        }
    )


# [render_chat_history]
# Redraws the saved messages so they stay on the page after each reply.
# 1. Draw each saved chat bubble.
# 2. Restore any tool captions saved with that reply.
def render_chat_history() -> None:
    # [render_chat_history-1] Draw each saved chat bubble.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # [render_chat_history-2] Restore any tool captions saved with that reply.
            for line in message.get("tool_lines", []):
                st.caption(line)
            st.markdown(message["content"])


# [require_api_key]
# Stops the page when the Groq key is missing.
# 1. Trim the key from the sidebar.
# 2. Stop with an error when it is empty.
# 3. Return the ready key.
def require_api_key(api_key: str, error_message: str) -> str:
    # [require_api_key-1] Trim the key from the sidebar.
    key = api_key.strip()

    # [require_api_key-2] Stop with an error when it is empty.
    if not key:
        st.error(error_message)
        st.stop()

    # [require_api_key-3] Return the ready key.
    return key


# [pause_for_approval]
# Stores the paused tool and reruns so Approve / Reject can show.
# 1. Stop if the interrupt has no tool details.
# 2. Save the paused tool in session state.
# 3. Optionally show a waiting note.
# 4. Rerun so the approval buttons appear.
def pause_for_approval(result, info_message: bool = False) -> None:
    # [pause_for_approval-1] Stop if the interrupt has no tool details.
    action = first_action_from_interrupts(result)
    if action is None:
        st.error("The agent paused, but no tool details came back.")
        st.stop()

    # [pause_for_approval-2] Save the paused tool in session state.
    store_pending(action)

    # [pause_for_approval-3] Optionally show a waiting note.
    if info_message:
        st.info(
            f"Waiting for approval: `{action.get('name')}` — {action.get('args')}"
        )

    # [pause_for_approval-4] Rerun so the approval buttons appear.
    st.rerun()


# [complete_assistant_turn]
# Shows the finished reply and keeps it in the chat history.
# 1. Collect tool captions and the reply text.
# 2. Show them in the assistant bubble.
# 3. Keep the reply and captions in the saved chat history.
def complete_assistant_turn(result) -> None:
    # [complete_assistant_turn-1] Collect tool captions and the reply text.
    tool_lines, reply = finish_turn(result)

    # [complete_assistant_turn-2] Show them in the assistant bubble.
    show_assistant_reply(tool_lines, reply)

    # [complete_assistant_turn-3] Keep the reply and captions in the saved chat history.
    save_assistant_message(tool_lines, reply)


# [handle_pending_approval]
# Shows Approve / Reject when LangGraph paused a risky tool, then resumes the agent.
# 1. Stop here when there is no pending approval.
# 2. Show the paused tool details.
# 3. Wait until Approve or Reject is clicked.
# 4. Stop if the API key is missing.
# 5. Resume the paused agent with the user's decision.
# 6. If another tool needs approval, store it and rerun.
# 7. Show tool results and the final reply, then clear pending state.
def handle_pending_approval(api_key: str, config: dict) -> None:
    # [handle_pending_approval-1] Stop here when there is no pending approval.
    if not st.session_state.pending_approval:
        return

    # [handle_pending_approval-2] Show the paused tool details.
    pending = st.session_state.pending_approval
    st.warning(f"Approval needed: `{pending['name']}` — {pending['args']}")

    # [handle_pending_approval-3] Wait until Approve or Reject is clicked.
    col_approve, col_reject = st.columns(2)
    approve = col_approve.button("Approve", type="primary")
    reject = col_reject.button("Reject")
    if not (approve or reject):
        st.stop()

    # [handle_pending_approval-4] Stop if the API key is missing.
    key = require_api_key(
        api_key,
        "Enter a Groq API key in the sidebar before approving.",
    )

    agent = build_agent(key)
    decision = approval_decision(approved=approve)

    with st.chat_message("assistant"):
        try:
            # [handle_pending_approval-5] Resume the paused agent with the user's decision.
            result = agent.invoke(
                Command(resume={"decisions": [decision]}),
                config=config,
                version="v2",
            )
        except Exception as error:
            st.error(str(error))
            st.stop()

        # [handle_pending_approval-6] If another tool needs approval, store it and rerun.
        if has_interrupts(result):
            pause_for_approval(result)
            return

        # [handle_pending_approval-7] Show tool results and the final reply, then clear pending state.
        tool_lines, reply = finish_turn(result)
        show_assistant_reply(tool_lines, reply)

    st.session_state.pending_approval = None
    save_assistant_message(tool_lines, reply)
    st.rerun()


# [handle_new_message]
# Adds the user's message to the chat and shows the agent's reply (or asks for approval).
# 1. Stop when there is no new message.
# 2. Stop if the API key is missing.
# 3. Show the user's message in the thread.
# 4. Ask the agent for a reply (checkpointer keeps earlier turns).
# 5. Stop and show the error if the call fails.
# 6. If a tool needs approval, store it and wait for the next rerun.
# 7. Show any tool results and the reply, then keep it in the thread.
def handle_new_message(api_key: str, config: dict) -> None:
    # [handle_new_message-1] Stop when there is no new message.
    prompt = st.chat_input("Message")
    if not prompt:
        st.stop()

    # [handle_new_message-2] Stop if the API key is missing.
    key = require_api_key(
        api_key,
        "Enter a Groq API key in the sidebar before sending.",
    )

    # [handle_new_message-3] Show the user's message in the thread.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = build_agent(key)

    with st.chat_message("assistant"):
        try:
            # [handle_new_message-4] Ask the agent for a reply (checkpointer keeps earlier turns).
            result = agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
                version="v2",
            )
        except Exception as error:
            # [handle_new_message-5] Stop and show the error if the call fails.
            st.error(str(error))
            st.stop()

        # [handle_new_message-6] If a tool needs approval, store it and wait for the next rerun.
        if has_interrupts(result):
            pause_for_approval(result, info_message=True)
            return

        # [handle_new_message-7] Show any tool results and the reply, then keep it in the thread.
        complete_assistant_turn(result)


# [load_dotenv]
# Loads the API key from the .env file so the field can start with it.
load_dotenv()

# [page]
# Names the browser tab and shows the page heading.
# 1. Set the browser tab title.
# 2. Show the page heading.
# 3. Init session state and the sidebar key field.
# 4. Redraw saved chat history.
# 5. Handle pending approval, or a new chat message.
# [page-1] Set the browser tab title.
st.set_page_config(page_title="LangChain Chatbot")

# [page-2] Show the page heading.
st.title("LangChain Chatbot")

# [page-3] Init session state and the sidebar key field.
init_session()
api_key = render_sidebar()
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# [page-4] Redraw saved chat history.
render_chat_history()

# [page-5] Handle pending approval, or a new chat message.
handle_pending_approval(api_key, config)
handle_new_message(api_key, config)
