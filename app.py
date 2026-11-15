# [imports]
# Loads Streamlit, the env file helper, chat messages, shared tools, and the agent builder.
import os

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from tools import TOOLS

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Use get_current_time for the current date or time. "
    "Use calculate for math. "
    "Answer in 2-3 sentences."
)

# [load_dotenv]
# Loads the API key from the .env file so the field can start with it.
load_dotenv()

# [page]
# Names the browser tab and shows the page heading.
# 1. Set the browser tab title.
# 2. Show the page heading.
# [page-1] Set the browser tab title.
st.set_page_config(page_title="LangChain Chatbot")

# [page-2] Show the page heading.
st.title("LangChain Chatbot")

# [messages]
# Keeps the thread in memory so a rerun does not wipe the chat.
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    # [api_key_field]
    # Reads the Groq key without using a password field, so Chrome does not offer to save it.
    # 1. Read the key, starting from the .env value.
    # 2. Let the user reveal the key.
    # 3. Mask the characters unless the key is revealed.
    # [api_key_field-1] Read the key, starting from the .env value.
    api_key = st.text_input(
        "Groq API key",
        value=os.getenv("GROQ_API_KEY", ""),
        key="groq_api_key",
        autocomplete="off",
    )

    # [api_key_field-2] Let the user reveal the key.
    show_key = st.checkbox("Show API key")

    # [api_key_field-3] Mask the characters unless the key is revealed.
    st.markdown(
        """
        <style>
        .st-key-groq_api_key input {
            -webkit-text-security: %s;
        }
        </style>
        """
        % ("none" if show_key else "disc"),
        unsafe_allow_html=True,
    )

# [show_history]
# Redraws the saved messages so they stay on the page after each reply.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# [send_message]
# Adds the user's message to the chat and shows the agent's reply.
# 1. Stop when there is no new message.
# 2. Stop if the API key is missing.
# 3. Show the user's message in the thread.
# 4. Build the conversation from saved messages (system prompt is on the agent).
# 5. Create the Groq model and the agent with tools.
# 6. Ask the agent for a reply (it may call tools first).
# 7. Stop and show the error if the call fails.
# 8. Show any tool results from this turn.
# 9. Turn a non-text reply into text.
# 10. Show the reply.
# 11. Keep the reply in the thread.
# [send_message-1] Stop when there is no new message.
prompt = st.chat_input("Message")
if not prompt:
    st.stop()

# [send_message-2] Stop if the API key is missing.
key = api_key.strip()
if not key:
    st.error("Enter a Groq API key in the sidebar before sending.")
    st.stop()

# [send_message-3] Show the user's message in the thread.
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.markdown(prompt)

# [send_message-4] Build the conversation from saved messages (system prompt is on the agent).
history = []
for message in st.session_state.messages:
    if message["role"] == "user":
        history.append(HumanMessage(content=message["content"]))
    else:
        history.append(AIMessage(content=message["content"]))

# [send_message-5] Create the Groq model and the agent with tools.
model = ChatOpenAI(
    model="openai/gpt-oss-20b",
    api_key=key,
    base_url="https://api.groq.com/openai/v1",
)
agent = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)

with st.chat_message("assistant"):
    try:
        # [send_message-6] Ask the agent for a reply (it may call tools first).
        prior_len = len(history)
        result = agent.invoke({"messages": history})
        result_messages = list(result["messages"])
        reply = result_messages[-1].content
    except Exception as error:
        # [send_message-7] Stop and show the error if the call fails.
        st.error(str(error))
        st.stop()

    # [send_message-8] Show any tool results from this turn.
    for message in result_messages[prior_len:]:
        if isinstance(message, ToolMessage):
            tool_name = message.name or "tool"
            st.caption(f"Tool `{tool_name}`: {message.content}")

    # [send_message-9] Turn a non-text reply into text.
    if not isinstance(reply, str):
        reply = str(reply)

    # [send_message-10] Show the reply.
    st.markdown(reply)

# [send_message-11] Keep the reply in the thread.
st.session_state.messages.append({"role": "assistant", "content": reply})
