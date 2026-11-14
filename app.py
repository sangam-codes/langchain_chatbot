# [imports]
# Loads Streamlit, the env file helper, and the pieces that talk to Groq.
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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
# Adds the user's message to the chat and shows the model's reply.
# 1. Stop when there is no new message.
# 2. Stop if the API key is missing.
# 3. Show the user's message in the thread.
# 4. Build the conversation, including the system instruction.
# 5. Ask Groq for a reply.
# 6. Stop and show the error if the call fails.
# 7. Turn a non-text reply into text.
# 8. Show the reply.
# 9. Keep the reply in the thread.
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

# [send_message-4] Build the conversation, including the system instruction.
history = [SystemMessage(content="You are a helpful assistant. Answer in 2-3 sentences.")]
for message in st.session_state.messages:
    if message["role"] == "user":
        history.append(HumanMessage(content=message["content"]))
    else:
        history.append(AIMessage(content=message["content"]))

# [send_message-5] Ask Groq for a reply.
model = ChatOpenAI(
    model="openai/gpt-oss-20b",
    api_key=key,
    base_url="https://api.groq.com/openai/v1",
)

with st.chat_message("assistant"):
    try:
        reply = model.invoke(history).content
    except Exception as error:
        # [send_message-6] Stop and show the error if the call fails.
        st.error(str(error))
        st.stop()

    # [send_message-7] Turn a non-text reply into text.
    if not isinstance(reply, str):
        reply = str(reply)

    # [send_message-8] Show the reply.
    st.markdown(reply)

# [send_message-9] Keep the reply in the thread.
st.session_state.messages.append({"role": "assistant", "content": reply})
