# LangChain Chatbot

A small chat app built with Streamlit and LangChain. It talks to Groq for replies.

This repo is also for learning how AI agents work. The goal is to practice the important agent patterns with **free, small demos** (Groq free plan, local files, terminal) — not to build a big product first. Start with tools, not deployment. The detailed learning checklist is kept outside this repo.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Put a Groq API key in the sidebar, type in the box at the bottom, and send. **Show API key** sits under the key field and shows what you typed.

Stop the app with Ctrl+C.

## API key

You can paste the key into the sidebar each time, or put it in a `.env` file in this folder:

```bash
GROQ_API_KEY=your_key_here
```

The field starts with that value. `.env` is kept out of git.

The app uses Groq’s API with the model `openai/gpt-oss-20b`. The chat history stays until you refresh the page (tool captions stay with each reply). Streamlit and the terminal share helpers in `agent_common.py`, tools in `tools.py`, and lasting facts in `memory_store.py` (written to `.memory.json`, gitignored). Try “my hobby is …” and “my favorite color is …”, then restart and ask what you remember. If the agent wants to delete a note, the page shows **Approve** / **Reject** first. Reject once, then ask to delete again → approval is requested again.

## Terminal version

`chatbot.py` is the terminal agent: Groq, chat history, tools, lasting memory, and LangGraph approval for `delete_note`. Put `GROQ_API_KEY` in `.env`, then run:

```bash
python chatbot.py
```

Type `quit` to exit. Tool calls print as `[Tool]`. Memory tools write `.memory.json`. If the agent wants to delete a note, it asks `Approve this tool? [y/n]` first. Reject once, then ask to delete again → approval is requested again.
