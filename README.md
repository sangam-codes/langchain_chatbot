# LangChain Chatbot

A small Streamlit chat page. It sends the conversation to Groq through LangChain's OpenAI-compatible client.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Put a Groq API key in the sidebar, type in the box at the bottom, and send. **Show API key** sits under the key field and reveals what you typed.

Stop the app with Ctrl+C.

## API key

You can paste the key into the sidebar each time, or put it in a `.env` file in this folder:

```bash
GROQ_API_KEY=your_key_here
```

The field starts with that value. `.env` is gitignored.

The app calls `https://api.groq.com/openai/v1` with the model `openai/gpt-oss-20b`. The thread stays in the browser session. A refresh clears it.

## Terminal script

`chatbot.py` is a separate one-shot script. It still calls OpenAI with `gpt-4o-mini` and expects `OPENAI_API_KEY` in `.env`.

```bash
python chatbot.py
```
