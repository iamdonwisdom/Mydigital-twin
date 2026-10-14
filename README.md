# Chi's Digital Twin

An AI chatbot that answers questions about me, using a private knowledge base, tools (time, calculator, web search, leave-a-message) and safety guardrails.

**Try it live:** https://chi-digital-twin.onrender.com
(The free server sleeps when idle, so the first load can take about a minute.)

## How it works
- LLM: Groq (`openai/gpt-oss-20b`)
- Backend: FastAPI (`server.py`) and the twin logic (`main.py`)
- Web chat page: `index.html`
- Safety: input and output filters, rate limits, red-team and eval scripts (`redteam.py`, `eval.py`)

## Run it locally
1. `pip install -r requirements-full.txt`
2. Create a `.env` file with `GROQ_API_KEY=your_key`
3. Create a `knowledge.txt` file with facts about yourself, separated by blank lines
4. `uvicorn server:app --reload`

Built for the ATF AI Challenge, "Building with LLMs".