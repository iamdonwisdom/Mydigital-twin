# Chi's Digital Twin

An AI chatbot that answers questions about me, using a private knowledge base, tools (time, calculator, web search, leave-a-message) and safety guardrails. Built for the ATF AI Challenge, "Building with LLMs".

**Try it live:** https://chi-digital-twin.onrender.com
(The free server sleeps when idle, so the first load can take about a minute.)

## Screenshots

**A normal answer**



![A normal answer](screenshots/normal-answer.png)



**A blocked prompt-injection attempt**



![A blocked attack](screenshots/blocked-attack.png)



## How it works
- **LLM:** Groq (`openai/gpt-oss-20b`)
- **Backend:** FastAPI (`server.py`) and the twin logic (`main.py`)
- **Web chat page:** `index.html`
- **Knowledge base:** a private text file about me, kept out of GitHub (loaded from an environment setting on the server)
- **Tools:** current time, safe calculator, web search, leave-a-message

## Security
This project was built with security in mind, because the twin is public.
- **Input filter:** blocks prompt-injection phrases such as "ignore previous instructions", including simple letter-swap tricks, and limits message length.
- **Output filter:** hides emails and phone numbers in replies, and blocks any reply that leaks the system prompt (checked with a random marker created at every start).
- **Rate limits:** limits per session, per visitor address and per day, to stop abuse and protect the API quota.
- **Secrets:** API keys and the knowledge file live in environment settings and are never committed.
- **Red-team testing:** `redteam.py` runs 22 test prompts (20 attacks and 2 normal-request checks) against the twin, and `eval.py` runs 14 automated quality and safety checks.

## Run it locally
1. Install the packages: `pip install -r requirements-full.txt`
2. Create a `.env` file containing `GROQ_API_KEY=your_key`
3. Create a `knowledge.txt` file with facts about yourself. Separate each fact or topic with a blank line.
4. Chat in the terminal: `python main.py`
5. Or start the web version: `uvicorn server:app --reload`, then open http://127.0.0.1:8000

Optional settings: `TWIN_MODEL` changes the model. `USE_RAG=0` runs light mode, where all facts go into every prompt.

## Deployment
Deployed on Render's free plan in light mode (`USE_RAG=0`), with `GROQ_API_KEY` and `KNOWLEDGE_B64` stored as private environment settings.