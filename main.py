import os
import re
import numpy as np
from datetime import date
from dotenv import load_dotenv
from groq import Groq
from fastembed import TextEmbedding

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"
MAX_MESSAGES = 20
TOP_K = 3
MIN_SCORE = 0.40  # ignore facts that are not related to the question

# ---------- load and embed the knowledge base ----------
embedder = TextEmbedding("BAAI/bge-small-en-v1.5")

with open("knowledge.txt", encoding="utf-8") as f:
    chunks = [c.strip() for c in f.read().split("\n\n") if c.strip()]

chunk_vecs = np.array(list(embedder.embed(chunks)))
chunk_norms = np.linalg.norm(chunk_vecs, axis=1)

def retrieve(question):
    q = np.array(list(embedder.embed([question])))[0]
    scores = chunk_vecs @ q / (chunk_norms * np.linalg.norm(q))
    best = np.argsort(scores)[::-1][:TOP_K]
    return [chunks[i] for i in best if scores[i] >= MIN_SCORE]

# ---------- prompts ----------
def build_system_prompt(facts):
    facts_text = "\n\n".join(facts) if facts else "(no matching facts found)"
    return f"""You are the digital twin of Chi Dozie.
Speak in the first person, as Chi, in a friendly and clear way.
Today's date is {date.today().strftime('%A, %d %B %Y')}.

Facts about Chi relevant to this question:
{facts_text}

Rules:
- For questions about Chi, use ONLY the facts above. If they don't cover it, say you don't know. Never invent personal details.
- For news, current events, prices or anything recent, use web search.
- Answer in plain text only. No tables, no markdown symbols like ** or |.
- Keep answers short and clear."""

def clean(text):
    text = re.sub(r"【[^】]*】", "", text)
    return text.replace("**", "").strip()

# ---------- chat loop ----------
history = []  # user and assistant messages only
print(f"Twin loaded with {len(chunks)} facts. Type 'clear' to reset, 'quit' to stop.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "quit":
        break
    if user_input.lower() == "clear":
        history = []
        print("Chat cleared.\n")
        continue

    facts = retrieve(user_input)
    history.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": build_system_prompt(facts)}] + history

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[{"type": "browser_search"}],
            tool_choice="auto",
            reasoning_effort="low",
            max_completion_tokens=2048,
        )
        reply = clean(response.choices[0].message.content or "")
    except Exception as e:
        print("Error:", e, "\n")
        history.pop()
        continue

    history.append({"role": "assistant", "content": reply})
    print("Twin:", reply, "\n")
    history = history[-MAX_MESSAGES:]