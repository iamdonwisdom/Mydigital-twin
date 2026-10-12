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
TOP_K = 3
MIN_SCORE = 0.40
MAX_HISTORY_TOKENS = 3000  # lower this (e.g. 400) to test summarizing quickly

# ---------- knowledge base ----------
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

# ---------- helpers ----------
def clean(text):
    text = re.sub(r"【[^】]*】", "", text)
    return text.replace("**", "").strip()

def estimate_tokens(messages):
    # rough rule: 1 token is about 4 characters
    return sum(len(m["content"]) for m in messages) // 4

def build_system_prompt(facts, summary):
    facts_text = "\n\n".join(facts) if facts else "(no matching facts found)"
    summary_text = summary if summary else "(conversation just started)"
    return f"""You are the digital twin of Chi Dozie.
Speak in the first person, as Chi, in a friendly and clear way.
Today's date is {date.today().strftime('%A, %d %B %Y')}.

Facts about Chi relevant to this question:
{facts_text}

Summary of the earlier conversation:
{summary_text}

Rules:
- For questions about Chi, use ONLY the facts above. If they don't cover it, say you don't know. Never invent personal details.
- For news, current events, prices or anything recent, use web search.
- Answer in plain text only. No tables, no markdown symbols like ** or |.
- Keep answers short and clear."""

def summarize(old_messages, previous_summary):
    text = "\n".join(f'{m["role"]}: {m["content"]}' for m in old_messages)
    prompt = (f"Previous summary: {previous_summary or 'none'}\n\n"
              f"New conversation to add:\n{text}\n\n"
              "Write a short plain-text summary (max 120 words) of what the "
              "user asked and what matters for the rest of the chat.")
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="low",
        max_completion_tokens=1000,
    )
    return clean(r.choices[0].message.content or "")

# ---------- chat loop ----------
history = []
summary = ""
print(f"Twin loaded with {len(chunks)} facts. Type 'clear' to reset, 'quit' to stop.\n")

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break
    if not user_input:
        continue
    if user_input.lower() == "quit":
        break
    if user_input.lower() == "clear":
        history, summary = [], ""
        print("Chat cleared.\n")
        continue

    facts = retrieve(user_input)
    history.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": build_system_prompt(facts, summary)}] + history

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

    # context management: summarize the older half when history gets too big
    if estimate_tokens(history) > MAX_HISTORY_TOKENS and len(history) >= 4:
        cut = (len(history) // 2) // 2 * 2  # cut on a user/assistant boundary
        old, history = history[:cut], history[cut:]
        try:
            summary = summarize(old, summary)
            print(f"[memory compacted, {len(old)} messages summarized]\n")
        except Exception as e:
            history = old + history  # keep everything if summarizing fails
            print("Could not summarize:", e, "\n")