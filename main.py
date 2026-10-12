import os
import re
import hashlib
import numpy as np
import chromadb
from datetime import date
from dotenv import load_dotenv
from groq import Groq
from fastembed import TextEmbedding

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"
TOP_K = 3
MIN_SCORE = 0.40
MAX_HISTORY_TOKENS = 3000  # lower to 400 to test summarizing

# ---------- knowledge base (stored in a vector database) ----------
embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
db = chromadb.PersistentClient(path="chroma_db")
HASH_FILE = os.path.join("chroma_db", "knowledge.hash")

with open("knowledge.txt", encoding="utf-8") as f:
    raw = f.read()
chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]
file_hash = hashlib.md5(raw.encode()).hexdigest()
old_hash = open(HASH_FILE).read() if os.path.exists(HASH_FILE) else ""

if old_hash != file_hash:
    # knowledge.txt is new or changed: rebuild the index
    try:
        db.delete_collection("twin_knowledge")
    except Exception:
        pass
    collection = db.create_collection(
        "twin_knowledge", metadata={"hnsw:space": "cosine"}
    )
    vecs = [v.tolist() for v in embedder.embed(chunks)]
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=vecs,
    )
    with open(HASH_FILE, "w") as f:
        f.write(file_hash)
    print("Knowledge indexed.")
else:
    collection = db.get_collection("twin_knowledge")


def retrieve(question):
    q = list(embedder.embed([question]))[0].tolist()
    res = collection.query(
        query_embeddings=[q],
        n_results=min(TOP_K, collection.count()),
    )
    facts = []
    for doc, dist in zip(res["documents"][0], res["distances"][0]):
        if 1 - dist >= MIN_SCORE:  # cosine distance -> similarity
            facts.append(doc)
    return facts


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