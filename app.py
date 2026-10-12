import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"

def ask(messages):
    rules = {"role": "system",
             "content": "Reply in plain text only. No LaTeX, no markdown symbols."}
    r = client.chat.completions.create(
        model=MODEL, messages=[rules] + messages, reasoning_effort="low"
    )
    return r.choices[0].message.content.strip()

# ---------- 1.3.2 Few-shot structured output ----------
print("=== FEW-SHOT: message classifier ===")

SYSTEM = """You classify messages sent to Chi's digital twin.
Reply with JSON only, no extra text, in this format:
{"category": "question|collaboration|spam", "summary": "short summary"}"""

examples = [
    {"role": "user", "content": "What are you studying?"},
    {"role": "assistant", "content": '{"category": "question", "summary": "asks what Chi studies"}'},
    {"role": "user", "content": "Want to build a security tool together?"},
    {"role": "assistant", "content": '{"category": "collaboration", "summary": "invites Chi to build a security tool"}'},
    {"role": "user", "content": "WIN $1000 NOW click this link!!!"},
    {"role": "assistant", "content": '{"category": "spam", "summary": "prize scam message"}'},
]

test = "Hi, can we team up for a hackathon next month?"
raw = ask([{"role": "system", "content": SYSTEM}] + examples +
          [{"role": "user", "content": test}])
print("Raw reply:", raw)

try:
    data = json.loads(raw)
    print("Category:", data["category"])
    print("Summary:", data["summary"])
except json.JSONDecodeError:
    print("Reply was not valid JSON")

# ---------- 1.3.3 Chain-of-thought comparison ----------
print("\n=== CHAIN OF THOUGHT ===")
question = ("A shop sells pens at 3 for 500 naira. I buy 14 pens and "
            "pay with 5000 naira. How much change do I get?")

direct = ask([{"role": "user",
               "content": question + " Give only the final number, no explanation."}])
steps = ask([{"role": "user",
              "content": question + " Think step by step, then give the final answer."}])

print("Direct answer:\n", direct)
print("\nStep-by-step answer:\n", steps)