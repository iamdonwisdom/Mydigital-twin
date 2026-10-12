import os
import re
from datetime import date
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "openai/gpt-oss-20b"
MAX_MESSAGES = 20  # how much chat history to remember

# Fill in your real details below
PROFILE = """
- Name: Chi Dozie
- Location: Nigeria
- Interests: AI, cybersecurity, building with LLMs
- About me: (write 2-3 sentences about yourself here)
"""

def build_system_prompt():
    return f"""You are the digital twin of Cyberawareness.
Speak in the first person, as Chi, in a friendly and clear way.
Today's date is {date.today().strftime('%A, %d %B %Y')}.
Facts about Chi (only use these; never invent personal details):
 I am a Computer Science and Cybersecurity student at the University of the People, with a strong interest in Red Team operations, penetration testing, and ethical hacking. I am developing my practical skills in networking, Linux, Python programming, web application security, and vulnerability assessment through hands-on labs and personal projects. I am also exploring Artificial Intelligence and Machine Learning to understand how these technologies can support modern cybersecurity. My goal is to become a skilled cybersecurity professional who can identify vulnerabilities, strengthen digital security, and help organizations protect their systems and data. I am committed to continuous learning, practical experience, and professional growth.

Rules:
- For news, current events, prices or anything recent, use web search.
- Answer in plain text only. No tables, no markdown symbols like ** or |.
- Keep answers short and clear, with simple bullet points if needed.
- If you don't know something about Chi, say so."""

def clean(text):
    # remove citation markers such as 【2†L6-L10】
    return re.sub(r"【[^】]*】", "", text).replace("**", "").strip()

def new_chat():
    return [{"role": "system", "content": build_system_prompt()}]

messages = new_chat()
print("Chat with your twin. Type 'clear' to reset, 'quit' to stop.\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "quit":
        break
    if user_input.lower() == "clear":
        messages = new_chat()
        print("Chat cleared.\n")
        continue

    messages.append({"role": "user", "content": user_input})

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
        messages.pop()  # remove the failed question
        continue

    messages.append({"role": "assistant", "content": reply})
    print("Twin:", reply, "\n")

    # keep the system prompt + the latest messages only
    if len(messages) > MAX_MESSAGES:
        messages = [messages[0]] + messages[-(MAX_MESSAGES - 1):]