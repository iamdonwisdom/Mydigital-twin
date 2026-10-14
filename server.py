import os
import re
import threading
import time
from collections import OrderedDict
from datetime import date

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import main

main.SHOW_SOURCES = False
main.SHOW_TOOLS = False
main.MAX_SAVES_PER_SESSION = 20   # cap on saved messages per server run

MAX_SESSIONS = 200       # chats kept in memory
MAX_MSGS_PER_MIN = 8     # per visitor address
DAILY_CAP = 300          # total messages per day, protects your Groq limit
SESSION_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")

app = FastAPI()
state_lock = threading.Lock()
gate = threading.Semaphore(2)    # at most 2 AI requests at once
sessions = OrderedDict()
ip_hits = {}
daily = {"day": date.today(), "count": 0}


class ChatIn(BaseModel):
    session_id: str = Field(max_length=64)
    message: str = Field(max_length=5000)


def get_session(sid):
    if sid not in sessions:
        sessions[sid] = {"history": [], "summary": ""}
        while len(sessions) > MAX_SESSIONS:
            sessions.popitem(last=False)
    sessions.move_to_end(sid)
    return sessions[sid]


def client_ip(request):
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}

from fastapi.responses import Response


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.post("/chat")
def chat(body: ChatIn, request: Request):
    if not SESSION_RE.match(body.session_id):
        return {"reply": "Invalid session. Please refresh the page."}
    message = body.message.strip()
    if not message:
        return {"reply": "Please type a message."}

    ip = client_ip(request)
    now = time.time()
    with state_lock:
        if daily["day"] != date.today():
            daily["day"], daily["count"] = date.today(), 0
        if daily["count"] >= DAILY_CAP:
            return {"reply": "I've reached my daily limit. Please come back tomorrow."}
        if len(ip_hits) > 5000:
            ip_hits.clear()
        hits = [t for t in ip_hits.get(ip, []) if now - t < 60]
        if len(hits) >= MAX_MSGS_PER_MIN:
            ip_hits[ip] = hits
            return {"reply": "You're sending messages very fast. Please wait a moment."}
        hits.append(now)
        ip_hits[ip] = hits
        daily["count"] += 1
        session = get_session(body.session_id)

    ok, reason = main.check_input(message)
    if not ok:
        main.log_security(reason, message)
        return {"reply": main.blocked_reply(reason)}

    try:
        with gate:
            reply, _ = main.ask_twin(message, session["history"], session["summary"])
    except Exception as e:
        if "429" in str(e):
            return {"reply": "I'm very busy right now. Please try again in a few minutes."}
        print("Error:", str(e)[:300])
        return {"reply": "Sorry, something went wrong. Please try again."}

    history = session["history"]
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})

    if main.estimate_tokens(history) > main.MAX_HISTORY_TOKENS and len(history) >= 4:
        cut = (len(history) // 2) // 2 * 2
        old, session["history"] = history[:cut], history[cut:]
        try:
            session["summary"] = main.summarize(old, session["summary"])
        except Exception:
            session["history"] = old + session["history"]

    return {"reply": reply}