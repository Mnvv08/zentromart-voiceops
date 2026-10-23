"""Browser demo: talk to the agent with your microphone (Gnani STT -> agent -> Gnani TTS).

    python web/app.py      then open http://localhost:5050
"""
from __future__ import annotations

import sys
import tempfile
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zentro.agent import Agent, get_retriever  # noqa: E402
from zentro.voice import synthesize, voice_turn  # noqa: E402

app = Flask(__name__, static_folder=str(ROOT / "web" / "static"))
AUDIO_DIR = Path(tempfile.mkdtemp(prefix="zentro_audio_"))
SESSIONS: dict[str, Agent] = {}


def agent_for(sid: str) -> Agent:
    if sid not in SESSIONS:
        SESSIONS[sid] = Agent(mode="agent", channel="voice")
    return SESSIONS[sid]


def payload(res, heard: str, audio_name: str | None, reply: str | None = None):
    chunks = []
    if res:
        r = get_retriever()
        for sid in res.sources:
            c = r.by_id.get(sid)
            if c:
                chunks.append({"id": c.id, "title": c.title, "text": c.text[:500]})
    return {"heard": heard, "reply": reply if reply is not None else (res.reply if res else ""),
            "decision": res.decision if res else "clarify", "sources": chunks,
            "tools": [c["tool"] for c in res.tool_calls] if res else [],
            "audio": f"/audio/{audio_name}" if audio_name else None}


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.post("/api/voice")
def api_voice():
    sid = request.form.get("sid") or "default"
    f = request.files["audio"]
    inp = AUDIO_DIR / f"in_{uuid.uuid4().hex}.wav"
    f.save(inp)
    name = f"reply_{uuid.uuid4().hex}.wav"
    r = voice_turn(inp, agent_for(sid), AUDIO_DIR / name)
    return jsonify(payload(r["result"], r["heard"], name, reply=r["reply"]))


@app.post("/api/text")
def api_text():
    data = request.get_json(force=True)
    sid = data.get("sid") or "default"
    res = agent_for(sid).respond(data.get("text", ""))
    name = f"reply_{uuid.uuid4().hex}.wav"
    audio = None
    if data.get("speak", True):
        try:
            synthesize(res.spoken, AUDIO_DIR / name)
            audio = name
        except Exception as e:  # text still works without Gnani
            print("TTS error:", e)
    return jsonify(payload(res, data.get("text", ""), audio))


@app.post("/api/reset")
def api_reset():
    sid = (request.get_json(force=True) or {}).get("sid") or "default"
    SESSIONS.pop(sid, None)
    return jsonify({"ok": True})


@app.get("/audio/<name>")
def audio(name: str):
    return send_file(AUDIO_DIR / Path(name).name, mimetype="audio/wav")


if __name__ == "__main__":
    get_retriever()  # build the index before the first request
    print("Open http://localhost:5050 in Chrome or Safari")
    app.run(host="127.0.0.1", port=5050, debug=False)
