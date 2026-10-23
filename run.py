#!/usr/bin/env python3
"""ZentroMart VoiceOps agent: one command for everything.

Batch voice run (what the organisers use):
    python run.py --input path/to/audio_folder --output outputs/
    -> for every audio file: Gnani STT -> agent (RAG + tools) -> Gnani TTS
       saves <name>_reply.wav, <name>.json (transcript, reply text, document chunks used, tool calls)
       plus results.csv and results.jsonl for the whole folder.

    Files are independent single-turn calls by default. Use --conversation to treat files that share a
    prefix (e.g. call1_01.wav, call1_02.wav) as turns of the same call, in filename order.

Text modes (no audio needed):
    python run.py --text "Can I return a swimsuit?"
    python run.py --chat                # interactive multi-turn chat in the terminal
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm", ".mp4", ".opus"}


def group_files(files: list[Path], conversation: bool) -> "OrderedDict[str, list[Path]]":
    groups: "OrderedDict[str, list[Path]]" = OrderedDict()
    for f in files:
        key = f.stem
        if conversation:
            m = re.match(r"^(.*?)[_\-\s]?(?:turn|t)?[_\-\s]?(\d+)$", f.stem, re.I)
            key = m.group(1) if m and m.group(1) else f.stem
        groups.setdefault(key, []).append(f)
    return groups


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def batch(input_dir: Path, output_dir: Path, conversation: bool):
    from zentro.agent import Agent
    from zentro.voice import voice_turn

    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in AUDIO_EXT], key=natural_key)
    if not files:
        sys.exit(f"No audio files found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    jsonl = open(output_dir / "results.jsonl", "w", encoding="utf-8")
    groups = group_files(files, conversation)
    print(f"Processing {len(files)} audio file(s) in {len(groups)} call(s) -> {output_dir}")
    for key, group in groups.items():
        agent = Agent(mode="agent", channel="voice")
        for turn, f in enumerate(group, 1):
            t0 = time.time()
            out_wav = output_dir / f"{f.stem}_reply.wav"
            try:
                r = voice_turn(f, agent, out_wav)
                res = r["result"]
                record = {
                    "file": f.name, "call": key, "turn": turn,
                    "transcript": r["heard"], "stt_transcripts": r["stt"].get("raw", r["stt"]["transcripts"]),
                    "reply_text": r["reply"], "reply_spoken_text": r["spoken"], "reply_audio": out_wav.name,
                    "decision": res.decision if res else "clarify",
                    "sources_used": res.sources if res else [],
                    "chunks_used": [h for h in (res.retrieved if res else []) if h["id"] in (res.sources if res else [])],
                    "chunks_retrieved": res.retrieved if res else [],
                    "tool_calls": res.tool_calls if res else [],
                    "seconds": round(time.time() - t0, 2), "error": "",
                }
                # sources cited from the always-on agent guides are not in chunks_retrieved: add their text
                if res:
                    from zentro.agent import get_retriever
                    known = {h["id"] for h in record["chunks_used"]}
                    for sid in res.sources:
                        if sid not in known and sid in get_retriever().by_id:
                            c = get_retriever().by_id[sid]
                            record["chunks_used"].append({"id": c.id, "title": c.title, "doc": c.doc_id,
                                                          "text": c.text})
            except Exception as e:  # never stop the batch on one bad file
                record = {"file": f.name, "call": key, "turn": turn, "transcript": "", "reply_text": "",
                          "reply_audio": "", "decision": "", "sources_used": [], "chunks_used": [],
                          "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t0, 2)}
            (output_dir / f"{f.stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
            jsonl.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            jsonl.flush()
            rows.append(record)
            status = "ERROR " + record["error"] if record["error"] else record["reply_text"][:100]
            print(f"[{len(rows)}/{len(files)}] {f.name}\n   heard: {record['transcript'][:100]}\n   reply: {status}"
                  f"\n   sources: {', '.join(record['sources_used'])}")
    jsonl.close()
    with open(output_dir / "results.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "call", "turn", "transcript", "reply_text", "reply_audio", "decision",
                    "sources_used", "error"])
        for r in rows:
            w.writerow([r["file"], r["call"], r["turn"], r["transcript"], r["reply_text"], r["reply_audio"],
                        r["decision"], " ; ".join(r["sources_used"]), r["error"]])
    print(f"\nDone. Replies (text + audio) and chunks saved in {output_dir}/")


def text_once(q: str):
    from zentro.agent import Agent

    r = Agent().respond(q)
    print(r.reply)
    print(f"\n[decision: {r.decision} | sources: {', '.join(r.sources) or 'none'}]")
    for c in r.tool_calls:
        print(f"[tool] {c['tool']} {c['args']}")


def chat():
    from zentro.agent import Agent

    agent = Agent()
    print("ZentroMart support chat. Type 'quit' to exit, 'reset' for a new conversation.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit"):
            break
        if q.lower() == "reset":
            agent = Agent()
            print("(new conversation)\n")
            continue
        if not q:
            continue
        r = agent.respond(q)
        print(f"agent> {r.reply}")
        print(f"       [decision: {r.decision} | sources: {', '.join(r.sources) or 'none'}"
              + "".join(f" | tool: {c['tool']}" for c in r.tool_calls) + "]\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", type=Path, help="folder of customer audio files")
    ap.add_argument("--output", "-o", type=Path, default=ROOT / "outputs", help="where to save replies")
    ap.add_argument("--conversation", action="store_true",
                    help="treat files sharing a prefix (call1_01, call1_02) as one multi-turn call")
    ap.add_argument("--stt-lang", help="force one Gnani STT language (e.g. en-IN, hi-IN) instead of dual en+hi")
    ap.add_argument("--text", help="ask one question in text")
    ap.add_argument("--chat", action="store_true", help="interactive text chat")
    a = ap.parse_args()
    if a.stt_lang:
        import os

        os.environ["STT_MODE"] = a.stt_lang
        from zentro import config

        config.STT_MODE = a.stt_lang
    if a.input:
        batch(a.input, a.output, a.conversation)
    elif a.text:
        text_once(a.text)
    elif a.chat:
        chat()
    else:
        ap.print_help()
