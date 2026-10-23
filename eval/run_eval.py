"""Evaluate the agent on the practice sets.

  python eval/run_eval.py --step 1            # prompt-only baseline on practice/questions.csv
  python eval/run_eval.py --step 2            # RAG agent on practice/questions.csv
  python eval/run_eval.py --step 3            # multi-turn agent on practice/conversations.csv
  python eval/run_eval.py --step 2 --asr      # use the asr_text column (simulated speech-to-text input)

Results go to eval/results/: one JSONL with every reply and a summary JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from scoring import gold_ids, must_include_ok, must_not_ok, recall, split_list  # noqa: E402
from zentro import config  # noqa: E402
from zentro.agent import Agent  # noqa: E402
from zentro import llm  # noqa: E402

OUT = ROOT / "eval" / "results"
WORKERS = 3


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def tool_ok(expected: str, calls: list) -> bool:
    exp = expected.strip()
    if not exp:
        return not any(c["tool"] == "create_ticket" for c in calls)
    ok = True
    for part in split_list(exp) or [exp]:
        if part.startswith("create_ticket"):
            ok &= any(c["tool"] == "create_ticket" and c["result"].get("created") for c in calls)
        elif part.startswith("lookup_order"):
            want_v = "verification_value" in part
            vv = part.split("verification_value=")[-1].rstrip(")") if want_v else ""
            # logged values are masked (README: no secrets in logs) -> compare length only
            ok &= any(c["tool"].startswith("lookup_order") and
                      (("verification_value" in c["args"]) == want_v or c["tool"] != "lookup_order") and
                      (not want_v or len(c["args"].get("verification_value", "")) == len(vv)
                       or c["tool"] != "lookup_order")
                      for c in calls)
    return ok


def run_questions(step: int, limit: int | None, use_asr: bool, ids: set | None) -> dict:
    rows = read_csv(config.PRACTICE_DIR / "questions.csv")
    if ids:
        rows = [r for r in rows if r["id"] in ids]
    rows = rows[:limit] if limit else rows
    mode = "baseline" if step == 1 else "agent"
    tag = f"step{step}{'_asr' if use_asr else ''}"
    def one(r):
        agent = Agent(mode=mode, channel="voice" if use_asr else "chat")
        q = r["asr_text"] if use_asr else r["question"]
        try:
            res = agent.respond(q)
        except Exception as e:  # keep going; count as wrong
            print(f"  ! {r['id']} error: {e}")
            res = None
        reply = res.reply if res else ""
        acceptable = split_list(r["acceptable_decisions"]) or [r["expected_decision"]]
        dec_ok = bool(res) and res.decision in acceptable
        mi_ok, missing = must_include_ok(reply, r["must_include"])
        mn_ok, bad = must_not_ok(reply, r["must_not_include"])
        gold = gold_ids(r["gold_evidence"])
        row = {
            "id": r["id"], "language": r["language"], "question": q, "expected_decision": r["expected_decision"],
            "acceptable": acceptable, "decision": res.decision if res else "", "reply": reply,
            "decision_ok": dec_ok, "must_include_ok": mi_ok, "missing": missing, "must_not_ok": mn_ok,
            "forbidden_found": bad, "correct": dec_ok and mi_ok and mn_ok,
            "gold": gold, "sources": res.sources if res else [],
            "retrieved": [h["id"] for h in res.retrieved] if res else [],
            "recall_cited": recall(gold, res.sources if res else []),
            "recall_retrieved": recall(gold, [h["id"] for h in res.retrieved] if res else []),
            "tool_calls": res.tool_calls if res else [], "latency_s": res.latency_s if res else None,
            "model_used": llm.last_model_used,
        }
        with open(OUT / "partial_questions.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        flag = "OK " if row["correct"] else "XX "
        print(f"{flag}{r['id']} dec={row['decision']}/{r['expected_decision']} "
              f"mi={mi_ok} mn={mn_ok} | {reply[:110]!r}", flush=True)
        return row

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(WORKERS) as ex:
        out_rows = list(ex.map(one, rows))
    return summarise(tag, out_rows, kind="questions")


def run_conversations(limit: int | None, ids: set | None) -> dict:
    rows = read_csv(config.PRACTICE_DIR / "conversations.csv")
    convs = defaultdict(list)
    for r in rows:
        convs[r["conversation_id"]].append(r)
    keys = sorted(convs)
    if ids:
        keys = [k for k in keys if k in ids]
    keys = keys[:limit] if limit else keys
    def one_conv(cid):
        agent = Agent(mode="agent", channel="chat")
        rows_c, lines = [], [f"--- {cid}: {convs[cid][0]['scenario']}"]
        for r in sorted(convs[cid], key=lambda x: int(x["turn"])):
            try:
                res = agent.respond(r["customer_message"])
            except Exception as e:
                lines.append(f"  ! error: {e}")
                res = None
            reply = res.reply if res else ""
            dec_ok = bool(res) and res.decision == r["expected_decision"]
            mi_ok, missing = must_include_ok(reply, r["must_include"])
            mn_ok, bad = must_not_ok(reply, r["must_not_include"])
            t_ok = bool(res) and tool_ok(r["expected_tool_calls"], res.tool_calls)
            gold = gold_ids(r["gold_evidence"])
            row = {"conversation_id": cid, "turn": int(r["turn"]), "message": r["customer_message"],
                   "expected_decision": r["expected_decision"], "decision": res.decision if res else "",
                   "reply": reply, "decision_ok": dec_ok, "must_include_ok": mi_ok, "missing": missing,
                   "must_not_ok": mn_ok, "forbidden_found": bad, "tool_ok": t_ok,
                   "expected_tools": r["expected_tool_calls"],
                   "tools": [(c["tool"], c["args"].get("verification_value", "")) for c in res.tool_calls] if res else [],
                   "correct": dec_ok and mi_ok and mn_ok and t_ok, "gold": gold,
                   "sources": res.sources if res else [],
                   "retrieved": [h["id"] for h in res.retrieved] if res else [],
                   "recall_cited": recall(gold, res.sources if res else []),
                   "recall_retrieved": recall(gold, ([h["id"] for h in res.retrieved] + ["STY-1", "IDV-2", "IDV-3",
                                               "ESC-1", "ESC-3", "ESC-4"]) if res else []),
                   "latency_s": res.latency_s if res else None, "model_used": llm.last_model_used}
            rows_c.append(row)
            flag = "OK " if row["correct"] else "XX "
            lines.append(f"  {flag}{cid} t{r['turn']} dec={row['decision']}/{r['expected_decision']} mi={mi_ok} "
                         f"mn={mn_ok} tool={t_ok} | {reply[:120]!r}")
        print("\n".join(lines), flush=True)
        return rows_c

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(WORKERS) as ex:
        out_rows = [row for rows_c in ex.map(one_conv, keys) for row in rows_c]
    return summarise("step3_conversations", out_rows, kind="conversations")


def summarise(tag: str, rows: list[dict], kind: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    n = len(rows)

    def rate(key):
        return round(sum(1 for r in rows if r.get(key)) / n, 3) if n else 0

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    s = {"run": tag, "model": f"{config.LLM_PROVIDER}:{config.LLM_MODEL}", "n": n,
         "correct": sum(1 for r in rows if r["correct"]), "accuracy": rate("correct"),
         "decision_acc": rate("decision_ok"), "must_include_acc": rate("must_include_ok"),
         "must_not_ok": rate("must_not_ok"), "retrieval_recall_cited": mean("recall_cited"),
         "retrieval_recall_retrieved": mean("recall_retrieved"),
         "avg_latency_s": mean("latency_s"), "time": time.strftime("%Y-%m-%d %H:%M")}
    if kind == "conversations":
        s["tool_acc"] = rate("tool_ok")
        s["conversations_fully_correct"] = sum(
            1 for cid in {r["conversation_id"] for r in rows}
            if all(r["correct"] for r in rows if r["conversation_id"] == cid))
        s["conversations"] = len({r["conversation_id"] for r in rows})
    (OUT / f"{tag}_summary.json").write_text(json.dumps(s, indent=2))
    print("\nSUMMARY", json.dumps(s, indent=2))
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", help="comma-separated question/conversation IDs")
    ap.add_argument("--asr", action="store_true", help="use asr_text instead of the typed question")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    WORKERS = a.workers
    idset = set(a.ids.split(",")) if a.ids else None
    if a.step == 3:
        run_conversations(a.limit, idset)
    else:
        run_questions(a.step, a.limit, a.asr, idset)
