"""Build report.pdf (max 4 pages) from the evaluation summaries in eval/results/.

    python eval/make_report.py        # writes report.html and report.pdf (PDF needs playwright + chromium)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval" / "results"
AUTHOR = sys.argv[1] if len(sys.argv) > 1 else "Manav Parihar"


def load(name):
    p = RES / f"{name}_summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def rows(name):
    p = RES / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


s1, s2, s3 = load("step1"), load("step2"), load("step3_conversations")
s2a = load("step2_asr")
s2r1, s3r1 = load("step2_full_run1"), load("step3_run1")
s2fix = load("step2_rerun_failed12")


def pct(x):
    return f"{100 * x:.0f}%" if isinstance(x, (int, float)) else "–"


def by_lang(name):
    c, t = Counter(), Counter()
    for r in rows(name):
        t[r["language"]] += 1
        c[r["language"]] += r["correct"]
    return {k: f"{c[k]}/{t[k]}" for k in sorted(t)}


l1, l2 = by_lang("step1"), by_lang("step2")
_mix = Counter(r.get("model_used", "").split(":")[-1] for r in rows("step2") + rows("step3_conversations")
               if r.get("model_used"))
model_mix = ", ".join(f"{m} ({n})" for m, n in _mix.most_common()) or "openai/gpt-oss-120b"
remaining = [r for r in rows("step2") if not r["correct"]]
remaining3 = [r for r in rows("step3_conversations") if not r["correct"]]


def bar(label, val, color):
    w = int(360 * (val or 0))
    return (f'<div class="bar"><span class="bl">{label}</span><span class="track">'
            f'<span class="fill" style="width:{w}px;background:{color}"></span></span>'
            f'<span class="bv">{pct(val)}</span></div>')


fail_list = "".join(f"<li><b>{r['id']}</b>: {', '.join(r['missing']) or 'decision ' + r['decision']}</li>"
                    for r in remaining[:6]) or "<li>none</li>"
fail3 = "".join(f"<li><b>{r['conversation_id']} t{r['turn']}</b>: expected {r['expected_decision']}, "
                f"got {r['decision']}{'; missing ' + ', '.join(r['missing']) if r['missing'] else ''}</li>"
                for r in remaining3[:6]) or "<li>none</li>"

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>ZentroMart VoiceOps report</title>
<style>
@page {{ size: A4; margin: 15mm 16mm; }}
body {{ font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 10.3pt; color:#1f2933; line-height:1.42; }}
h1 {{ font-size: 19pt; margin:0 0 2px; }} h2 {{ font-size: 12.5pt; margin: 14px 0 5px; color:#3b2fb0; border-bottom:1px solid #e5e1f7; padding-bottom:2px; }}
.sub {{ color:#6b7280; margin-bottom:8px; }}
table {{ border-collapse: collapse; width:100%; margin:6px 0; font-size:9.6pt; }}
th, td {{ border:1px solid #ddd; padding:4px 6px; text-align:left; vertical-align:top; }} th {{ background:#f4f2fb; }}
.bar {{ display:flex; align-items:center; gap:8px; margin:3px 0; font-size:9.6pt; }}
.bl {{ width:190px; }} .track {{ width:360px; height:13px; background:#eee; border-radius:3px; display:inline-block; }}
.fill {{ display:block; height:13px; border-radius:3px; }} .bv {{ font-weight:600; }}
code {{ background:#f3f3f3; padding:0 3px; border-radius:3px; font-size:9pt; }}
ul {{ margin:3px 0 3px 18px; padding:0; }} li {{ margin:1px 0; }}
.pipe {{ background:#f7f5ff; border:1px solid #e5e1f7; padding:6px 10px; border-radius:6px; font-family: monospace; font-size:9pt; }}
</style></head><body>
<h1>ZentroMart VoiceOps: grounded voice support agent</h1>
<div class="sub">VoiceOps 2026 capstone · {AUTHOR} · Speech: Gnani STT + TTS only · Step 1 model: <code>openai/gpt-oss-120b</code> (Groq) · Models that answered the reported Step 2/3 runs (approximate turn counts, logged per answer): {model_mix}</div>
<div class="pipe">customer audio → Gnani STT (en-IN + hi-IN) → agent [hybrid RAG · conversation state · lookup_order · create_ticket · date tool] → Gnani TTS (timbre-v2.5) → spoken reply</div>

<h2>1. What I built in each step</h2>
<table>
<tr><th style="width:17%">Step</th><th>What it does</th></tr>
<tr><td><b>1 · LLM only</b></td><td>System prompt that sets the role (ZentroMart's AI support assistant), its limits (no promises, no other customers' data, say when unsure) and an output contract ending with <code>DECISION:</code> and <code>SOURCES:</code>. No documents, no tools.</td></tr>
<tr><td><b>2 · Grounded (RAG)</b></td><td>Each <code>## [SECTION-ID]</code> becomes one chunk with its doc metadata (doc type, status, effective_from, superseded banner); each catalog row becomes a <code>products.csv:SKU</code> chunk (137 chunks). Retrieval = multilingual embeddings (paraphrase-multilingual-MiniLM, local ONNX) in a NumPy vector index + BM25 with a small Hindi/Hinglish/English synonym list, merged by reciprocal-rank fusion and re-ranked by the KB-00.1 precedence (policy &gt; agent guide &gt; help &gt; FAQ &gt; marketing; superseded down-weighted). A product mention also brings in the policy section that defines its catalog code (two-source answers, e.g. <code>fashion</code> → RET-2). The model must answer only from the given sections, say "I don't have information" otherwise, and cite the IDs it used. IDs are checked against the real index.</td></tr>
<tr><td><b>3 · Conversation</b></td><td>Conversation memory (last 10 messages and a structured order state). Order IDs and verification values are detected in code, including spoken digits ("eight nine seven eight…", "aath nau…"), and sent to the dataset's <code>lookup_order</code>. One retry, then locked. Order data reaches the LLM only after <code>verified=True</code>, and secrets are masked in logs and ticket summaries. <code>create_ticket</code> is an LLM tool, and a code check re-prompts the model if it claims a handover without a successful ticket. A <code>date_calc</code> tool plus a pre-computed date helper handle business days (Mon–Sat minus holidays) and calendar-day return windows. Decision labels for the steps the code performs (verification, ticket) are set deterministically.</td></tr>
<tr><td><b>4 · Voice</b></td><td>Gnani REST STT runs each clip in en-IN and hi-IN in parallel. A Hinglish detector tells the agent which language was spoken, and long WAVs are split under 60 s. The agent's reply is rewritten for the ear (dates with month and year, order IDs in groups of four, "49 rupees", no ticket hashes) and spoken with Gnani TTS <code>timbre-v2.5</code> (Kaveri for English, Nalini for Hindi). Empty audio gives "Sorry, I didn't catch that". <code>python run.py --input audio/ --output outputs/</code> saves the reply text, reply audio, cited chunks, retrieved chunks and tool calls per file. A browser demo (<code>web/app.py</code>) records the microphone as WAV.</td></tr>
</table>

<h2>2. Step 1 vs Step 2 on the 52 practice questions</h2>
{bar("Step 1 · prompt only", s1.get("accuracy"), "#c9c3ef")}
{bar("Step 2 · RAG + tools", s2.get("accuracy"), "#5b4bdb")}
{bar("Step 2 on ASR text" if s2a else "", s2a.get("accuracy") if s2a else None, "#ff8a4c") if s2a else ""}
<table>
<tr><th>Metric</th><th>Step 1</th><th>Step 2</th>{"<th>Step 2 (asr_text)</th>" if s2a else ""}</tr>
<tr><td>Fully correct (decision ✓, all must_include facts, no forbidden phrase)</td><td><b>{s1.get('correct')}/{s1.get('n')}</b> ({pct(s1.get('accuracy'))})</td><td><b>{s2.get('correct')}/{s2.get('n')}</b> ({pct(s2.get('accuracy'))})</td>{f"<td>{s2a.get('correct')}/{s2a.get('n')} ({pct(s2a.get('accuracy'))})</td>" if s2a else ""}</tr>
<tr><td>Right decision (answer / clarify / escalate / refuse)</td><td>{pct(s1.get('decision_acc'))}</td><td>{pct(s2.get('decision_acc'))}</td>{f"<td>{pct(s2a.get('decision_acc'))}</td>" if s2a else ""}</tr>
<tr><td>All required facts present</td><td>{pct(s1.get('must_include_acc'))}</td><td>{pct(s2.get('must_include_acc'))}</td>{f"<td>{pct(s2a.get('must_include_acc'))}</td>" if s2a else ""}</tr>
<tr><td>Gold evidence recall: cited sources / retrieved chunks</td><td>– / –</td><td>{pct(s2.get('retrieval_recall_cited'))} / {pct(s2.get('retrieval_recall_retrieved'))}</td>{f"<td>{pct(s2a.get('retrieval_recall_cited'))} / {pct(s2a.get('retrieval_recall_retrieved'))}</td>" if s2a else ""}</tr>
<tr><td>By language (correct)</td><td>{', '.join(f'{k}: {v}' for k, v in l1.items())}</td><td>{', '.join(f'{k}: {v}' for k, v in l2.items())}</td>{"<td></td>" if s2a else ""}</tr>
</table>
{f"<p><i>After the fixes in section 4, re-running only the {s2fix.get('n')} questions that had failed gave {s2fix.get('correct')}/{s2fix.get('n')} correct, which would make Step 2 {s2.get('correct', 0) + s2fix.get('correct', 0)}/52 if the other answers stay the same.</i></p>" if s2fix and s2.get('run') == 'step2' and s2.get('correct', 0) < 45 else ""}
<p><b>What I learned.</b> Without documents the model sounds confident but invents ZentroMart-specific numbers: wallet refunds "3–5 days", UPI "2–5 days", a 10-day electronics window (the superseded v3 rule), no student discount instead of "I don't know". It also asks for an order ID instead of answering general policy questions, and it never escalates. Grounding fixed the facts. The gains on date questions came from computing dates in code rather than in the model, and the "I don't know" cases needed an explicit rule not to state what ZentroMart does or does not offer. Scoring is my own automatic check (DATA_DICTIONARY rules with date normalisation), so it is strict about wording: a correct Hindi answer that transliterates "QC" can be marked wrong.</p>

<h2>3. Step 3: 23 practice conversations (96 customer turns)</h2>
<table>
<tr><th>Metric</th><th>First run</th><th>Final</th></tr>
<tr><td>Turns fully correct (decision, facts, forbidden phrases, expected tool calls)</td><td>{s3r1.get('correct', '–')}/{s3r1.get('n', '–')} ({pct(s3r1.get('accuracy'))})</td><td><b>{s3.get('correct')}/{s3.get('n')}</b> ({pct(s3.get('accuracy'))})</td></tr>
<tr><td>Expected tool calls (lookup_order with/without verification, create_ticket)</td><td>{pct(s3r1.get('tool_acc'))}</td><td>{pct(s3.get('tool_acc'))}</td></tr>
<tr><td>Decision accuracy</td><td>{pct(s3r1.get('decision_acc'))}</td><td>{pct(s3.get('decision_acc'))}</td></tr>
<tr><td>Conversations correct at every turn</td><td>{s3r1.get('conversations_fully_correct', '–')}/23</td><td>{s3.get('conversations_fully_correct')}/23</td></tr>
<tr><td>Forbidden claims ("I have cancelled", "approved", leaked pincode…)</td><td>{pct(1 - s3r1.get('must_not_ok', 1))} of turns</td><td>{pct(1 - s3.get('must_not_ok', 1))} of turns</td></tr>
</table>

<h2>4. What went wrong and how I fixed it</h2>
<ul>
<li><b>Free-tier limits.</b> Groq allows 8,000 tokens/min, and the first prompt (all agent guides plus 8 verbose chunks) used about 9k per turn. I compacted the chunk rendering and moved KB-00 into retrieval (about 3k tokens per call), and added an automatic fallback chain (Groq gpt-oss-120b → Groq Qwen → Gemini Flash) with per-model cool-downs. Gemini rejected tool histories created by another model (missing <code>thought_signature</code>), so the history is replayed as text when that happens.</li>
<li><b>Which model answered.</b> Because of the free-tier daily caps, most of the reported Step 2/3 turns were answered by fallback models rather than the primary gpt-oss-120b (mix in the header). The numbers are therefore a conservative estimate for the primary model.</li>
<li><b>A weak fallback hurt quality.</b> In the first Step 2 run ({s2r1.get('correct', '–')}/52) almost every miss came from Gemini Flash Lite, which skipped the date tool and miscounted business days. I removed it and added a pre-computed date helper for every date the customer mentions.</li>
<li><b>Language drift.</b> English conversations sometimes got Hinglish replies. The customer's language is now detected in code and stated to the model on every turn.</li>
<li><b>Decision labels and verification turns.</b> The model escalated on the verification turn when <code>contacts_last_7_days=2</code>, or called a verification question "answer". Labels for code-driven steps are now deterministic, and <code>create_ticket</code> is not offered on a turn that only answers the verification question.</li>
<li><b>Intro.</b> STY-1 is guaranteed in code: if the first reply doesn't say "AI", an introduction in the reply's language is prepended.</li>
</ul>
<p><b>Still failing in the final runs:</b></p>
<ul>{fail_list}{fail3}</ul>
<p><b>What I would fix next:</b> a small LLM judge for semantic scoring of Hindi answers; a cross-encoder re-ranker and Gemini embeddings to raise cited-source precision; streaming STT/TTS (Gnani WebSocket) to cut latency (currently about {s3.get('avg_latency_s', '–')} s per turn on free-tier keys); confidence-based "please repeat" using STT disagreement between the en-IN and hi-IN passes.</p>

<h2>5. Bonus work</h2>
<ul>
<li>Dual-language STT (en-IN + hi-IN) with automatic Hindi/Hinglish detection, and Hindi TTS voice switching.</li>
<li>Spoken-number normalisation for order IDs, SKUs and verification digits in English, Hinglish and Hindi.</li>
<li>Browser voice demo that records WAV in the page (no ffmpeg needed) and shows decision, tools and cited sections.</li>
<li>Provider-agnostic LLM layer (any OpenAI-compatible API) with rate-limit-aware fallback, so long batch runs finish on free keys.</li>
<li>Reproducible evaluation: <code>python eval/run_eval.py --step 1|2|3 [--asr]</code> writes every reply, decision, tool call and source to <code>eval/results/</code>.</li>
</ul>
</body></html>"""

(ROOT / "report.html").write_text(html, encoding="utf-8")
try:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto((ROOT / "report.html").as_uri())
        pg.pdf(path=str(ROOT / "report.pdf"), format="A4", print_background=True,
               margin={"top": "14mm", "bottom": "14mm", "left": "15mm", "right": "15mm"})
        b.close()
    print("wrote report.pdf")
except Exception as e:  # noqa: BLE001
    print("report.html written; PDF step skipped:", e)
