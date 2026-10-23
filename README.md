# ZentroMart VoiceOps: grounded voice support agent

Capstone project for **VoiceOps 2026** (LLM & GenAI course, Consulting & Analytics Club IIT Guwahati × Gnani.ai).

A customer support agent for ZentroMart (a fictional Indian online store) that:

1. answers with an LLM (Step 1, prompt-only baseline),
2. answers **only from the company documents** with hybrid RAG and reports the section IDs it used (Step 2),
3. holds a multi-turn conversation: memory, identity-checked `lookup_order`, `create_ticket` handover, polite refusals (Step 3),
4. listens and speaks through **Gnani STT and Gnani TTS** (Step 4). No other speech service is used.

```
customer audio ──► Gnani STT (en-IN + hi-IN) ──► agent (LLM + RAG + tools) ──► Gnani TTS ──► spoken reply
```

## Quick start (macOS / Linux)

```bash
git clone https://github.com/Mnvv08/zentromart-voiceops.git && cd zentromart-voiceops
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# put the extracted dataset folder next to the code (or set DATASET_DIR in .env)
#   zentromart-voiceops/Capstone_Project_Dataset_FTTT_2026/{kb,data,tools,practice}
cp .env.example .env        # then add GROQ_API_KEY and GNANI_API_KEY
```

The first run downloads a small multilingual embedding model (~220 MB, cached in `.cache/`).

## The run command (batch voice evaluation)

```bash
python run.py --input path/to/audio_folder --output outputs/
```

For **each audio file** (wav, mp3, flac, ogg, m4a, aac) it runs Gnani STT, then the agent, then Gnani TTS, and saves:

| File | Contents |
| --- | --- |
| `outputs/<name>_reply.wav` | the spoken reply (Gnani TTS) |
| `outputs/<name>.json` | transcript, reply text, decision, **`sources_used` (section IDs) and `chunks_used` (their text)**, all retrieved chunks, tool calls |
| `outputs/results.csv`, `outputs/results.jsonl` | one row per file |

Each file is treated as a separate call. If the folder contains multi-turn calls named with a shared prefix
(`call1_01.wav`, `call1_02.wav`, …), add `--conversation` so turns share memory.
`--stt-lang hi-IN` forces one STT language instead of the default dual en-IN + hi-IN pass.

Other modes:

```bash
python run.py --text "Can I return a swimsuit that doesn't fit?"   # one text question
python run.py --chat                                               # multi-turn text chat
python web/app.py                                                  # browser voice demo at http://localhost:5050
```

## Evaluation (practice sets)

```bash
python eval/run_eval.py --step 1        # prompt-only baseline, practice/questions.csv
python eval/run_eval.py --step 2        # RAG agent, same questions
python eval/run_eval.py --step 2 --asr  # RAG agent on the asr_text (speech-like) version
python eval/run_eval.py --step 3        # 23 multi-turn conversations (96 turns)
```

Scoring follows `DATA_DICTIONARY.md`: decision in `acceptable_decisions`, every `must_include` fact present
(alternatives with `/`, dates normalised), no `must_not_include` phrase, plus expected tool calls for conversations
and gold-evidence recall for retrieval. Results are written to `eval/results/`.

## How it works

| Part | File | Notes |
| --- | --- | --- |
| Chunking | `src/zentro/kb.py` | One chunk per `## [SECTION-ID]` section, keeping `doc_type`, `status`, `effective_from` and document banners (e.g. "v3 applies only to electronics delivered before 1 Aug"). Each product row is a chunk `products.csv:SKU`. |
| Retrieval | `src/zentro/retriever.py` | Multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`, local ONNX via fastembed) in a NumPy vector index + BM25 with a small bilingual synonym expansion, fused by reciprocal rank and re-ranked by document precedence (KB-00.1). Product mentions pull in the catalog row and the policy sections that define its codes (two-source answers). |
| Agent | `src/zentro/agent.py`, `prompts.py` | OpenAI-compatible tool calling (Groq `openai/gpt-oss-120b` by default). Tools: `search_knowledge_base`, `date_calc`, `create_ticket`, `lookup_order`. The agent guides (style, escalation, verification) are always in context; the retrieved sections change every turn. The model must end with `DECISION:` and `SOURCES:`; sources are checked against real section IDs. |
| Verification | `agent.py` | Order IDs and phone-last-4 / pincode values are detected in code (including spoken digits like "eight nine seven eight") and passed to the dataset's `lookup_order`. One retry, then locked. Order data reaches the LLM only after `verified=True`. Secrets are masked in logs and removed from ticket summaries. |
| Handover | `agent.py` | `create_ticket` from the dataset; the agent can only claim a ticket when the tool returned `created=True` (a code check re-prompts the model if it claims a handover without the tool call). |
| Dates | `src/zentro/dates.py` | Business days (Mon–Sat, skipping `holidays.csv`) and calendar days, computed in code from today = 20 Oct 2026. |
| Voice | `src/zentro/voice.py` | Gnani STT REST (each clip is transcribed with en-IN and hi-IN, and the agent is told which language was spoken). Gnani TTS `timbre-v2.5` (Kaveri for English, Nalini for Hindi). Replies are rewritten for the ear (dates with month and year, order IDs in groups of four, "49 rupees", no ticket IDs). Empty audio gives "Sorry, I didn't catch that". |

## Configuration (`.env`)

See `.env.example`. Main variables: `LLM_PROVIDER` (groq | gemini | openai), `GROQ_API_KEY`, `GNANI_API_KEY`,
`DATASET_DIR`, optional `LLM_MODEL`, `STT_MODE`, `TTS_VOICE_EN`, `TTS_VOICE_HI`.
Keys are read from the environment only. `.env` is git-ignored.

## Repository layout

```
run.py                 batch voice run / text modes
src/zentro/            agent, retrieval, voice, dates, prompts
eval/                  practice-set evaluation and scoring
web/                   browser voice demo (records WAV in the browser)
sample_outputs/        example output of run.py on test audio
report.pdf             short report
```

Data used: only the synthetic dataset supplied with the capstone. The dataset itself is not included in this repo.
