"""Central configuration. Every value can be overridden with an environment variable (see .env.example)."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


# ---- Dataset ---------------------------------------------------------------
DATASET_DIR = Path(_env("DATASET_DIR", str(ROOT / "Capstone_Project_Dataset_FTTT_2026"))).expanduser().resolve()
KB_DIR = DATASET_DIR / "kb"
DATA_DIR = DATASET_DIR / "data"
PRACTICE_DIR = DATASET_DIR / "practice"
TOOLS_DIR = DATASET_DIR / "tools"

# Fixed "today" for the dataset (README / KB-00.2)
TODAY = _env("TODAY", "2026-10-20")

# ---- LLM ---------------------------------------------------------------------
# Any OpenAI-compatible endpoint works. Built-in presets: groq, gemini.
LLM_PROVIDER = _env("LLM_PROVIDER", "groq").lower()
_PRESETS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-2.5-flash"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
}
_base, _key_var, _model = _PRESETS.get(LLM_PROVIDER, _PRESETS["groq"])
LLM_BASE_URL = _env("LLM_BASE_URL", _base)
LLM_API_KEY = _env("LLM_API_KEY", _env(_key_var))
LLM_MODEL = _env("LLM_MODEL", _model)
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.1"))
LLM_MAX_RETRIES = int(_env("LLM_MAX_RETRIES", "8"))
# Fallback models used automatically when the primary is rate-limited or overloaded.
# Format: provider:model,provider:model  (only providers whose key is set are used)
LLM_FALLBACKS = _env("LLM_FALLBACKS", "groq:qwen/qwen3.8-27b,gemini:gemini-3-flash-preview,gemini:gemini-3.5-flash,groq:openai/gpt-oss-20b")


def provider_settings(provider: str, model: str = "") -> tuple[str, str, str]:
    base, key_var, default_model = _PRESETS.get(provider, _PRESETS["groq"])
    return base, _env(key_var), model or default_model

# ---- Retrieval ---------------------------------------------------------------
EMBED_MODEL = _env("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
TOP_K = int(_env("TOP_K", "6"))
CACHE_DIR = Path(_env("CACHE_DIR", str(ROOT / ".cache"))).expanduser()

# ---- Voice (Gnani only) -----------------------------------------------------
GNANI_API_KEY = _env("GNANI_API_KEY")
# dual = transcribe each clip with en-IN AND hi-IN and let the agent use the better one
STT_MODE = _env("STT_MODE", "dual")  # dual | en-IN | hi-IN | any Gnani language code
TTS_MODEL = _env("TTS_MODEL", "timbre-v2.5")
TTS_VOICE_EN = _env("TTS_VOICE_EN", "Kaveri")
TTS_VOICE_HI = _env("TTS_VOICE_HI", "Nalini")
TTS_VOICE_HINGLISH = _env("TTS_VOICE_HINGLISH", "Kaveri")
TTS_LANG_HINGLISH = _env("TTS_LANG_HINGLISH", "en-IN")
