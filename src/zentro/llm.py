"""Thin wrapper around OpenAI-compatible chat APIs (Groq, Gemini, OpenAI...).

Retries on rate limits and, when the primary model is throttled or overloaded, falls back to the next
configured model (LLM_FALLBACKS) so long batch runs keep going on free-tier keys.
"""
from __future__ import annotations

import re
import time

from openai import OpenAI

from . import config

_clients: dict = {}
_cooldown: dict = {}   # (provider, model) -> time when it may be used again


def _chain() -> list[tuple[str, str, str, str]]:
    """[(provider, model, base_url, key)] primary first; skips entries without a key."""
    out = [(config.LLM_PROVIDER, config.LLM_MODEL, config.LLM_BASE_URL, config.LLM_API_KEY)]
    for item in [x.strip() for x in config.LLM_FALLBACKS.split(",") if x.strip()]:
        prov, _, model = item.partition(":")
        base, key, model = config.provider_settings(prov.strip(), model.strip())
        if key and (prov, model) not in [(p, m) for p, m, _, _ in out]:
            out.append((prov.strip(), model, base, key))
    return [c for c in out if c[3]]


def client(base: str, key: str) -> OpenAI:
    if (base, key) not in _clients:
        _clients[(base, key)] = OpenAI(base_url=base, api_key=key, max_retries=0, timeout=90)
    return _clients[(base, key)]


def _wait_seconds(err: Exception, attempt: int) -> float:
    msg = str(err)
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)\s*(ms|s)", msg)
    if m:
        mins = float(m.group(1) or 0)
        v = float(m.group(2))
        return mins * 60 + (v / 1000 if m.group(3) == "ms" else v) + 0.5
    m = re.search(r"retry(?:_delay|Delay)?\D+([\d.]+)s", msg)
    if m:
        return float(m.group(1)) + 1
    return min(30.0, 2 ** attempt + 1)


last_model_used = ""


def chat(messages: list[dict], tools: list[dict] | None = None, temperature: float | None = None,
         max_tokens: int = 2500):
    """Return the assistant message object."""
    global last_model_used
    chain = _chain()
    if not chain:
        raise RuntimeError("No LLM API key found. Set GROQ_API_KEY or GEMINI_API_KEY in your .env file.")
    last: Exception | None = None
    for attempt in range(config.LLM_MAX_RETRIES * len(chain) + 1):
        now = time.time()
        ready = [c for c in chain if _cooldown.get(c[:2], 0) <= now]
        if not ready:
            time.sleep(max(0.5, min(_cooldown.get(c[:2], 0) for c in chain) - now))
            continue
        prov, model, base, key = ready[0]
        kwargs = dict(model=model, messages=messages, max_tokens=max_tokens,
                      temperature=config.LLM_TEMPERATURE if temperature is None else temperature)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            resp = client(base, key).chat.completions.create(**kwargs)
            last_model_used = f"{prov}:{model}"
            return resp.choices[0].message
        except Exception as e:  # noqa: BLE001
            last = e
            status = getattr(e, "status_code", None)
            text = str(e).lower()
            if status in (429, 500, 502, 503, 504) or "rate" in text or "timeout" in text or "timed out" in text or "connection" in text \
                    or "overloaded" in text or "unavailable" in text:
                wait = _wait_seconds(e, attempt) if status == 429 else 20.0
                _cooldown[(prov, model)] = time.time() + wait
                continue
            if status == 400 and "thought_signature" in text:
                # tool calls made earlier by another model carry no Gemini signature: replay them as text
                messages = flatten_tool_history(messages)
                kwargs["messages"] = messages
                continue
            if status == 400 and ("tool" in text or "failed_generation" in text) and attempt < 3:
                continue  # model produced a malformed tool call: just retry
            if status in (400, 404) and len(ready) > 1:
                _cooldown[(prov, model)] = time.time() + 300   # e.g. model retired: use the next one
                continue
            raise
    raise last  # type: ignore[misc]


def flatten_tool_history(messages: list[dict]) -> list[dict]:
    """Turn assistant tool calls + tool results into plain text so any provider accepts the history."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = "; ".join(f"{tc['function']['name']}({tc['function'].get('arguments', '')})" for tc in m["tool_calls"])
            out.append({"role": "assistant", "content": (m.get("content") or "") + f"\n[called tools: {calls}]"})
        elif m.get("role") == "tool":
            out.append({"role": "user", "content": f"[tool result] {m.get('content', '')}"})
        else:
            out.append(m)
    return out
