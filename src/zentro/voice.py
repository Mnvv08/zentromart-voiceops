"""Step 4: Gnani speech-to-text and text-to-speech (the only speech services used).

STT: Gnani REST STT (gnani-vachana SDK), clips up to 60 s; longer WAV clips are split.
TTS: Gnani timbre-v2.5, voice chosen by reply language.
"""
from __future__ import annotations

import io
import time
import re
import shutil
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config

GNANI_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
HINDI_ROMAN = {"hai", "hain", "mera", "meri", "mere", "mujhe", "kya", "nahi", "nahin", "kab", "kaise", "aur", "ka",
               "ki", "ke", "ko", "hoon", "hu", "tha", "thi", "chahiye", "kitne", "din", "kar", "karna", "karo",
               "wala", "wali", "ho", "gaya", "gayi", "abhi", "baare", "mein", "se", "bhi", "paisa", "kyun", "kahan",
               "haan", "theek", "achha", "sakta", "sakti", "raha", "rahi", "hua", "hui", "lekin", "toh", "yeh", "woh"}

_stt = None
_tts = None

import logging

logging.getLogger("gnani").setLevel(logging.CRITICAL)   # SDK logs every retried 429 as ERROR


def _clients():
    global _stt, _tts
    if _stt is None:
        if not config.GNANI_API_KEY:
            raise RuntimeError("GNANI_API_KEY is not set. Add it to your .env file (see .env.example).")
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient

        _stt = GnaniSTTClient(api_key=config.GNANI_API_KEY)
        _tts = GnaniTTSClient(api_key=config.GNANI_API_KEY)
    return _stt, _tts


# ----------------------------------------------------------------------------- input audio
def prepare_audio(path: Path) -> list[Path]:
    """Return one or more Gnani-compatible clips (<= 55 s each)."""
    path = Path(path)
    if path.suffix.lower() not in GNANI_EXT:
        if not shutil.which("ffmpeg"):
            raise ValueError(f"{path.name}: format not supported by Gnani STT and ffmpeg is not installed")
        out = Path(tempfile.mkdtemp()) / (path.stem + ".wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "16000", str(out)],
                       check=True)
        path = out
    if path.suffix.lower() == ".wav":
        try:
            fix_wav_header(path)
            with wave.open(str(path)) as w:
                frames, rate = w.getnframes(), w.getframerate()
                if frames / rate > 58:
                    return _split_wav(path, 55)
        except (wave.Error, EOFError, ValueError):
            pass
    return [path]


def fix_wav_header(path: Path) -> None:
    """Streaming WAVs (e.g. TTS output) carry 0xFFFFFFFF as size; rewrite RIFF/data sizes from the real file size."""
    import struct

    data = bytearray(Path(path).read_bytes())
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return
    pos = data.find(b"data", 12)
    if pos < 0:
        return
    real = len(data) - (pos + 8)
    if struct.unpack("<I", data[pos + 4:pos + 8])[0] != real:
        data[pos + 4:pos + 8] = struct.pack("<I", real)
        data[4:8] = struct.pack("<I", len(data) - 8)
        Path(path).write_bytes(bytes(data))


def _split_wav(path: Path, seconds: int) -> list[Path]:
    out = []
    tmp = Path(tempfile.mkdtemp())
    with wave.open(str(path)) as w:
        params = w.getparams()
        per = params.framerate * seconds
        i = 0
        while True:
            data = w.readframes(per)
            if not data:
                break
            p = tmp / f"{path.stem}_{i}.wav"
            with wave.open(str(p), "wb") as o:
                o.setparams(params)
                o.writeframes(data)
            out.append(p)
            i += 1
    return out


def _transcribe_one(path: Path, lang: str) -> str:
    stt, _ = _clients()
    texts = []
    for clip in prepare_audio(path):
        fmt = "transcribe" if lang in ("en-IN", "hi-IN") else "verbatim"   # ITN: numbers as digits
        r = None
        for attempt in range(5):
            try:
                r = stt.transcribe(str(clip), language_code=lang, format=fmt)
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "429" in msg or "rate" in msg.lower() or "50" in msg[:5]:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if fmt != "verbatim":       # ITN not available -> plain transcript
                    fmt = "verbatim"
                    continue
                raise
        if r is None:
            raise RuntimeError(f"Gnani STT failed for {clip.name} ({lang})")
        texts.append((r.get("transcript") or "").strip())
    return " ".join(t for t in texts if t).strip()


def transcribe(path: str | Path) -> dict:
    """Return {"heard": best transcript, "transcripts": {lang: text}, "language_hint": "en"|"hi"}."""
    path = Path(path)
    mode = config.STT_MODE
    if mode != "dual":
        t = _transcribe_one(path, mode)
        return {"heard": t, "transcripts": {mode: t}, "language_hint": "hi" if mode == "hi-IN" else "en"}
    with ThreadPoolExecutor(2) as ex:
        f_en = ex.submit(_transcribe_one, path, "en-IN")
        f_hi = ex.submit(_transcribe_one, path, "hi-IN")
        en, hi = _safe(f_en), _safe(f_hi)
    words = re.findall(r"[a-z]+", en.lower())
    hindi_words = sum(1 for w in words if w in HINDI_ROMAN)
    spoke_hindi = bool(words) and (hindi_words >= 2 and hindi_words / max(len(words), 1) >= 0.15)
    if not en and hi:
        spoke_hindi = True
    if spoke_hindi:
        transcripts = {"en-IN (romanised)": en, "hi-IN": hi}
        heard = en if en else hi
    else:
        transcripts = {"en-IN": en}
        heard = en
    return {"heard": heard, "transcripts": transcripts, "language_hint": "hi" if spoke_hindi else "en",
            "raw": {"en-IN": en, "hi-IN": hi}}


def _safe(fut) -> str:
    try:
        return fut.result()
    except Exception as e:  # noqa: BLE001
        print(f"  [stt warning] {e}")
        return ""


# ----------------------------------------------------------------------------- output audio
def reply_language(text: str) -> str:
    if len(DEVANAGARI.findall(text)) > 0.3 * max(len(re.sub(r"\s", "", text)), 1):
        return "hi"
    words = re.findall(r"[a-z]+", text.lower())
    if words and sum(1 for w in words if w in HINDI_ROMAN) / len(words) >= 0.15:
        return "hinglish"
    return "en"


def synthesize(text: str, out_file: str | Path) -> Path:
    _, tts = _clients()
    lang = reply_language(text)
    voice, code = {"hi": (config.TTS_VOICE_HI, "hi-IN"),
                   "hinglish": (config.TTS_VOICE_HINGLISH, config.TTS_LANG_HINGLISH),
                   "en": (config.TTS_VOICE_EN, "en-IN")}[lang]
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    from gnani.tts import AudioConfig

    cfg = AudioConfig(sample_rate=24000, container="wav")
    try:
        tts.synthesize(text, voice=voice, model=config.TTS_MODEL, language=code, audio_config=cfg,
                       output_file=str(out_file))
    except Exception:
        # fall back to the documented minimal call from the capstone brief
        tts.synthesize(text, voice=config.TTS_VOICE_EN, model=config.TTS_MODEL, language="en-IN",
                       output_file=str(out_file))
    fix_wav_header(out_file)
    return out_file


SORRY = {
    "en": "Sorry, I didn't catch that. Could you please say it again?",
    "hi": "माफ़ कीजिए, मैं ठीक से सुन नहीं पाई। क्या आप फिर से बोल सकते हैं?",
}


def voice_turn(audio_path: str | Path, agent, out_wav: str | Path) -> dict:
    """One spoken turn: Gnani STT -> agent (same brain as text) -> Gnani TTS."""
    stt = transcribe(audio_path)
    heard = stt["heard"]
    if not heard.strip():
        reply_text = SORRY["en"]
        res = None
        spoken = reply_text
    else:
        hint = ("(Language hint: the customer spoke Hindi/Hinglish.) " if stt["language_hint"] == "hi" else "")
        alt = dict(stt["transcripts"])
        if hint:
            alt = {k + " " + hint if i == 0 else k: v for i, (k, v) in enumerate(alt.items())}
        res = agent.respond(heard, alt_transcripts=alt if len(alt) > 1 or hint else None)
        reply_text = res.reply
        spoken = res.spoken
    synthesize(spoken, out_wav)
    return {"heard": heard, "stt": stt, "reply": reply_text, "spoken": spoken, "result": res,
            "audio": str(out_wav)}
