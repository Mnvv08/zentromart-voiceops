"""Text normalisation for voice: spoken digits -> numbers (input side) and
"write for the ear" rules (STY-3) for the TTS side."""
from __future__ import annotations

import re

DIGIT_WORDS = {
    # English
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "to": None, "three": "3", "four": "4", "for": None,
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    # Hinglish (romanised Hindi)
    "shunya": "0", "ek": "1", "do": None, "teen": "3", "char": "4", "chaar": "4", "paanch": "5", "panch": "5",
    "chhe": "6", "chheh": "6", "che": "6", "saat": "7", "aath": "8", "aat": "8", "nau": "9", "no": None,
    # Hindi (Devanagari)
    "शून्य": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पांच": "5", "पाँच": "5", "छह": "6", "छः": "6",
    "सात": "7", "आठ": "8", "नौ": "9",
}
# Ambiguous words (to, for, do, no) only count as digits inside a run of other digit words.
AMBIGUOUS = {"to": "2", "for": "4", "do": "2", "no": "9"}
DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
WORD_RE = re.compile(r"[A-Za-zऀ-ॿ]+|\d+|[^\sA-Za-zऀ-ॿ\d]+|\s+")


def spoken_digits_to_numbers(text: str) -> str:
    """'eight nine seven eight two eight three seven' -> '8978 2837' style digit runs.

    Handles 'double'/'triple', Devanagari digits and digit words in English, Hinglish and Hindi.
    Only runs of 3+ digit words are converted, so ordinary sentences are untouched.
    """
    text = text.translate(DEV_DIGITS)
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)      # ITN output "8,752" -> "8752"
    tokens = WORD_RE.findall(text)
    out: list[str] = []
    run: list[str] = []      # collected digits
    run_tokens: list[str] = []
    mult = 1

    def close():
        nonlocal run, run_tokens, mult
        if len(run) >= 3:
            out.append("".join(run))
            if run_tokens and run_tokens[-1].isspace():
                out.append(run_tokens[-1])
        else:
            out.extend(run_tokens)
        run, run_tokens, mult = [], [], 1

    for tok in tokens:
        low = tok.lower()
        if tok.isspace() or tok in {",", "-"}:
            if run_tokens:
                run_tokens.append(tok)
            else:
                out.append(tok)
            continue
        digit = None
        if low.isdigit() and len(low) <= 4 and run:
            digit = low
        elif low in DIGIT_WORDS and DIGIT_WORDS[low] is not None and low not in ("o",):
            digit = DIGIT_WORDS[low]
        elif low in AMBIGUOUS and run:
            digit = AMBIGUOUS[low]
        elif low == "o" and run:
            digit = "0"
        elif low in ("double", "triple"):
            run_tokens.append(tok)
            mult = 2 if low == "double" else 3
            continue
        if digit is not None:
            run.append(digit * mult)
            run_tokens.append(tok)
            mult = 1
        else:
            close()
            out.append(tok)
    close()
    return re.sub(r"[ \t]+", " ", "".join(out)).strip()


ORDER_ID_RE = re.compile(r"(?<!\d)(\d{4})[\s\-]?(\d{4})(?!\d)")
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def for_the_ear(text: str) -> str:
    """STY-3 + Gnani text-normalisation guide: dates with month and year, IDs in groups of four,
    rupee amounts in words-friendly form, no markdown."""
    t = re.sub(r"[*_`#>]+", "", text)
    t = re.sub(r"\[([A-Za-z]+-?[\w.]*)\]", "", t)                       # stray [RET-2] citations
    t = re.sub(r"(\d{4})-(\d{2})-(\d{2})",
               lambda m: f"{int(m.group(3))} {MONTHS[int(m.group(2)) - 1]} {m.group(1)}", t)
    t = re.sub(r"₹\s?([\d,]+)", lambda m: f"{m.group(1).replace(',', '')} rupees", t)
    t = re.sub(r"\bRs\.?\s?([\d,]+)", lambda m: f"{m.group(1).replace(',', '')} rupees", t)
    t = re.sub(r"(?<!\d)(\d{4})(\d{4})(?!\d)", r"\1 \2", t)          # 89782837 -> 8978 2837
    t = re.sub(r"\(?\s*(?:ticket\s*(?:ID|id|number)?\s*:?\s*)?\bTKT[0-9a-f]{20,}\b\s*\)?", " ", t)
    t = re.sub(r"\s*\n+\s*", " ", t)
    t = re.sub(r"^\s*[-•]\s*", "", t)
    return re.sub(r"\s{2,}", " ", t).strip()
