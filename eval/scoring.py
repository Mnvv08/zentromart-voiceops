"""Automatic scoring that follows DATA_DICTIONARY.md:
- must_include: every ' ; ' fact required, ' / ' alternatives equivalent; case-insensitive; dates and
  native-script digits normalised.
- must_not_include: none of the forbidden phrases may appear.
- decision: in acceptable_decisions (questions) / equal to expected_decision (conversations).
- retrieval: recall of gold_evidence section IDs among the sources the agent reported / retrieved.
"""
from __future__ import annotations

import re

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
HI_MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]
HI_ALT = {"फरवरी": 1, "सितम्बर": 8, "नवम्बर": 10, "दिसम्बर": 11, "अक्तूबर": 9}
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty-eight": 48,
             "forty eight": 48, "seventy-nine": 79, "seventy nine": 79, "forty-nine": 49, "forty nine": 49}
DEV = str.maketrans("०१२३४५६७८९", "0123456789")


def split_list(cell: str) -> list[str]:
    return [x.strip() for x in (cell or "").split(" ; ") if x.strip()] if cell else []


def clean_unicode(t: str) -> str:
    for a, b in (("\u2019", "'"), ("\u2018", "'"), ("\u202f", " "), ("\u00a0", " "), ("\u2011", "-"),
                 ("\u2010", "-"), ("\u2013", "-"), ("\u2014", "-"), ("\u200b", "")):
        t = t.replace(a, b)
    return t


def normalise(text: str) -> str:
    t = (text or "").translate(DEV)
    t = clean_unicode(t)
    low = t.lower()
    extra = []
    month_alt = "|".join(MONTHS)
    # 27 October 2026 / 27th of October / October 27, 2026
    for m in re.finditer(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({month_alt})(?:,?\s+(\d{{4}}))?", low):
        extra.append(_fmt(int(m.group(1)), MONTHS.index(m.group(2)) + 1, m.group(3)))
    for m in re.finditer(rf"({month_alt})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?", low):
        extra.append(_fmt(int(m.group(2)), MONTHS.index(m.group(1)) + 1, m.group(3)))
    hi_names = {n: i for i, n in enumerate(HI_MONTHS)} | HI_ALT
    for name, idx in hi_names.items():
        for m in re.finditer(rf"(\d{{1,2}})\s*{name}(?:\s*,?\s*(\d{{4}}))?", t):
            extra.append(_fmt(int(m.group(1)), idx + 1, m.group(2)))
    for w, n in NUM_WORDS.items():
        if re.search(rf"\b{w}\b", low):
            extra.append(str(n))
    return low + " || " + " ".join(extra)


def _fmt(day: int, month: int, year: str | None) -> str:
    y = year or "2026"
    return (f"{y}-{month:02d}-{day:02d} {day} {MONTHS[month - 1]} {y} "
            f"{day} {HI_MONTHS[month - 1]} {y}")


def contains(norm: str, phrase: str) -> bool:
    p = phrase.strip().lower().translate(DEV)
    if not p:
        return True
    if re.fullmatch(r"[\d,.]+", p):
        return re.search(rf"(?<![\d]){re.escape(p)}(?![\d])", norm) is not None
    return p in norm


def must_include_ok(reply: str, cell: str) -> tuple[bool, list]:
    norm = normalise(reply)
    missing = []
    for fact in split_list(cell):
        alts = [a for a in fact.split(" / ")]
        if not any(contains(norm, a) for a in alts):
            missing.append(fact)
    return not missing, missing


def must_not_ok(reply: str, cell: str) -> tuple[bool, list]:
    norm = clean_unicode(reply or "").lower().translate(DEV)
    bad = [p for p in split_list(cell) if p.strip().lower() in norm]
    return not bad, bad


def gold_ids(cell: str) -> list[str]:
    return split_list(cell)


def recall(gold: list[str], got: list[str]) -> float | None:
    if not gold:
        return None
    return sum(1 for g in gold if g in got) / len(gold)
