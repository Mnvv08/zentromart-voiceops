"""Load the knowledge base (kb/*.md) and product catalog into section-level chunks.

Each chunk keeps its [SECTION-ID], doc metadata (doc_type, status, effective_from)
so the agent can follow the precedence rules in KB-00.1 and report sources.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config

SECTION_RE = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*)$")

# KB-00.1 precedence: lower number wins
PRECEDENCE = {"policy": 1, "agent_guide": 2, "help": 3, "faq": 4, "marketing": 5, "meta": 2, "catalog": 1}

# Agent guides describe how the agent must behave; they are always given to the LLM.
ALWAYS_ON_DOCS = {"AGT-STY", "AGT-ESC", "AGT-IDV"}


@dataclass
class Chunk:
    id: str                     # section ID, e.g. RET-2, or products.csv:ELE-0012
    doc_id: str
    doc_title: str
    title: str
    text: str                   # body text
    doc_type: str
    status: str = "current"
    effective_from: str = ""
    language: str = "en"
    source_file: str = ""
    note: str = ""              # document-level note (e.g. superseded banner)
    meta: dict = field(default_factory=dict)

    @property
    def precedence(self) -> int:
        p = PRECEDENCE.get(self.doc_type, 5)
        return p + (3 if self.status == "superseded" else 0)

    def embed_text(self) -> str:
        return f"{self.doc_title}. {self.title}. {self.note} {self.text}".strip()

    def render(self) -> str:
        """How the chunk is shown to the LLM (compact: the free-tier token budget is small)."""
        if self.doc_type == "catalog":
            return f"[{self.id}] {self.text}"
        head = f"[{self.id}] {self.title} ({self.doc_id}, {self.doc_type}, {self.status}"
        head += f", from {self.effective_from})" if self.effective_from else ")"
        body = re.sub(r"\n{2,}", "\n", self.text.strip())
        body = re.sub(r"\|\s*-{3,}[\s|-]*\n", "", body)          # drop markdown table rulers
        if self.note:
            body = f"NOTE: {self.note}\n{body}"
        return f"{head}\n{body}"


def _parse_header(lines: list[str]) -> tuple[dict, int]:
    meta: dict = {}
    if not lines or lines[0].strip() != "---":
        return meta, 0
    for i in range(1, len(lines)):
        line = lines[i].strip()
        if line == "---":
            return meta, i + 1
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, 0


def load_kb_chunks(kb_dir: Path | None = None) -> list[Chunk]:
    kb_dir = kb_dir or config.KB_DIR
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        meta, start = _parse_header(lines)
        doc_id = meta.get("doc_id", path.stem)
        doc_title = meta.get("title", path.stem)
        # Preamble = text between the H1 and the first section (e.g. "superseded" banners)
        preamble: list[str] = []
        current: Chunk | None = None
        body: list[str] = []

        def flush():
            if current is not None:
                current.text = "\n".join(body).strip()
                chunks.append(current)

        for line in lines[start:]:
            m = SECTION_RE.match(line.strip())
            if m:
                flush()
                body = []
                current = Chunk(
                    id=m.group(1).strip(), doc_id=doc_id, doc_title=doc_title, title=m.group(2).strip(),
                    text="", doc_type=meta.get("doc_type", "other"), status=meta.get("status", "current"),
                    effective_from=meta.get("effective_from", ""), language=meta.get("language", "en"),
                    source_file=path.name, meta=meta,
                )
            elif current is None:
                s = line.strip()
                if s and not s.startswith("# "):
                    preamble.append(s.lstrip("> ").strip())
            else:
                body.append(line)
        flush()
        note = " ".join(preamble).strip()
        if note:
            for c in chunks:
                if c.doc_id == doc_id:
                    c.note = note
    return chunks


def load_products(data_dir: Path | None = None) -> list[dict]:
    data_dir = data_dir or config.DATA_DIR
    with open(data_dir / "products.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def product_chunks(products: list[dict]) -> list[Chunk]:
    out = []
    for p in products:
        text = (f"SKU {p['sku']}: {p['title']} (brand {p['brand']}). return_category: {p['return_category']}. "
                f"price_inr: {p['price_inr']}. fulfilled_by: {p['fulfilled_by']}. assured: {p['assured']}. "
                f"warranty_months: {p['warranty_months']}. {p['description']}")
        out.append(Chunk(id=f"products.csv:{p['sku']}", doc_id="CATALOG", doc_title="Product catalog",
                         title=p["title"], text=text, doc_type="catalog", source_file="products.csv",
                         meta=dict(p)))
    return out


def load_all() -> list[Chunk]:
    return load_kb_chunks() + product_chunks(load_products())
