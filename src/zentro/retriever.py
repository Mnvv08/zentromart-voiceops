"""Hybrid retrieval: multilingual dense embeddings (vector index) + BM25, fused with
reciprocal-rank fusion, then light re-ranking by document precedence (KB-00.1).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

import numpy as np

from . import config
from .kb import ALWAYS_ON_DOCS, Chunk, load_all

RRF_K = 20
BM25_W = 2.0
TOKEN_RE = re.compile(r"[a-z0-9ऀ-ॿ]+")
SKU_RE = re.compile(r"\b([A-Z]{3})[-\s]?(\d{4})\b", re.IGNORECASE)

# Customers don't use the documents' words (README). Small bilingual expansion for the keyword side.
EXPANSIONS = {
    "send back": "return reverse pickup", "send it back": "return reverse pickup",
    "give back": "return", "collect": "reverse pickup", "collection": "reverse pickup",
    "money back": "refund", "paisa": "refund", "paise": "refund", "rupaye": "refund amount",
    "stop it": "cancel cancellation", "on its way": "shipped", "late": "late delivery promised date",
    "torn": "damaged defective", "broken": "damaged", "crack": "damaged high-value",
    "not home": "delivery attempts", "not at home": "delivery attempts failed attempt reschedule", "ghar par nahi": "delivery attempts", "घर पर नहीं": "delivery attempts", "nobody came": "missed pickup", "voucher": "compensation",
    "person": "human agent handover", "insaan": "human agent handover", "human": "human agent handover",
    "warranty": "warranty assured", "guarantee": "warranty",
    "वापस": "return", "लौटा": "return", "रिटर्न": "return", "रिफ़ंड": "refund", "रिफंड": "refund",
    "कैंसिल": "cancel", "टूटा": "damaged", "खराब": "damaged defective", "डिलीवरी": "delivery",
    "wapas": "return", "lautana": "return", "kab tak": "timeline", "kitne din": "days window",
    "shuru": "initiated", "dukaan": "store", "दुकान": "store",
    "opened": "unopened seal intact condition", "open kar": "unopened seal intact condition",
    "खोल": "unopened seal intact condition", "tried it": "unused condition",
    "cash on delivery": "cash-on-delivery cod refunds wallet bank transfer",
    "cod": "cash-on-delivery refunds wallet bank transfer",
    "upi": "refund payment method time to reach business days",
    "net banking": "refund payment method time to reach business days",
    "credit card": "refund payment method time to reach business days",
    "card": "refund payment method", "wallet": "refund payment method instant",
    "sale": "festive sale returns", "सेल": "festive sale returns",
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def expand_query(q: str) -> str:
    ql = q.lower()
    extra = [v for k, v in EXPANSIONS.items() if k in ql]
    return q + (" " + " ".join(extra) if extra else "")


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(len(d) for d in docs) / max(len(docs), 1)
        df: dict[str, int] = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [{t: d.count(t) for t in set(d)} for d in docs]

    def scores(self, q: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out


class Embedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # local ONNX model, no API key needed

        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(config.CACHE_DIR / "models"))

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense: float = 0.0
    bm25: float = 0.0

    def to_dict(self) -> dict:
        return {"id": self.chunk.id, "title": self.chunk.title, "doc": self.chunk.doc_id,
                "score": round(self.score, 4), "text": self.chunk.text}


class Retriever:
    def __init__(self):
        all_chunks = load_all()
        self.always_on = [c for c in all_chunks if c.doc_id in ALWAYS_ON_DOCS]
        self.chunks = [c for c in all_chunks if c.doc_id not in ALWAYS_ON_DOCS]
        self.by_id = {c.id: c for c in all_chunks}
        self.products = [c for c in self.chunks if c.doc_type == "catalog"]
        self.bm25 = BM25([tokenize(expand_query(c.embed_text())) for c in self.chunks])
        self.embedder = Embedder(config.EMBED_MODEL)
        self.matrix = self._load_or_build_index()

    # -- vector index (cached on disk) --------------------------------------
    def _load_or_build_index(self) -> np.ndarray:
        texts = [c.embed_text() for c in self.chunks]
        h = hashlib.sha1((config.EMBED_MODEL + "\n".join(texts)).encode()).hexdigest()[:16]
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = config.CACHE_DIR / f"index_{h}.npy"
        if path.exists():
            return np.load(path)
        m = self.embedder.embed(texts)
        np.save(path, m)
        (config.CACHE_DIR / f"index_{h}.json").write_text(json.dumps([c.id for c in self.chunks]))
        return m

    # -- product matching -----------------------------------------------------
    def match_products(self, text: str) -> list[Chunk]:
        found: list[Chunk] = []
        for m in SKU_RE.finditer(text):
            cid = f"products.csv:{m.group(1).upper()}-{m.group(2)}"
            if cid in self.by_id and self.by_id[cid] not in found:
                found.append(self.by_id[cid])
        if found:
            return found
        toks = set(tokenize(text))
        best: list[tuple[float, Chunk]] = []
        for c in self.products:
            base = c.title.split(",")[0]          # drop colour variants
            ptoks = [t for t in tokenize(base) if len(t) > 1]
            if len(ptoks) < 2:
                continue
            hit = sum(1 for t in ptoks if t in toks) / len(ptoks)
            if hit >= 0.75:
                best.append((hit, c))
        best.sort(key=lambda x: -x[0])
        if not best:
            return []
        top = best[0][0]
        # same product name in several colours -> return them all (they share policy facts)
        return [c for s, c in best if s == top][:4]

    # -- search ---------------------------------------------------------------
    def search(self, query: str, k: int | None = None, include_products: bool = True) -> list[Hit]:
        k = k or config.TOP_K
        q_exp = expand_query(query)
        dense = self.matrix @ self.embedder.embed([query])[0]
        bm = self.bm25.scores(tokenize(q_exp))
        rank_d = {i: r for r, i in enumerate(np.argsort(-dense))}
        rank_b = {i: r for r, i in enumerate(np.argsort(-bm))}
        fused = []
        for i, c in enumerate(self.chunks):
            if not include_products and c.doc_type == "catalog":
                continue
            s = 1 / (RRF_K + rank_d[i]) + (BM25_W / (RRF_K + rank_b[i]) if bm[i] > 0 else 0)
            # products only surface through strong matches; marketing/superseded slightly down-weighted
            if c.doc_type == "catalog":
                s *= 0.6
            s *= {1: 1.0, 2: 1.0, 3: 0.97, 4: 0.93, 5: 0.85}.get(min(c.precedence, 5), 0.85)
            fused.append(Hit(c, s, float(dense[i]), float(bm[i])))
        fused.sort(key=lambda h: -h.score)
        return fused[:k]

    def retrieve_for_turn(self, message: str, history_user: list[str] | None = None, k: int | None = None) -> list[Hit]:
        """Retrieve for the current message; also use the previous user message for follow-ups."""
        k = k or config.TOP_K
        hits = {h.chunk.id: h for h in self.search(message, k)}
        if history_user:
            for h in self.search(history_user[-1] + " " + message, k=max(2, k // 2)):
                if h.chunk.id not in hits:
                    h.score *= 0.9
                    hits[h.chunk.id] = h
        ordered = sorted(hits.values(), key=lambda h: -h.score)[:k]
        # Two-source join: a mentioned product brings its catalog row plus the policy
        # sections that define its catalog codes (return_category -> RET-2, assured -> WAR-2).
        extra: list[Hit] = []
        for c in self.match_products(message + " " + " ".join((history_user or [])[-1:])):
            extra.append(Hit(c, 1.0))
            for linked in self.linked_policy_sections(c):
                extra.append(Hit(linked, 0.5))
        seen = set()
        out = []
        for h in extra + ordered:
            if h.chunk.id not in seen:
                seen.add(h.chunk.id)
                out.append(h)
        return out

    def linked_policy_sections(self, product: Chunk) -> list[Chunk]:
        cat = product.meta.get("return_category", "")
        assured = product.meta.get("assured", "") == "True"
        out = []
        for c in self.chunks:
            if c.doc_type == "catalog" or c.status == "superseded":
                continue
            if cat and f"`{cat}`" in c.text:
                out.append(c)
            elif assured and "`assured = true`" in c.text:
                out.append(c)
        return out
