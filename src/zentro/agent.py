"""ZentroMart support agent.

mode="baseline"  -> Step 1: LLM + system prompt only (no documents, no tools)
mode="agent"     -> Steps 2-4: RAG over kb/ + product catalog, conversation memory,
                    identity-checked order lookup, ticket creation and date tools.
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field

from . import config, llm
from .dates import business_days_between, date_calc, date_hints, find_dates, today_line
from .prompts import AGENT_SYSTEM, BASELINE_SYSTEM, CHANNEL_CHAT, CHANNEL_VOICE, OUTPUT_FORMAT
from .textnorm import ORDER_ID_RE, for_the_ear, spoken_digits_to_numbers

# The dataset's mock tools (lookup_order, create_ticket) are imported unchanged from the dataset folder.
sys.path.insert(0, str(config.DATASET_DIR))
from tools import create_ticket as _create_ticket  # noqa: E402
from tools import lookup_order as _lookup_order  # noqa: E402

_RETRIEVER = None


def get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        from .retriever import Retriever

        _RETRIEVER = Retriever()
    return _RETRIEVER


TOOLS = [
    {"type": "function", "function": {
        "name": "search_knowledge_base",
        "description": "Search ZentroMart's policy documents, help articles, FAQs and product catalog. Use it when the "
                       "sections already provided do not answer the question. Write the query in English.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "date_calc",
        "description": "Exact date arithmetic. add_business_days (refund/logistics timelines: Mon-Sat, skips holidays), "
                       "add_calendar_days (return windows), business_days_between / calendar_days_between "
                       "(end_date defaults to today).",
        "parameters": {"type": "object", "properties": {
            "operation": {"type": "string", "enum": ["add_business_days", "add_calendar_days",
                                                     "business_days_between", "calendar_days_between"]},
            "start_date": {"type": "string", "description": "YYYY-MM-DD"},
            "days": {"type": "integer"},
            "end_date": {"type": "string", "description": "YYYY-MM-DD"}},
            "required": ["operation", "start_date"]}}},
    {"type": "function", "function": {
        "name": "create_ticket",
        "description": "Hand the case over to a human colleague by creating a support ticket. Required before any "
                       "handover (ESC-2). The current order ID is attached automatically.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "description": "Issue, detected intent, order ID, what the customer said, "
                                                          "what you checked, reason for escalation. No phone digits/pincode."},
            "reason": {"type": "string", "description": "short code, e.g. high_value_damage, human_requested, "
                                                         "repeat_contact, refund_overdue, unrecognised_order, legal, safety"},
            "intent": {"type": "string"}},
            "required": ["summary", "reason"]}}},
    {"type": "function", "function": {
        "name": "lookup_order",
        "description": "Identity-checked order lookup. Normally done automatically; call it only if the customer gave "
                       "an order ID or verification value that the system missed.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"}, "verification_value": {"type": "string"}},
            "required": ["order_id"]}}},
]

DECISIONS = ("answer", "clarify", "escalate", "refuse")
HIDE_FIELDS = {"delivery_pincode", "found", "verified", "customer_id"}


@dataclass
class OrderState:
    order_id: str
    verified: bool = False
    failed: int = 0
    data: dict | None = None
    secrets: set = field(default_factory=set)


@dataclass
class TurnResult:
    user_text: str
    reply: str
    spoken: str
    decision: str
    sources: list
    retrieved: list
    tool_calls: list
    latency_s: float
    raw: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class Agent:
    def __init__(self, mode: str = "agent", channel: str = "chat"):
        assert mode in ("agent", "baseline")
        self.mode = mode
        self.channel = channel
        self.history: list[dict] = []
        self.orders: dict[str, OrderState] = {}
        self.current: str | None = None
        self.tickets: list[dict] = []
        self.turns = 0
        self._turn_dates: list = []
        self.retriever = get_retriever() if mode == "agent" else None

    # ------------------------------------------------------------------ helpers
    def _log_tool(self, calls: list, name: str, args: dict, result: dict):
        # README: never expose verification secrets in logs -> mask the value the caller gave
        safe_args = {k: ("*" * len(str(v)) if k == "verification_value" else v) for k, v in args.items()}
        calls.append({"tool": name, "args": safe_args, "result": _short(result)})

    def _do_lookup(self, order_id: str, value: str | None, calls: list) -> dict:
        oid = re.sub(r"\D", "", str(order_id))
        st = self.orders.setdefault(oid, OrderState(oid))
        self.current = oid
        if value is None:
            res = _lookup_order(oid)
            self._log_tool(calls, "lookup_order", {"order_id": oid}, res)
            return res
        if st.verified:
            return {"verified": True, "note": "already verified"}
        if st.failed >= 2:  # IDV-3: one retry only
            res = {"verified": False, "locked": True, "message": "Verification failed twice; do not share order details."}
            self._log_tool(calls, "lookup_order(blocked)", {"order_id": oid}, res)
            return res
        val = re.sub(r"\D", "", str(value))
        res = _lookup_order(oid, verification_value=val)
        self._log_tool(calls, "lookup_order", {"order_id": oid, "verification_value": val}, res)
        st.secrets.add(val)
        if res.get("verified"):
            st.verified, st.data = True, res
        else:
            st.failed += 1
        return {k: v for k, v in res.items() if k in ("verified", "found", "message", "verification_required")}

    def _auto_tools(self, text: str, calls: list):
        """Deterministic part of the verification flow: detect order IDs and verification values."""
        norm = spoken_digits_to_numbers(text)
        oid = None
        m = ORDER_ID_RE.search(norm)
        if m:
            oid = m.group(1) + m.group(2)
            if oid not in self.orders:
                self._do_lookup(oid, None, calls)
            self.current = oid
            norm = norm.replace(m.group(0), " ")
        st = self.orders.get(self.current) if self.current else None
        if st and not st.verified and st.failed < 2:
            runs = re.findall(r"(?<!\d)(\d{6}|\d{4})(?!\d)", norm)
            runs = [r for r in runs if not (len(r) == 4 and r.startswith(("19", "20")) and
                                            re.search(r"(january|february|march|april|may|june|july|august|"
                                                      r"september|october|november|december)\s*" + r, norm, re.I))]
            if runs:
                self._do_lookup(st.order_id, runs[0], calls)

    def _order_block(self) -> str:
        lines = []
        if not self.orders:
            lines.append("No order ID has been given in this conversation. For general policy questions none is needed. "
                         "If the customer asks about a specific order, ask for the order ID.")
        for oid, st in self.orders.items():
            spoken = f"{oid[:4]} {oid[4:]}"
            cur = " (current)" if oid == self.current else ""
            if st.verified and st.data:
                lines.append(f"Order {spoken}{cur}: VERIFIED. You may share its details with this customer.")
                lines.append(json.dumps(self._present_order(st.data), ensure_ascii=False))
            elif st.failed >= 2:
                lines.append(f"Order {spoken}{cur}: verification FAILED TWICE. Do NOT share any detail of this order "
                             "(status, dates, items, amounts, pincode). Say you can't share order details, offer general "
                             "policy help, and suggest contacting support from the registered phone number.")
            elif st.failed == 1:
                lines.append(f"Order {spoken}{cur}: verification attempt 1 FAILED (value did not match). Tell the "
                             "customer it did not match and ask them to try once more (one retry allowed): last 4 "
                             "digits of the registered phone OR the delivery pincode. Share no order details and never "
                             "hint at the correct value.")
            else:
                lines.append(f"Order {spoken}{cur}: NOT VERIFIED yet. Before sharing anything about it, ask for ONE of: "
                             "the last 4 digits of the registered phone number, or the delivery pincode of this order.")
        dates = list(self._turn_dates)
        for st in self.orders.values():
            if st.verified and st.data:
                for k in ("delivered_on", "return_refund_initiated_on", "cancellation_refund_initiated_on",
                          "rto_refund_initiated_on", "return_pickup_on", "shipped_on"):
                    if st.data.get(k):
                        dates.append(str(st.data[k])[:10])
        if dates:
            lines.append("DATE HELPER (exact, already computed; use these instead of counting):\n"
                         + date_hints(list(dict.fromkeys(dates))))
        if self.tickets:
            lines.append("Tickets already created in this conversation: " + "; ".join(
                f"{t['ticket_id']} ({t['reason']})" for t in self.tickets))
        return "\n".join(lines)

    def _present_order(self, data: dict) -> dict:
        d = {k: v for k, v in data.items() if v not in (None, "") and k not in HIDE_FIELDS}
        prods = {c.meta["sku"]: c.meta for c in self.retriever.products} if self.retriever else {}
        items = []
        for it in data.get("items", []):
            p = prods.get(it.get("sku"), {})
            items.append({"line_no": it.get("line_no"), "sku": it.get("sku"), "title": it.get("title"),
                          "return_category": it.get("return_category"), "qty": it.get("qty"),
                          "unit_price_inr": it.get("unit_price_inr"), "fulfilled_by": p.get("fulfilled_by"),
                          "assured": p.get("assured"), "warranty_months": p.get("warranty_months")})
        d["items"] = items
        helpers = {}
        today = config.TODAY
        if data.get("delivered_on"):
            from .dates import parse
            helpers["calendar_days_since_delivery"] = (parse(today) - parse(data["delivered_on"])).days
        for k in ("return_refund_initiated_on", "cancellation_refund_initiated_on", "rto_refund_initiated_on"):
            if data.get(k):
                helpers[f"business_days_since_{k}"] = business_days_between(data[k], today)
        if helpers:
            d["computed_helpers"] = helpers
        return d

    def _tool_exec(self, name: str, args: dict, calls: list) -> dict:
        if name == "search_knowledge_base":
            hits = self.retriever.search(args.get("query", ""), k=5)
            calls.append({"tool": "search_knowledge_base", "args": args, "result": [h.chunk.id for h in hits]})
            self._extra_hits.extend(hits)
            return {"sections": [h.chunk.render() for h in hits]}
        if name == "date_calc":
            res = date_calc(args.get("operation", ""), args.get("start_date", ""), int(args.get("days") or 0),
                            args.get("end_date", "") or "")
            self._log_tool(calls, "date_calc", args, res)
            return res
        if name == "create_ticket":
            reason = str(args.get("reason", "")).lower()
            # Block only handovers whose reason is "the documents don't cover it" (offer a colleague instead).
            blocked = ("no_info", "noinfo", "not_covered", "unanswer", "information", "unknown", "general",
                       "out_of_scope", "not_found", "no_policy", "policy_not", "faq", "question")
            if any(b in reason for b in blocked) and "human" not in reason:
                calls.append({"tool": "create_ticket(blocked)", "args": args, "result": {"created": False}})
                return {"created": False, "error": "No ESC-1 handover rule applies. Do not create a ticket: answer from "
                        "the knowledge base (search it if needed), or say you don't have the information and OFFER "
                        "to connect a human colleague."}
            summary = str(args.get("summary", ""))
            for st in self.orders.values():           # never leak verification secrets (IDV-4)
                for s in st.secrets:
                    summary = summary.replace(s, "[redacted]")
            oid = self.current if self.current else None
            res = _create_ticket(summary=summary, order_id=oid, reason=args.get("reason"),
                                 intent=args.get("intent"))
            self._log_tool(calls, "create_ticket", {"summary": summary, "order_id": oid,
                                                    "reason": args.get("reason")}, res)
            if res.get("created"):
                self.tickets.append({"ticket_id": res["ticket_id"], "reason": args.get("reason", "")})
                self._ticket_this_turn = True
            return res
        if name == "lookup_order":
            return self._do_lookup(args.get("order_id", ""), args.get("verification_value") or None, calls)
        return {"error": f"unknown tool {name}"}

    # ------------------------------------------------------------------ main
    def respond(self, user_text: str, alt_transcripts: dict | None = None) -> TurnResult:
        t0 = time.time()
        calls: list = []
        self._extra_hits = []
        self._ticket_this_turn = False
        self._turn_dates = find_dates(" ".join([user_text or "", *((alt_transcripts or {}).values())]))
        # spoken numbers -> digits ("ele zero zero one two" -> "ele 0012", "two zero two six" -> "2026")
        user_text = spoken_digits_to_numbers((user_text or "").strip())
        if alt_transcripts:
            alt_transcripts = {k: spoken_digits_to_numbers(v or "") for k, v in alt_transcripts.items()}

        # Message the LLM sees (voice: may include both STT hypotheses)
        llm_user = user_text
        if alt_transcripts:
            parts = [f"[{k} transcript]: {v}" for k, v in alt_transcripts.items() if v and v.strip()]
            llm_user = ("(Customer spoke; speech-to-text hypotheses below. Use the one that makes sense and reply in "
                        "the language the customer actually spoke.)\n" + "\n".join(parts)) if parts else ""

        if self.mode == "baseline":
            system = BASELINE_SYSTEM
            retrieved = []
            if self.turns > 0:
                system += "\n(You already introduced yourself earlier in this conversation.)"
            messages = [{"role": "system", "content": system}, *self.history[-10:], {"role": "user", "content": llm_user}]
            msg = llm.chat(messages)
            raw = msg.content or ""
        else:
            self._before = {k: (v.verified, v.failed) for k, v in self.orders.items()}
            self._auto_tools(" ".join([user_text, *((alt_transcripts or {}).values())]), calls)
            hist_users = [m["content"] for m in self.history if m["role"] == "user"]
            query = user_text if not alt_transcripts else " ".join(v for v in alt_transcripts.values() if v)
            hits = self.retriever.retrieve_for_turn(query, hist_users)
            retrieved = hits
            system = AGENT_SYSTEM.format(
                today=today_line(),
                intro=("This is your FIRST reply in the conversation: begin by introducing yourself as ZentroMart's "
                       "AI support assistant (one short clause), then help." if self.turns == 0 else
                       "You already introduced yourself; do not repeat the introduction."),
                channel=CHANNEL_VOICE if self.channel == "voice" else CHANNEL_CHAT,
                output_format=OUTPUT_FORMAT,
                always_on="\n\n".join(c.render() for c in self.retriever.always_on if c.doc_id != "KB-00"
                                      or c.id in ("KB-00.1", "KB-00.3")),
                retrieved="\n\n".join(h.chunk.render() for h in hits),
                order_block=self._order_block(),
            )
            from .voice import reply_language
            lang = reply_language(user_text) if not (alt_transcripts and self._spoke_hindi(alt_transcripts)) else "hi"
            lang_note = {"en": "English", "hi": "Hindi (Devanagari script)",
                         "hinglish": "Hinglish (Hindi in Latin letters)"}[lang]
            system += f"\n\nTHIS TURN: the customer's latest message is in {lang_note}. Reply in {lang_note}."
            messages = [{"role": "system", "content": system}, *self.history[-10:],
                        {"role": "user", "content": llm_user or "(silence / nothing recognised)"}]
            raw = self._loop(messages, calls)

        reply, decision, sources = parse_output(raw)
        if self.turns == 0:
            reply = ensure_intro(reply)
        valid_ids = self.retriever.by_id if self.retriever else {}
        sources = [s for s in sources if s in valid_ids] if self.mode == "agent" else []
        if self.mode == "agent":
            decision, sources = self._postprocess(decision, sources, calls, getattr(self, "_before", {}))
        all_hits = list(retrieved) + list(getattr(self, "_extra_hits", []))
        seen, retrieved_out = set(), []
        for h in all_hits:
            if h.chunk.id not in seen:
                seen.add(h.chunk.id)
                retrieved_out.append(h.to_dict())

        self.history += [{"role": "user", "content": llm_user or user_text},
                         {"role": "assistant", "content": reply}]
        self.turns += 1
        return TurnResult(user_text=user_text, reply=reply, spoken=for_the_ear(reply), decision=decision,
                          sources=sources, retrieved=retrieved_out, tool_calls=calls,
                          latency_s=round(time.time() - t0, 2), raw=raw)

    @staticmethod
    def _spoke_hindi(alt: dict) -> bool:
        return any("Hindi" in k for k in alt)

    def _postprocess(self, decision: str, sources: list, calls: list, before: dict) -> tuple[str, list]:
        """Deterministic decision labels / source IDs for the steps the code itself performed."""
        st = self.orders.get(self.current) if self.current else None
        lookups = [c for c in calls if c["tool"].startswith("lookup_order")]
        extra = []
        if self.turns == 0:
            extra.append("STY-1")
        if self._ticket_this_turn:
            decision = "escalate"
            extra += ["ESC-1", "ESC-2"]
        elif st and lookups:
            prev_failed = before.get(st.order_id, (False, 0))[1]
            if st.verified and not before.get(st.order_id, (False, 0))[0]:
                extra.append("IDV-2")
                if decision == "clarify":
                    decision = "answer"
            elif st.failed >= 2 and prev_failed < 2:
                decision, _ = "refuse", extra.append("IDV-3")
            elif not st.verified:
                decision = "clarify"
                extra.append("IDV-3" if st.failed == 1 else "IDV-2")
        for e in extra:
            if e not in sources:
                sources.append(e)
        return decision, sources

    def _loop(self, messages: list[dict], calls: list) -> str:
        nudged = False
        # A message that only answers the verification question is not a new issue: no handover on that turn.
        last_user = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
        only_verification = bool(re.fullmatch(r"[\W\w]{0,60}", last_user)) and bool(
            re.search(r"(?<!\d)(\d{4}|\d{6})(?!\d)", spoken_digits_to_numbers(last_user))) and any(
            c["tool"] == "lookup_order" and "verification_value" in c["args"] for c in calls)
        tools = [t for t in TOOLS if not (only_verification and t["function"]["name"] == "create_ticket")]
        for _ in range(6):
            msg = llm.chat(messages, tools=tools)
            if msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content or "",
                                 # keep provider extras (Gemini needs its thought_signature echoed back)
                                 "tool_calls": [{**tc.model_dump(exclude_none=True),
                                                 "function": {"name": tc.function.name,
                                                              "arguments": tc.function.arguments or "{}"}}
                                                for tc in msg.tool_calls]})
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}") or {}
                    except json.JSONDecodeError:
                        args = {}
                    res = self._tool_exec(tc.function.name, args, calls)
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": json.dumps(res, ensure_ascii=False, default=str)})
                # order data may have changed (lookup) -> refresh the order block in the system prompt
                messages[0]["content"] = re.sub(r"(=+ ORDER STATUS =+\n)[\s\S]*$",
                                                lambda m: m.group(1) + self._order_block(), messages[0]["content"])
                continue
            raw = msg.content or ""
            _, decision, _ = parse_output(raw)
            claims_ticket = bool(re.search(r"ticket|टिकट|colleague will|सहयोगी|priority resolution", raw, re.I))
            if not nudged and not self._ticket_this_turn and (decision == "escalate" or claims_ticket) \
                    and not self.tickets:
                nudged = True
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                                 "[SYSTEM CHECK] Your reply hands over to a human / mentions a ticket, but create_ticket "
                                 "has not been called. If a handover is needed, call create_ticket now and then write "
                                 "the final reply. If no handover is needed, rewrite the reply without claiming a "
                                 "ticket (you may offer to connect them). Keep the same output format."})
                continue
            return raw
        return raw if "raw" in locals() else "Sorry, something went wrong. Let me connect you to a colleague.\nDECISION: escalate\nSOURCES: none"


INTRO = {
    "hi": "नमस्ते, मैं ZentroMart की AI सपोर्ट असिस्टेंट हूँ। ",
    "hinglish": "Namaste, main ZentroMart ki AI support assistant hoon. ",
    "en": "Hi, I'm ZentroMart's AI support assistant. ",
}


def ensure_intro(reply: str) -> str:
    """STY-1: the first reply must say it is an AI assistant (guaranteed in code, not left to the model)."""
    if re.search(r"\bAI\b|artificial intelligence|एआई|कृत्रिम", reply, re.I):
        return reply
    from .voice import reply_language

    return INTRO[reply_language(reply)] + reply


def parse_output(raw: str) -> tuple[str, str, list]:
    raw = (raw or "").strip()
    decision = ""
    sources: list = []
    m = re.search(r"DECISION\s*:\s*\**\s*([A-Za-z]+)", raw, re.I)
    if m and m.group(1).lower() in DECISIONS:
        decision = m.group(1).lower()
    m2 = re.search(r"SOURCES\s*:\s*(.*)", raw, re.I)
    if m2:
        txt = m2.group(1).strip().strip("*")
        if txt.lower() not in ("none", "-", ""):
            sources = [s.strip().strip("[]`*. ") for s in re.split(r"[,;]", txt) if s.strip()]
    reply = re.split(r"\n?\s*\**\s*(?:DECISION|SOURCES)\s*:", raw, flags=re.I)[0].strip()
    for a, b in (("\u202f", " "), ("\u00a0", " "), ("\u2011", "-"), ("\u2010", "-"), ("\u200b", "")):
        reply = reply.replace(a, b)
    reply = re.sub(r"\n?-{3,}\s*$", "", reply).strip()
    reply = re.sub(r"\s*\[(?:[A-Z]{2,5}[vH]?\d?-[\w.]+|products\.csv:[A-Z]{3}-\d{4})(?:\s*[,;]\s*[^\]]+)?\]", "", reply)
    if not decision:
        decision = "clarify" if reply.rstrip().endswith("?") else "answer"
    return reply, decision, sources


def _short(res):
    if isinstance(res, dict):
        out = {k: v for k, v in res.items() if k not in ("items", "timeline", "delivery_pincode", "customer_id")}
        if len(json.dumps(out, default=str)) > 600:
            out = {k: out[k] for k in list(out)[:12]}
        return out
    return res
