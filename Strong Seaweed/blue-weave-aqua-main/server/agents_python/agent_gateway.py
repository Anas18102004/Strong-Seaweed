import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="[agent-gateway] %(levelname)s %(message)s")
logger = logging.getLogger("agent_gateway")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _masked_secret(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:6]}...{value[-4:]}"


# If Groq returns policy block (e.g., 1010), pause Groq attempts for this many seconds.
GROQ_BLOCK_COOLDOWN_S = int(_env("GROQ_BLOCK_COOLDOWN_S", "900") or "900")
_groq_blocked_until_epoch = 0.0
DEFAULT_GROQ_API_KEY = ""


LANGCHAIN_AVAILABLE = False
LANGGRAPH_AVAILABLE = False
CREWAI_AVAILABLE = False

try:
    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    from langchain_openai import ChatOpenAI  # type: ignore

    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

GROQ_SDK_AVAILABLE = False
try:
    from openai import OpenAI as OpenAIClient  # type: ignore

    GROQ_SDK_AVAILABLE = True
except Exception:
    GROQ_SDK_AVAILABLE = False

try:
    import langgraph  # type: ignore  # noqa: F401

    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

try:
    import crewai  # type: ignore  # noqa: F401

    CREWAI_AVAILABLE = True
except Exception:
    CREWAI_AVAILABLE = False


AGENT_SYSTEM = {
    "cultivation": (
        "Role: Senior Seaweed Cultivation Planner.\n"
        "Mission: Turn user context into a practical cultivation plan that can be executed by a farm team.\n"
        "Scope: species choice, seed/line setup, deployment method, maintenance cadence, harvest windows.\n"
        "Reasoning rules:\n"
        "- Use only information present in the request/context; if key inputs are missing, state assumptions.\n"
        "- Prefer low-regret, field-practical guidance over academic theory.\n"
        "- Mention risk-sensitive constraints (waves, salinity shocks, contamination, biofouling).\n"
        "Output format:\n"
        "1) Recommendation summary (2-3 lines)\n"
        "2) Step-by-step plan (week/phase based)\n"
        "3) Assumptions and missing data\n"
        "4) Immediate next 3 actions\n"
        "Style: concise, operational, no fluff."
    ),
    "risk": (
        "Role: Senior Marine Risk Analyst for seaweed operations.\n"
        "Mission: Identify, rank, and mitigate operational and environmental risk.\n"
        "Scope: monsoon/cyclone exposure, wave energy, salinity variability, disease/fouling, logistics and downtime.\n"
        "Reasoning rules:\n"
        "- Prioritize by impact x likelihood.\n"
        "- Give clear thresholds/triggers for pause/resume decisions when possible.\n"
        "- Distinguish known facts vs assumptions.\n"
        "Output format:\n"
        "1) Top risks ranked (critical/high/medium)\n"
        "2) Why each risk matters\n"
        "3) Mitigation playbook\n"
        "4) Monitoring checklist (daily/weekly)\n"
        "Style: direct, risk-first, action-oriented."
    ),
    "yield": (
        "Role: Senior Yield Optimization Specialist for seaweed farms.\n"
        "Mission: Increase biomass output and quality using measurable interventions.\n"
        "Scope: spacing, seeding density, cleaning cadence, harvest timing, batch tracking.\n"
        "Reasoning rules:\n"
        "- Recommend changes with measurable KPIs.\n"
        "- Focus on interventions that can be run as short field experiments.\n"
        "- Avoid generic advice; tie actions to expected yield impact.\n"
        "Output format:\n"
        "1) Yield diagnosis\n"
        "2) Intervention plan (what to change)\n"
        "3) KPI table (metric, baseline, target, frequency)\n"
        "4) 14-day optimization sprint\n"
        "Style: quantitative and practical."
    ),
    "site": (
        "Role: Senior Site Suitability and Expansion Strategist.\n"
        "Mission: Compare candidate locations and recommend where to expand safely.\n"
        "Scope: depth, wave regime, salinity stability, shore distance, conflict constraints, access/logistics.\n"
        "Reasoning rules:\n"
        "- Evaluate trade-offs explicitly.\n"
        "- Flag high-uncertainty factors and what data is required to resolve them.\n"
        "- Prefer phased validation (pilot first, then scale).\n"
        "Output format:\n"
        "1) Site ranking with rationale\n"
        "2) Trade-offs and red flags\n"
        "3) Data gaps\n"
        "4) Pilot plan and go/no-go criteria\n"
        "Style: structured, comparative, decision-ready."
    ),
    "market": (
        "Role: Senior Seaweed Market Strategy Advisor.\n"
        "Mission: Optimize harvest timing, quality grade, and sell strategy for margin and reliability.\n"
        "Scope: demand windows, pricing signals, quality requirements, inventory and logistics.\n"
        "Reasoning rules:\n"
        "- Separate tactical (this cycle) from strategic (next 3-6 months) actions.\n"
        "- Balance price opportunity vs operational risk.\n"
        "- Provide alternatives for conservative vs aggressive sell strategies.\n"
        "Output format:\n"
        "1) Market-read summary\n"
        "2) Sell timing recommendation\n"
        "3) Quantity/quality strategy\n"
        "4) Contingency plan\n"
        "Style: commercial, pragmatic, concrete."
    ),
    "copilot": (
        "Role: BlueWeave Operations Copilot (orchestrator).\n"
        "Mission: Understand user intent, route to the best specialist lens, and deliver one cohesive execution plan.\n"
        "Routing policy:\n"
        "- Cultivation setup/questions -> cultivation lens\n"
        "- Hazard/weather/operational uncertainty -> risk lens\n"
        "- Output improvement/performance -> yield lens\n"
        "- New location comparison -> site lens\n"
        "- Selling/pricing/timing -> market lens\n"
        "Reasoning rules:\n"
        "- If user question is broad, provide a phased plan across relevant lenses.\n"
        "- Keep answers beginner-friendly but technically correct.\n"
        "- Ask only essential follow-up questions; otherwise proceed with explicit assumptions.\n"
        "Response format policy:\n"
        "- For greeting/small-talk/very short messages (e.g., hi, hello, ok): reply naturally in 1-2 short lines.\n"
        "- Do NOT use numbered sections for small-talk.\n"
        "- Use structured sections only for real planning/analysis requests.\n"
        "Style: clear, decisive, execution-first."
    ),
}


@dataclass
class CacheItem:
    expires_at: float
    value: Dict[str, Any]


class TTLCache:
    def __init__(self, ttl_seconds: int = 90, max_items: int = 1024) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._items: Dict[str, CacheItem] = {}
        self._lock = threading.Lock()

    def _prune(self) -> None:
        now = time.time()
        expired = [k for k, v in self._items.items() if v.expires_at <= now]
        for key in expired:
            self._items.pop(key, None)
        if len(self._items) > self.max_items:
            for key in list(self._items.keys())[: len(self._items) - self.max_items]:
                self._items.pop(key, None)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._items.get(key)
            if not item:
                return None
            if item.expires_at <= time.time():
                self._items.pop(key, None)
                return None
            return dict(item.value)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._prune()
            self._items[key] = CacheItem(expires_at=time.time() + self.ttl_seconds, value=dict(value))


class AgentRequest(BaseModel):
    agent: str
    question: str
    context: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    question: str
    routedAgent: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class VoiceRequest(BaseModel):
    question: str
    routedAgent: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    locale: Optional[str] = None


def _log_provider_bootstrap() -> None:
    groq_env = _env("GROQ_API_KEY")
    groq_effective = groq_env or DEFAULT_GROQ_API_KEY
    groq_source = "env" if groq_env else "default"
    logger.info(
        "Provider bootstrap: Groq key source=%s key=%s model=%s",
        groq_source,
        _masked_secret(groq_effective),
        _env("GROQ_MODEL", "openai/gpt-oss-20b"),
    )


class AgentOrchestrator:
    def __init__(self) -> None:
        self.cache = TTLCache(
            ttl_seconds=int(_env("AGENT_CACHE_TTL_SECONDS", "90") or "90"),
            max_items=int(_env("AGENT_CACHE_MAX_ITEMS", "1024") or "1024"),
        )

    @staticmethod
    def route_question(question: str) -> str:
        q = (question or "").lower()
        weighted_keywords: List[Tuple[str, Dict[str, int]]] = [
            ("risk", {"risk": 2, "cyclone": 4, "storm": 4, "monsoon": 4, "hazard": 2}),
            ("yield", {"yield": 3, "harvest": 3, "growth": 2, "biomass": 3, "productivity": 2}),
            ("site", {"site": 3, "location": 3, "expand": 2, "where": 2, "coast": 1}),
            ("market", {"market": 3, "price": 3, "sell": 2, "demand": 2, "buyer": 2}),
            ("cultivation", {"kappaphycus": 3, "gracilaria": 3, "farm": 2, "cultivation": 3}),
        ]
        scores: Dict[str, int] = {k: 0 for k in AGENT_SYSTEM.keys()}
        for agent, keymap in weighted_keywords:
            for token, weight in keymap.items():
                if token in q:
                    scores[agent] += weight
        best_agent = max(scores, key=scores.get)
        return best_agent if scores[best_agent] > 0 else "copilot"

    @staticmethod
    def stack_for(agent: str) -> List[str]:
        if agent in {"cultivation", "site"}:
            return ["LangChain", "Python Gateway"]
        if agent in {"risk", "copilot"}:
            return ["LangGraph", "Python Gateway"]
        return ["CrewAI", "Python Gateway"]

    @staticmethod
    def _is_smalltalk(question: str) -> bool:
        q = (question or "").strip().lower()
        if not q:
            return True
        smalltalk_tokens = {
            "hi",
            "hello",
            "hey",
            "yo",
            "ok",
            "okay",
            "thanks",
            "thank you",
            "sup",
            "hii",
            "hola",
        }
        if q in smalltalk_tokens:
            return True
        words = q.split()
        if len(words) <= 3 and any(tok in q for tok in ["hi", "hello", "hey"]):
            return True
        return False

    @staticmethod
    def _cache_key(mode: str, agent: str, question: str, context: Optional[Dict[str, Any]]) -> str:
        payload = {
            "mode": mode,
            "agent": agent,
            "question": question.strip().lower(),
            "context": context or {},
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _context_block(context: Optional[Dict[str, Any]]) -> str:
        if not context:
            return "{}"
        try:
            compact = json.dumps(context, sort_keys=True)
        except Exception:
            compact = str(context)
        return compact[:1400]

    @staticmethod
    def _fallback_answer(
        agent: str,
        question: str,
        context: Optional[Dict[str, Any]],
        failure_reason: Optional[str] = None,
    ) -> str:
        base = (
            f"Live AI model is unavailable right now for {agent}. "
            "Please verify provider key/model/quota and retry."
        )
        if failure_reason:
            return f"{base}\n\nReason: {failure_reason[:220]}"
        return (
            base
        )

    @staticmethod
    def _groq_answer(
        system: str,
        question: str,
        context: Optional[Dict[str, Any]],
        is_voice: bool,
    ) -> Tuple[Optional[str], Optional[str]]:
        global _groq_blocked_until_epoch
        env_api_key = _env("GROQ_API_KEY")
        api_key = env_api_key or DEFAULT_GROQ_API_KEY
        if not api_key:
            return None, "groq_not_configured"
        if not GROQ_SDK_AVAILABLE:
            return None, "groq_sdk_not_available"

        now = time.time()
        if _groq_blocked_until_epoch > now:
            remaining = int(_groq_blocked_until_epoch - now)
            return None, f"groq_temporarily_blocked_{remaining}s"

        model = _env("GROQ_MODEL", "openai/gpt-oss-20b")
        timeout_s = float(_env("GROQ_TIMEOUT_S", "10") or "10")

        style = (
            "Respond for voice output: short sentences, no markdown tables, no long lists."
            if is_voice
            else "Respond concise and practical. Use short bullet points when needed."
        )

        def _call_with_key(key: str) -> Tuple[Optional[str], Optional[str]]:
            client = OpenAIClient(
                api_key=key,
                base_url=_env("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                timeout=timeout_s,
            )
            prompt = (
                f"{system} {style}\n\n"
                f"Question: {question}\n"
                f"Context JSON: {AgentOrchestrator._context_block(context)}\n"
                "Return actionable guidance with assumptions called out."
            )
            rsp = client.responses.create(
                input=prompt,
                model=model,
            )
            text = (getattr(rsp, "output_text", None) or "").strip()
            if not text:
                text = str(rsp).strip()
            return (text if text else None), (None if text else "groq_empty_text")

        try:
            return _call_with_key(api_key)
        except Exception as exc:
            detail = str(exc)
            logger.error("Groq SDK error: %s", detail[:500])
            if "invalid_api_key" in detail.lower():
                can_retry_default = bool(env_api_key) and env_api_key != DEFAULT_GROQ_API_KEY
                if can_retry_default:
                    logger.warning("Groq env key invalid; retrying once with default Groq key.")
                    try:
                        return _call_with_key(DEFAULT_GROQ_API_KEY)
                    except Exception as retry_exc:
                        retry_detail = str(retry_exc)
                        logger.error("Groq retry with default key failed: %s", retry_detail[:500])
                        return None, f"groq_error: {retry_detail[:260]}"
            if "1010" in detail:
                _groq_blocked_until_epoch = time.time() + GROQ_BLOCK_COOLDOWN_S
                logger.warning(
                    "Groq blocked by provider policy (1010). Skipping Groq for %ss and falling back to next provider.",
                    GROQ_BLOCK_COOLDOWN_S,
                )
            if "timed out" in detail.lower():
                return None, "groq_timeout"
            return None, f"groq_error: {detail[:260]}"

    @staticmethod
    def _openrouter_answer(
        system: str,
        question: str,
        context: Optional[Dict[str, Any]],
        is_voice: bool,
    ) -> Tuple[Optional[str], Optional[str]]:
        api_key = _env("OPENROUTER_API_KEY")
        if not api_key:
            return None, "openrouter_not_configured"

        model = _env("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
        timeout_s = float(_env("OPENROUTER_TIMEOUT_S", "12") or "12")
        site_url = _env("OPENROUTER_SITE_URL", "http://localhost")
        app_name = _env("OPENROUTER_APP_NAME", "BlueWeave")

        style = (
            "Respond for voice output: short sentences, no markdown tables, no long lists."
            if is_voice
            else "Respond concise and practical. Use short bullet points when needed."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": f"{system} {style}"},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Context JSON: {AgentOrchestrator._context_block(context)}\n"
                        "Return actionable guidance with assumptions called out."
                    ),
                },
            ],
            "temperature": 0.1,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": site_url,
                "X-Title": app_name,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            parsed = json.loads(raw)
            choices = parsed.get("choices") or []
            if not choices:
                return None, "openrouter_no_choices"
            msg = (choices[0] or {}).get("message") or {}
            content = msg.get("content")
            if isinstance(content, list):
                text = "\n".join(str(x.get("text", "")).strip() for x in content if isinstance(x, dict) and x.get("text"))
            else:
                text = str(content or "").strip()
            return (text if text else None), (None if text else "openrouter_empty_text")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = str(exc)
            logger.error("OpenRouter HTTP error %s: %s", exc.code, detail[:500])
            return None, f"openrouter_http_{exc.code}: {detail[:260]}"
        except urllib.error.URLError as exc:
            logger.error("OpenRouter network error: %s", str(exc.reason))
            return None, f"openrouter_network_error: {str(exc.reason)}"
        except TimeoutError:
            logger.error("OpenRouter timeout")
            return None, "openrouter_timeout"
        except (ValueError, KeyError) as exc:
            logger.error("OpenRouter parse error: %s", str(exc))
            return None, f"openrouter_parse_error: {str(exc)}"

    @staticmethod
    def _llm_answer(
        system: str,
        question: str,
        context: Optional[Dict[str, Any]],
        is_voice: bool,
    ) -> Tuple[Optional[str], Optional[str]]:
        api_key = _env("OPENAI_API_KEY")
        if not (LANGCHAIN_AVAILABLE and api_key):
            return None, "openai_not_configured"

        model = _env("OPENAI_MODEL", "gpt-4o-mini")
        timeout_s = float(_env("OPENAI_TIMEOUT_S", "8") or "8")

        style = (
            "Respond for voice output: short sentences, no markdown tables, no long lists."
            if is_voice
            else "Respond concise and practical. Use short bullet points when needed."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"{system} {style}"),
                (
                    "human",
                    "Question: {question}\nContext JSON: {context_json}\n"
                    "Return actionable guidance with assumptions called out.",
                ),
            ]
        )

        try:
            llm = ChatOpenAI(model=model, temperature=0.1, api_key=api_key, timeout=timeout_s)
            chain = prompt | llm
            msg = chain.invoke({"question": question, "context_json": AgentOrchestrator._context_block(context)})
            out = getattr(msg, "content", None) or str(msg)
            out = out.strip()
            return (out if out else None), (None if out else "openai_empty_response")
        except Exception as exc:
            logger.exception("OpenAI call failed")
            return None, f"openai_error: {str(exc)}"

    @staticmethod
    def _gemini_answer(
        system: str,
        question: str,
        context: Optional[Dict[str, Any]],
        is_voice: bool,
    ) -> Tuple[Optional[str], Optional[str]]:
        api_key = _env("GEMINI_API_KEY")
        if not api_key:
            return None, "gemini_not_configured"

        model = _env("GEMINI_MODEL", "gemini-1.5-flash")
        timeout_s = float(_env("GEMINI_TIMEOUT_S", "10") or "10")

        style = (
            "Respond for voice output: short sentences, no markdown tables, no long lists."
            if is_voice
            else "Respond concise and practical. Use short bullet points when needed."
        )

        prompt = (
            f"{system} {style}\n\n"
            f"Question: {question}\n"
            f"Context JSON: {AgentOrchestrator._context_block(context)}\n"
            "Return actionable guidance with assumptions called out."
        )

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }
        data = json.dumps(payload).encode("utf-8")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            parsed = json.loads(raw)
            candidates = parsed.get("candidates") or []
            if not candidates:
                return None, "gemini_no_candidates"
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            text = "\n".join(str(p.get("text", "")).strip() for p in parts if p.get("text"))
            text = text.strip()
            return (text if text else None), (None if text else "gemini_empty_text")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = str(exc)
            logger.error("Gemini HTTP error %s: %s", exc.code, detail[:500])
            return None, f"gemini_http_{exc.code}: {detail[:260]}"
        except urllib.error.URLError as exc:
            logger.error("Gemini network error: %s", str(exc.reason))
            return None, f"gemini_network_error: {str(exc.reason)}"
        except TimeoutError:
            logger.error("Gemini timeout")
            return None, "gemini_timeout"
        except (ValueError, KeyError) as exc:
            logger.error("Gemini parse error: %s", str(exc))
            return None, f"gemini_parse_error: {str(exc)}"

    def run(self, mode: str, question: str, routed_agent: Optional[str], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        start = time.perf_counter()
        question = (question or "").strip()
        agent = (routed_agent or "").strip().lower() or self.route_question(question)
        if agent not in AGENT_SYSTEM:
            agent = "copilot"

        key = self._cache_key(mode, agent, question, context)
        cached = self.cache.get(key)
        if cached:
            cached["cached"] = True
            cached["latencyMs"] = int((time.perf_counter() - start) * 1000)
            return cached

        system = AGENT_SYSTEM.get(agent, AGENT_SYSTEM["copilot"])
        if agent == "copilot" and self._is_smalltalk(question):
            system = (
                "Role: BlueWeave assistant.\n"
                "For greetings/small-talk, respond naturally in 1-2 short lines.\n"
                "Do not use numbered headings/sections.\n"
                "Invite the user to ask a specific seaweed farming question."
            )
        groq_answer, groq_err = self._groq_answer(system, question, context, is_voice=(mode == "voice"))
        answer = groq_answer
        llm_provider = "groq" if answer else "none"
        failure_reason = groq_err

        if not answer:
            openrouter_answer, openrouter_err = self._openrouter_answer(system, question, context, is_voice=(mode == "voice"))
            answer = openrouter_answer
            failure_reason = openrouter_err if openrouter_err else failure_reason
            if answer:
                llm_provider = "openrouter"

        # Requested priority chain is strictly Groq -> OpenRouter.
        # Keep OpenAI/Gemini helpers available for future use, but do not call them here.

        used_llm = bool(answer)
        if not answer:
            logger.warning(
                "Fallback used | mode=%s agent=%s reason=%s question=%s",
                mode,
                agent,
                failure_reason or "unknown",
                (question or "")[:180],
            )
            answer = self._fallback_answer(agent, question, context, failure_reason=failure_reason)

        model_name = (
            _env("GROQ_MODEL", "llama-3.1-8b-instant")
            if llm_provider == "groq"
            else _env("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
            if llm_provider == "openrouter"
            else "gateway-fallback"
        )

        out = {
            "answer": answer,
            "agent": agent,
            "routedAgent": agent,
            "stack": self.stack_for(agent),
            "provider": "python-agent-gateway",
            "model": model_name,
            "status": "live" if used_llm else "fallback",
            "failureReason": failure_reason if not used_llm else None,
            "cached": False,
            "latencyMs": int((time.perf_counter() - start) * 1000),
        }
        self.cache.set(key, out)
        return dict(out)


orchestrator = AgentOrchestrator()
app = FastAPI(title="BlueWeave Agent Gateway", version="2.0.0")
_log_provider_bootstrap()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "langchain_available": LANGCHAIN_AVAILABLE,
        "langgraph_available": LANGGRAPH_AVAILABLE,
        "crewai_available": CREWAI_AVAILABLE,
        "groq_sdk_available": GROQ_SDK_AVAILABLE,
        "groq_key_configured": bool(_env("GROQ_API_KEY")),
        "openrouter_key_configured": bool(_env("OPENROUTER_API_KEY")),
        "openai_key_configured": bool(_env("OPENAI_API_KEY")),
        "gemini_key_configured": bool(_env("GEMINI_API_KEY")),
        "voice_supported": True,
        "version": "2.0.0",
    }


@app.post("/agent")
def run_agent(req: AgentRequest):
    out = orchestrator.run(mode="agent", question=req.question, routed_agent=req.agent, context=req.context)
    return {
        "agent": out["agent"],
        "answer": out["answer"],
        "stack": out["stack"],
        "provider": out["provider"],
        "status": out["status"],
        "failureReason": out.get("failureReason"),
        "latencyMs": out["latencyMs"],
        "cached": out["cached"],
    }


@app.post("/chat")
def run_chat(req: ChatRequest):
    out = orchestrator.run(mode="chat", question=req.question, routed_agent=req.routedAgent, context=req.context)
    return {
        "answer": out["answer"],
        "model": out["model"],
        "stack": out["stack"],
        "routedAgent": out["routedAgent"],
        "provider": out["provider"],
        "status": out["status"],
        "failureReason": out.get("failureReason"),
        "latencyMs": out["latencyMs"],
        "cached": out["cached"],
    }


@app.post("/voice/respond")
def voice_respond(req: VoiceRequest):
    out = orchestrator.run(mode="voice", question=req.question, routed_agent=req.routedAgent, context=req.context)
    tts_text = " ".join(str(out["answer"]).split())
    return {
        "answer": out["answer"],
        "ttsText": tts_text,
        "model": out["model"],
        "stack": out["stack"],
        "routedAgent": out["routedAgent"],
        "provider": out["provider"],
        "status": out["status"],
        "failureReason": out.get("failureReason"),
        "latencyMs": out["latencyMs"],
        "cached": out["cached"],
        "locale": req.locale or "en-US",
    }
