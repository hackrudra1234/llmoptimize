"""
llmoptimize/patcher.py — self-contained SDK patcher.

Works without the server package installed. Used when running from
pip install llmoptimize (Jupyter, Anaconda, any env without server/).
"""

import time
import uuid
import functools
from types import SimpleNamespace
from typing import Any, Tuple

# Fallback session ID used when __init__.RUN_ID is not yet importable
_MODULE_SESSION_ID = str(uuid.uuid4())

# ── Minimal pricing table for local cost estimation ───────────────────────────
_MODEL_COSTS = {
    # OpenAI chat
    "gpt-4":              {"input": 0.03,    "output": 0.06},
    "gpt-4-turbo":        {"input": 0.01,    "output": 0.03},
    "gpt-4o":             {"input": 0.005,   "output": 0.015},
    "gpt-4o-mini":        {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo":      {"input": 0.0005,  "output": 0.0015},
    "o1":                 {"input": 0.015,   "output": 0.06},
    "o1-mini":            {"input": 0.003,   "output": 0.012},
    "o3-mini":            {"input": 0.0011,  "output": 0.0044},
    # Anthropic
    "claude-3-opus":      {"input": 0.015,   "output": 0.075},
    "claude-3-5-sonnet":  {"input": 0.003,   "output": 0.015},
    "claude-3-haiku":     {"input": 0.00025, "output": 0.00125},
    "claude-3-5-haiku":   {"input": 0.0008,  "output": 0.004},
    # Groq
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "llama-3.1-8b-instant":    {"input": 0.00005, "output": 0.00008},
    # Embeddings — OpenAI
    "text-embedding-3-large":  {"input": 0.00013,  "output": 0},
    "text-embedding-3-small":  {"input": 0.00002,  "output": 0},
    "text-embedding-ada-002":  {"input": 0.0001,   "output": 0},
    # Embeddings — Voyage AI
    "voyage-3":                {"input": 0.00006,  "output": 0},
    "voyage-3-lite":           {"input": 0.000016, "output": 0},
    # Embeddings — Jina AI
    "jina-embeddings-v3":      {"input": 0.000018, "output": 0},
    # Google Gemini
    "gemini-1.5-pro":          {"input": 0.00125,  "output": 0.005},
    "gemini-1.5-flash":        {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash":        {"input": 0.0001,   "output": 0.0004},
    "gemini-pro":              {"input": 0.0005,   "output": 0.0015},
    # Mistral
    "mistral-large-latest":    {"input": 0.002,    "output": 0.006},
    "mistral-small-latest":    {"input": 0.0002,   "output": 0.0006},
    "mistral-tiny":            {"input": 0.00015,  "output": 0.00045},
    "open-mixtral-8x7b":       {"input": 0.0007,   "output": 0.0007},
    # Cohere
    "command-r-plus":          {"input": 0.002,    "output": 0.01},
    "command-r":               {"input": 0.00015,  "output": 0.0006},
    "command-light":           {"input": 0.00015,  "output": 0.00015},
    # Cohere embeddings
    "embed-english-v3.0":      {"input": 0.0001,   "output": 0},
    "embed-multilingual-v3.0": {"input": 0.0001,   "output": 0},
}

# Maps each model to a list of cheaper alternatives (best first).
# Dashboard iterates the list and shows each as a separate recommendation card.
_ALTERNATIVES = {
    "gpt-4":             ["gpt-4o-mini"],
    "gpt-4-turbo":       ["gpt-4o-mini"],
    "gpt-4o":            ["gpt-4o-mini"],
    "claude-3-opus":     ["claude-3-5-haiku"],
    "text-embedding-3-large": ["text-embedding-3-small", "voyage-3-lite", "jina-embeddings-v3"],
    "text-embedding-ada-002": ["text-embedding-3-small", "voyage-3-lite"],
}

_EMBEDDING_PREFIXES = ("text-embedding", "embed-", "embedding-", "mistral-embed",
                       "voyage-", "jina-embedding", "amazon.titan-embed")

def _is_embedding(model: str) -> bool:
    m = model.lower()
    return any(m.startswith(p) or p in m for p in _EMBEDDING_PREFIXES)

def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
    c = _MODEL_COSTS.get(model, {"input": 0.001, "output": 0.002})
    return (prompt_tokens * c["input"] + completion_tokens * c["output"]) / 1000


# ── Local session (replaces server.core.magic when server not available) ──────

class _CallRecord:
    def __init__(self, provider, model, prompt_tokens, completion_tokens, cost):
        self.provider          = provider
        self.model             = model
        self.prompt_tokens     = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost              = cost

class BudgetExceeded(Exception):
    """Raised when a budget() context manager limit is exceeded."""

class StepBudgetExceeded(Exception):
    """Raised when a single API call exceeds the step_budget() limit."""


class _LocalSession:
    def __init__(self):
        self.calls: list = []
        self.total_cost: float = 0.0
        self._budget_limit: float = None
        self._budget_baseline: float = 0.0
        self._step_limit: float = None
        self.last_call_tokens: int = 0

    def record_call(self, provider, model, prompt_tokens, completion_tokens,
                    duration_ms=0, caller_frame=None, prompt_preview=""):
        cost = _calculate_cost(model, prompt_tokens, completion_tokens)
        total_tokens = prompt_tokens + completion_tokens

        # Per-step budget check (single call cost)
        if self._step_limit is not None and cost > self._step_limit:
            raise StepBudgetExceeded(
                f"Single call cost ${cost:.6f} exceeded step limit ${self._step_limit:.6f} "
                f"({model}, {prompt_tokens}+{completion_tokens} tokens)"
            )

        # Context growth alert: warn if tokens grew >2x since last call
        if self.last_call_tokens > 50 and total_tokens > self.last_call_tokens * 2:
            print(
                f"\n[llmoptimize] Context growth alert: "
                f"{self.last_call_tokens:,} \u2192 {total_tokens:,} tokens "
                f"({total_tokens / self.last_call_tokens:.1f}\u00d7 growth) "
                f"\u2014 consider compressing older history to cut input costs\n"
            )

        self.calls.append(_CallRecord(provider, model, prompt_tokens, completion_tokens, cost))
        self.total_cost += cost
        self.last_call_tokens = total_tokens

        # Cumulative budget check
        if self._budget_limit is not None:
            spent = self.total_cost - self._budget_baseline
            if spent > self._budget_limit:
                raise BudgetExceeded(
                    f"Budget of ${self._budget_limit:.4f} exceeded "
                    f"(spent ${spent:.4f} on this session)"
                )

    def set_budget(self, limit: float) -> None:
        """Activate a cumulative spend limit. Called by BudgetContext.__enter__."""
        self._budget_limit    = limit
        self._budget_baseline = self.total_cost

    def clear_budget(self) -> None:
        """Deactivate the cumulative spend limit. Called by BudgetContext.__exit__."""
        self._budget_limit = None

    def set_step_budget(self, limit: float) -> None:
        """Activate a per-call spend limit. Called by StepBudgetContext.__enter__."""
        self._step_limit = limit

    def clear_step_budget(self) -> None:
        """Deactivate the per-call spend limit. Called by StepBudgetContext.__exit__."""
        self._step_limit = None

    def reset(self):
        self.calls = []
        self.total_cost = 0.0
        self.last_call_tokens = 0

_session = _LocalSession()

def get_local_session() -> _LocalSession:
    return _session

def reset_local_session():
    _session.reset()


# ── Dry-run state ─────────────────────────────────────────────────────────────

_DRY_RUN: bool = False

def enable_dry_run():
    global _DRY_RUN
    _DRY_RUN = True

def disable_dry_run():
    global _DRY_RUN
    _DRY_RUN = False

def is_dry_run() -> bool:
    return _DRY_RUN




# ── Hardcoded API key warnings ───────────────────────────────────────────────────────

import logging as _logging
_log = _logging.getLogger(__name__)
_warned_keys: set = set()


def _check_api_key_kwarg(provider: str, kwargs: dict):
    """Warn (once per key) if api_key= is hardcoded in constructor."""
    _KEY_NAMES = ("api_key", "token", "api_token")
    if not any(k in kwargs for k in _KEY_NAMES):
        return
    value = next((kwargs[k] for k in _KEY_NAMES if k in kwargs and kwargs[k]), None)
    display = value if (value and isinstance(value, str)) else "<empty>"
    if display not in _warned_keys:
        _warned_keys.add(display)
        redacted = display[:8] + "..." if len(display) > 8 else "***"
        _log.debug("llmoptimize: %s client constructed with explicit api_key= (%s)", provider, redacted)


def _check_positional_key(provider: str, args: tuple):
    """Warn if api_key passed as first positional arg (Mistral, Cohere)."""
    key = args[0] if args and isinstance(args[0], str) else None
    if key and key not in _warned_keys:
        _warned_keys.add(key)
        redacted = key[:8] + "..." if len(key) > 8 else "***"
        _log.debug("llmoptimize: %s client with positional api_key= (%s)", provider, redacted)


# ── Lightweight prompt security scan ────────────────────────────────────────────

import re as _re

_SECRET_PATTERNS = [
    _re.compile(r"(sk-[a-zA-Z0-9]{20,})"),           # OpenAI key
    _re.compile(r"(sk-ant-[a-zA-Z0-9\-]{20,})"),     # Anthropic key
    _re.compile(r"(gsk_[a-zA-Z0-9]{20,})"),           # Groq key
    _re.compile(r"(AIza[a-zA-Z0-9_\-]{30,})"),        # Google API key
    _re.compile(r"(ghp_[a-zA-Z0-9]{36})"),            # GitHub token
    _re.compile(r"(AKIA[A-Z0-9]{16})"),               # AWS access key
]


def _redact_preview(preview: str) -> str:
    """Redact obvious API keys/tokens from prompt preview before sending."""
    if not preview:
        return ""
    for pattern in _SECRET_PATTERNS:
        preview = pattern.sub("[REDACTED]", preview)
    return preview


# ── Mock responses ────────────────────────────────────────────────────────────

def _mock_chat_response(model: str, prompt_tokens: int):
    usage   = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=0,
                               total_tokens=prompt_tokens)
    message = SimpleNamespace(content="", role="assistant",
                               function_call=None, tool_calls=None)
    choice  = SimpleNamespace(message=message, finish_reason="stop", index=0)
    return SimpleNamespace(id="dry-run", object="chat.completion",
                            model=model, choices=[choice], usage=usage)

def _mock_embed_response(model: str, prompt_tokens: int):
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens)
    data  = SimpleNamespace(embedding=[], index=0, object="embedding")
    return SimpleNamespace(object="list", model=model, data=[data], usage=usage)

def _mock_anthropic_response(model: str, prompt_tokens: int):
    usage   = SimpleNamespace(input_tokens=prompt_tokens, output_tokens=0)
    content = SimpleNamespace(type="text", text="")
    return SimpleNamespace(id="dry-run", type="message", role="assistant",
                            model=model, content=[content],
                            stop_reason="end_turn", usage=usage)


# ── Token helpers ─────────────────────────────────────────────────────────────

def _extract_preview(messages, max_chars: int = 200) -> str:
    """Extract first user message as prompt preview for server-side analysis."""
    try:
        if isinstance(messages, list):
            for m in messages:
                role    = m.get("role", "") if isinstance(m, dict) else ""
                content = m.get("content", "") if isinstance(m, dict) else str(m)
                if role in ("user", "human") or not role:
                    return str(content)[:max_chars]
        elif isinstance(messages, str):
            return messages[:max_chars]
    except Exception:
        pass
    return ""


def _estimate_tokens(messages) -> int:
    try:
        if isinstance(messages, list):
            text = " ".join(
                (m.get("content", "") if isinstance(m, dict) else str(m))
                for m in messages
            )
        elif isinstance(messages, str):
            text = messages
        else:
            text = str(messages)
        return max(1, len(text) // 4)
    except Exception:
        return 100

def _openai_tokens(resp) -> Tuple[int, int]:
    try:
        u = resp.usage
        return getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0
    except Exception:
        return 0, 0

def _anthropic_tokens(resp) -> Tuple[int, int]:
    try:
        u = resp.usage
        return getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0
    except Exception:
        return 0, 0



def _gemini_tokens(resp) -> Tuple[int, int]:
    try:
        if hasattr(resp, "usage_metadata"):
            meta = resp.usage_metadata
            return (
                getattr(meta, "prompt_token_count", 0) or 0,
                getattr(meta, "candidates_token_count", 0) or 0,
            )
    except Exception:
        pass
    return 0, 0

def _mistral_tokens(resp) -> Tuple[int, int]:
    try:
        u = resp.usage
        return getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0
    except Exception:
        return 0, 0

def _cohere_tokens(resp) -> Tuple[int, int]:
    try:
        if hasattr(resp, "usage") and resp.usage:
            tokens = resp.usage
            input_t  = getattr(tokens, "input_tokens", None)
            output_t = getattr(tokens, "output_tokens", None)
            if input_t is None and hasattr(tokens, "billed_units"):
                input_t  = getattr(tokens.billed_units, "input_tokens", 0)
                output_t = getattr(tokens.billed_units, "output_tokens", 0)
            return (input_t or 0, output_t or 0)
    except Exception:
        pass
    return 0, 0


# ── Agent framework detection ─────────────────────────────────────────────────

# Known agent framework module path fragments
_FRAMEWORK_SIGNATURES = {
    "langchain":   "langchain",
    "crewai":      "crewai",
    "autogen":     "autogen",
    "llama_index": "llama_index",
    "haystack":    "haystack",
    "agentops":    "agentops",
    "smolagents":  "smolagents",
    "pydantic_ai": "pydantic_ai",
}

_detected_framework: str = None  # cached after first detection


def _detect_framework() -> str:
    """Walk the Python call stack to identify the agent framework in use."""
    global _detected_framework
    if _detected_framework:
        return _detected_framework
    try:
        import sys
        frame = sys._getframe(2)   # start above _detect_framework + _record
        while frame is not None:
            filename = frame.f_code.co_filename.replace("\\", "/").lower()
            for key, name in _FRAMEWORK_SIGNATURES.items():
                if f"/{key}/" in filename or f"\\{key}\\" in filename:
                    _detected_framework = name
                    print(f"\n[llmoptimize] Agent framework detected: {name}\n")
                    return name
            frame = frame.f_back
    except Exception:
        pass
    return "unknown"


def _record(provider, model, prompt_tokens, completion_tokens, prompt_preview=""):
    _session.record_call(provider=provider, model=model,
                          prompt_tokens=prompt_tokens,
                          completion_tokens=completion_tokens,
                          prompt_preview=prompt_preview)
    # Also sync to server session when running in full dev project
    try:
        from server.core.magic import get_session
        get_session().record_call(
            provider=provider, model=model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            duration_ms=0, caller_frame=None, prompt_preview=prompt_preview[:80],
        )
    except Exception:
        pass
    # Fire-and-forget POST to server — enables loop detection, RAG analysis, etc.
    try:
        import urllib.request, json as _json, os as _os
        _server = _os.environ.get("AIOPTIMIZE_SERVER_URL", "https://aioptimize.up.railway.app")
        # Use RUN_ID from __init__ if available, else fall back to a module-level id
        try:
            from llmoptimize import RUN_ID as _sid, __version__ as _ver
        except Exception:
            _sid = _MODULE_SESSION_ID
            _ver = "unknown"
        _framework = _detect_framework()
        _payload = _json.dumps({
            "model":             model,
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "session_id":        _sid,
            "provider":          provider,
            "prompt_preview":    _redact_preview(prompt_preview[:200]),
            "sdk_version":       _ver,
            "agent_framework":   _framework,
        }).encode("utf-8")
        _req = urllib.request.Request(
            f"{_server}/track",
            data=_payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(_req, timeout=3)
    except Exception:
        pass


# ── Patch state ───────────────────────────────────────────────────────────────

_patched = {"openai": False, "anthropic": False, "groq": False,
            "gemini": False, "mistral": False, "cohere": False}


# ── OpenAI patcher ────────────────────────────────────────────────────────────

def patch_openai():
    if _patched["openai"]:
        return

    # Warn if OpenAI(api_key=...) is hardcoded
    try:
        import openai as _openai_mod
        _OpenAI = getattr(_openai_mod, "OpenAI", None)
        if _OpenAI:
            _orig_oi_init = _OpenAI.__init__
            @functools.wraps(_orig_oi_init)
            def _oi_init(self_client, *args, **kwargs):
                _check_api_key_kwarg("openai", kwargs)
                _orig_oi_init(self_client, *args, **kwargs)
            _OpenAI.__init__ = _oi_init
    except Exception:
        pass

    try:
        from openai.resources.chat.completions import Completions
        _orig = Completions.create

        @functools.wraps(_orig)
        def _chat(self_obj, *args, **kwargs):
            model = kwargs.get("model") or "openai/unknown"
            msgs  = kwargs.get("messages", [])
            preview = _extract_preview(msgs)
            if _DRY_RUN:
                p = _estimate_tokens(msgs)
                _record("openai", model, p, 0, preview)
                return _mock_chat_response(model, p)
            t0   = time.time()
            resp = _orig(self_obj, *args, **kwargs)
            p, c = _openai_tokens(resp)
            _record("openai", model, p, c, preview)
            return resp

        Completions.create = _chat

        try:
            from openai.resources.chat.completions import AsyncCompletions
            _orig_async_chat = AsyncCompletions.create

            @functools.wraps(_orig_async_chat)
            async def _async_chat(self_obj, *args, **kwargs):
                model = kwargs.get("model") or "openai/unknown"
                msgs  = kwargs.get("messages", [])
                preview = _extract_preview(msgs)
                if _DRY_RUN:
                    p = _estimate_tokens(msgs)
                    _record("openai", model, p, 0, preview)
                    return _mock_chat_response(model, p)
                resp = await _orig_async_chat(self_obj, *args, **kwargs)
                p, c = _openai_tokens(resp)
                _record("openai", model, p, c, preview)
                return resp

            AsyncCompletions.create = _async_chat
        except (ImportError, AttributeError):
            pass

    except Exception:
        pass

    try:
        from openai.resources.embeddings import Embeddings
        _orig_e = Embeddings.create

        @functools.wraps(_orig_e)
        def _embed(self_obj, *args, **kwargs):
            model   = kwargs.get("model") or "openai-embed/unknown"
            inp     = kwargs.get("input", "")
            preview = (inp if isinstance(inp, str) else (inp[0] if isinstance(inp, list) and inp else ""))[:200]
            if _DRY_RUN:
                p = _estimate_tokens(inp)
                _record("openai", model, p, 0, preview)
                return _mock_embed_response(model, p)
            resp = _orig_e(self_obj, *args, **kwargs)
            p    = getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0
            _record("openai", model, p, 0, preview)
            return resp

        Embeddings.create = _embed
    except Exception:
        pass

    _patched["openai"] = True


# ── Anthropic patcher ─────────────────────────────────────────────────────────

def patch_anthropic():
    if _patched["anthropic"]:
        return

    # Warn if Anthropic(api_key=...) is hardcoded
    try:
        import anthropic as _anthropic_mod
        _Anthropic = getattr(_anthropic_mod, "Anthropic", None)
        if _Anthropic:
            _orig_an_init = _Anthropic.__init__
            @functools.wraps(_orig_an_init)
            def _an_init(self_client, *args, **kwargs):
                _check_api_key_kwarg("anthropic", kwargs)
                _orig_an_init(self_client, *args, **kwargs)
            _Anthropic.__init__ = _an_init
    except Exception:
        pass

    try:
        from anthropic.resources.messages import Messages
        _orig = Messages.create

        @functools.wraps(_orig)
        def _msg(self_obj, *args, **kwargs):
            model   = kwargs.get("model") or "anthropic/unknown"
            msgs    = kwargs.get("messages", [])
            preview = _extract_preview(msgs)
            if _DRY_RUN:
                p = _estimate_tokens(msgs)
                _record("anthropic", model, p, 0, preview)
                return _mock_anthropic_response(model, p)
            resp = _orig(self_obj, *args, **kwargs)
            p, c = _anthropic_tokens(resp)
            _record("anthropic", model, p, c, preview)
            return resp

        Messages.create = _msg

        try:
            from anthropic.resources.messages import AsyncMessages
            _orig_async_msg = AsyncMessages.create

            @functools.wraps(_orig_async_msg)
            async def _async_msg(self_obj, *args, **kwargs):
                model   = kwargs.get("model") or "anthropic/unknown"
                msgs    = kwargs.get("messages", [])
                preview = _extract_preview(msgs)
                if _DRY_RUN:
                    p = _estimate_tokens(msgs)
                    _record("anthropic", model, p, 0, preview)
                    return _mock_anthropic_response(model, p)
                resp = await _orig_async_msg(self_obj, *args, **kwargs)
                p, c = _anthropic_tokens(resp)
                _record("anthropic", model, p, c, preview)
                return resp

            AsyncMessages.create = _async_msg
        except (ImportError, AttributeError):
            pass

        _patched["anthropic"] = True
    except Exception:
        pass


# ── Groq patcher ──────────────────────────────────────────────────────────────

def patch_groq():
    if _patched["groq"]:
        return

    # Warn if Groq(api_key=...) is hardcoded
    try:
        import groq as _groq_mod
        _Groq = getattr(_groq_mod, "Groq", None)
        if _Groq:
            _orig_gq_init = _Groq.__init__
            @functools.wraps(_orig_gq_init)
            def _gq_init(self_client, *args, **kwargs):
                _check_api_key_kwarg("groq", kwargs)
                _orig_gq_init(self_client, *args, **kwargs)
            _Groq.__init__ = _gq_init
    except Exception:
        pass

    try:
        from groq.resources.chat.completions import Completions as GC
        _orig = GC.create

        @functools.wraps(_orig)
        def _chat(self_obj, *args, **kwargs):
            model   = kwargs.get("model") or "groq/unknown"
            msgs    = kwargs.get("messages", [])
            preview = _extract_preview(msgs)
            if _DRY_RUN:
                p = _estimate_tokens(msgs)
                _record("groq", model, p, 0, preview)
                return _mock_chat_response(model, p)
            resp = _orig(self_obj, *args, **kwargs)
            p, c = _openai_tokens(resp)
            _record("groq", model, p, c, preview)
            return resp

        GC.create = _chat

        try:
            from groq.resources.chat.completions import AsyncCompletions as GAsyncC
            _orig_async_groq = GAsyncC.create

            @functools.wraps(_orig_async_groq)
            async def _async_groq(self_obj, *args, **kwargs):
                model   = kwargs.get("model") or "groq/unknown"
                msgs    = kwargs.get("messages", [])
                preview = _extract_preview(msgs)
                if _DRY_RUN:
                    p = _estimate_tokens(msgs)
                    _record("groq", model, p, 0, preview)
                    return _mock_chat_response(model, p)
                resp = await _orig_async_groq(self_obj, *args, **kwargs)
                p, c = _openai_tokens(resp)
                _record("groq", model, p, c, preview)
                return resp

            GAsyncC.create = _async_groq
        except (ImportError, AttributeError):
            pass

        _patched["groq"] = True
    except Exception:
        pass


# ── patch_all ─────────────────────────────────────────────────────────────────



# ── Gemini patcher ──────────────────────────────────────────────────────────────────

def patch_gemini():
    if _patched["gemini"]:
        return

    # Constructor key warning
    try:
        import google.genai as genai
        _orig_configure = getattr(genai, "configure", None)
        if _orig_configure:
            @functools.wraps(_orig_configure)
            def _patched_configure(*args, **kwargs):
                _check_api_key_kwarg("gemini", kwargs)
                return _orig_configure(*args, **kwargs)
            genai.configure = _patched_configure
    except Exception:
        pass

    try:
        import google.genai as genai
        GenerativeModel = getattr(genai, "GenerativeModel", None)
        if GenerativeModel is None:
            return

        _orig_generate = GenerativeModel.generate_content

        @functools.wraps(_orig_generate)
        def _sync_gen(self_obj, *args, **kwargs):
            model    = (getattr(self_obj, "model_name", None) or "gemini/unknown").replace("models/", "")
            contents = args[0] if args else kwargs.get("contents", "")
            preview  = _extract_preview(contents if isinstance(contents, list) else str(contents))
            if _DRY_RUN:
                p = _estimate_tokens(contents)
                _record("gemini", model, p, 0, preview)
                return _mock_chat_response(model, p)
            resp = _orig_generate(self_obj, *args, **kwargs)
            p, c = _gemini_tokens(resp)
            _record("gemini", model, p, c, preview)
            return resp

        GenerativeModel.generate_content = _sync_gen

        _orig_gen_async = getattr(GenerativeModel, "generate_content_async", None)
        if _orig_gen_async:
            @functools.wraps(_orig_gen_async)
            async def _async_gen(self_obj, *args, **kwargs):
                model    = (getattr(self_obj, "model_name", None) or "gemini/unknown").replace("models/", "")
                contents = args[0] if args else kwargs.get("contents", "")
                preview  = _extract_preview(contents if isinstance(contents, list) else str(contents))
                if _DRY_RUN:
                    p = _estimate_tokens(contents)
                    _record("gemini", model, p, 0, preview)
                    return _mock_chat_response(model, p)
                resp = await _orig_gen_async(self_obj, *args, **kwargs)
                p, c = _gemini_tokens(resp)
                _record("gemini", model, p, c, preview)
                return resp

            GenerativeModel.generate_content_async = _async_gen

        _patched["gemini"] = True

    except ImportError:
        pass
    except Exception:
        pass


# ── Mistral patcher ─────────────────────────────────────────────────────────────────

def patch_mistral():
    if _patched["mistral"]:
        return
    try:
        from mistralai import Mistral
        _orig_init = Mistral.__init__

        @functools.wraps(_orig_init)
        def _patched_init(self_client, *args, **kwargs):
            _check_api_key_kwarg("mistral", kwargs)
            _check_positional_key("mistral", args)
            _orig_init(self_client, *args, **kwargs)
            try:
                _orig_complete = self_client.chat.complete

                @functools.wraps(_orig_complete)
                def _sync_complete(*a, **kw):
                    model   = kw.get("model") or "mistral/unknown"
                    msgs    = kw.get("messages", [])
                    preview = _extract_preview(msgs)
                    if _DRY_RUN:
                        p = _estimate_tokens(msgs)
                        _record("mistral", model, p, 0, preview)
                        return _mock_chat_response(model, p)
                    resp = _orig_complete(*a, **kw)
                    p, c = _mistral_tokens(resp)
                    _record("mistral", model, p, c, preview)
                    return resp

                self_client.chat.complete = _sync_complete

                if hasattr(self_client.chat, "complete_async"):
                    _orig_async = self_client.chat.complete_async

                    @functools.wraps(_orig_async)
                    async def _async_complete(*a, **kw):
                        model   = kw.get("model") or "mistral/unknown"
                        msgs    = kw.get("messages", [])
                        preview = _extract_preview(msgs)
                        if _DRY_RUN:
                            p = _estimate_tokens(msgs)
                            _record("mistral", model, p, 0, preview)
                            return _mock_chat_response(model, p)
                        resp = await _orig_async(*a, **kw)
                        p, c = _mistral_tokens(resp)
                        _record("mistral", model, p, c, preview)
                        return resp

                    self_client.chat.complete_async = _async_complete

            except (AttributeError, TypeError):
                pass

        Mistral.__init__ = _patched_init
        _patched["mistral"] = True

    except ImportError:
        pass
    except Exception:
        pass


# ── Cohere patcher ──────────────────────────────────────────────────────────────────

def patch_cohere():
    if _patched["cohere"]:
        return
    try:
        import cohere
        CohereClient = getattr(cohere, "ClientV2", None) or getattr(cohere, "Client", None)
        if CohereClient is None:
            return

        _orig_init = CohereClient.__init__

        @functools.wraps(_orig_init)
        def _patched_init(self_client, *args, **kwargs):
            _check_api_key_kwarg("cohere", kwargs)
            _check_positional_key("cohere", args)
            _orig_init(self_client, *args, **kwargs)
            try:
                _orig_chat = self_client.chat

                @functools.wraps(_orig_chat)
                def _sync_chat(*a, **kw):
                    model   = kw.get("model") or "cohere/unknown"
                    preview = _extract_preview(kw.get("messages", []))
                    if _DRY_RUN:
                        p = _estimate_tokens(kw.get("messages", []))
                        _record("cohere", model, p, 0, preview)
                        return _mock_chat_response(model, p)
                    resp = _orig_chat(*a, **kw)
                    p, c = _cohere_tokens(resp)
                    _record("cohere", model, p, c, preview)
                    return resp

                self_client.chat = _sync_chat

                if hasattr(self_client, "embed"):
                    _orig_embed = self_client.embed

                    @functools.wraps(_orig_embed)
                    def _sync_embed(*a, **kw):
                        model = kw.get("model") or "cohere-embedding/unknown"
                        texts = kw.get("texts", [])
                        preview = texts[0][:200] if texts else ""
                        if _DRY_RUN:
                            p = sum(len(t) for t in texts) // 4
                            _record("cohere", model, p, 0, str(preview))
                            return _mock_embed_response(model, p)
                        resp = _orig_embed(*a, **kw)
                        p = sum(len(t) for t in texts) // 4
                        _record("cohere", model, p, 0, str(preview))
                        return resp

                    self_client.embed = _sync_embed

            except (AttributeError, TypeError):
                pass

        CohereClient.__init__ = _patched_init
        _patched["cohere"] = True

    except ImportError:
        pass
    except Exception:
        pass


# ── patch_all ─────────────────────────────────────────────────────────────────────

def patch_all():
    patch_openai()
    patch_anthropic()
    patch_groq()
    patch_gemini()
    patch_mistral()
    patch_cohere()

def get_patch_status() -> dict:
    return dict(_patched)
