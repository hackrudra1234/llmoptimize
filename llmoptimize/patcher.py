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


class _LocalSession:
    def __init__(self):
        self.calls: list = []
        self.total_cost: float = 0.0
        self._budget_limit: float = None
        self._budget_baseline: float = 0.0

    def record_call(self, provider, model, prompt_tokens, completion_tokens,
                    duration_ms=0, caller_frame=None, prompt_preview=""):
        cost = _calculate_cost(model, prompt_tokens, completion_tokens)
        self.calls.append(_CallRecord(provider, model, prompt_tokens, completion_tokens, cost))
        self.total_cost += cost
        if self._budget_limit is not None:
            spent = self.total_cost - self._budget_baseline
            if spent > self._budget_limit:
                raise BudgetExceeded(
                    f"Budget of ${self._budget_limit:.4f} exceeded "
                    f"(spent ${spent:.4f} on this session)"
                )

    def set_budget(self, limit: float) -> None:
        """Activate a spend limit. Called by BudgetContext.__enter__."""
        self._budget_limit    = limit
        self._budget_baseline = self.total_cost

    def clear_budget(self) -> None:
        """Deactivate the spend limit. Called by BudgetContext.__exit__."""
        self._budget_limit = None

    def reset(self):
        self.calls = []
        self.total_cost = 0.0

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
        _payload = _json.dumps({
            "model":             model,
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "session_id":        _sid,
            "provider":          provider,
            "prompt_preview":    prompt_preview[:100],
            "sdk_version":       _ver,
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
        _patched["anthropic"] = True
    except Exception:
        pass


# ── Groq patcher ──────────────────────────────────────────────────────────────

def patch_groq():
    if _patched["groq"]:
        return
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
        _patched["groq"] = True
    except Exception:
        pass


# ── patch_all ─────────────────────────────────────────────────────────────────

def patch_all():
    patch_openai()
    patch_anthropic()
    patch_groq()

def get_patch_status() -> dict:
    return dict(_patched)
