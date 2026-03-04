"""
LLMOptimize v3.2.2 - AI Cost Optimization SDK

llmoptimize is a READ-ONLY cost advisor.
It intercepts AI API calls silently, then tells you how to save money.
It never makes AI API calls itself — no API key needed for recommendations.
"""

__version__ = "3.2.2"
__author__  = "LLMOptimize Team"

import os
import uuid
import json
import logging
import urllib.request

log = logging.getLogger(__name__)

from llmoptimize.dashboard import LLMOptimize

SERVER_URL = os.getenv("AIOPTIMIZE_SERVER_URL", "https://aioptimize.up.railway.app")


def _get_session_id() -> str:
    """Get or create a persistent anonymous session ID."""
    session_file = os.path.expanduser("~/.llmoptimize_session")
    try:
        if os.path.exists(session_file):
            with open(session_file) as f:
                sid = f.read().strip()
                if sid:
                    return sid
        sid = str(uuid.uuid4())
        with open(session_file, "w") as f:
            f.write(sid)
        return sid
    except OSError:
        return str(uuid.uuid4())


SESSION_ID = _get_session_id()

# RUN_ID: fresh UUID every time Python imports this module (one per script run).
# Keeps each run isolated on the server — independent of the persistent SESSION_ID.
RUN_ID: str = str(uuid.uuid4())


# ── Tracking ──────────────────────────────────────────────────────────────────

def _track_call(
    model:             str,
    prompt_tokens:     int,
    completion_tokens: int = 0,
    provider:          str = "unknown",
    prompt_preview:    str = "",
) -> None:
    """
    Record one call in both the local MagicSession (instant, no server needed)
    and fire-and-forget POST to the server (for cross-session analytics).
    Never blocks or raises — completely silent.
    """
    # 1. Local in-process session — always works, no server needed
    try:
        from server.core.magic import get_session, get_caller_frame
        get_session().record_call(
            provider          = provider,
            model             = model,
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            duration_ms       = 0,
            caller_frame      = get_caller_frame(),
            prompt_preview    = prompt_preview[:80],
        )
    except Exception:
        pass

    # 2. Server POST — best-effort, silent failure is fine
    try:
        payload = json.dumps({
            "model":             model,
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "session_id":        RUN_ID,
            "provider":          provider,
            "prompt_preview":    prompt_preview[:100],
            "sdk_version":       __version__,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{SERVER_URL}/track",
            data    = payload,
            method  = "POST",
            headers = {
                "Content-Type": "application/json",
                "User-Agent":   f"llmoptimize/{__version__}",
            },
        )
        urllib.request.urlopen(req, timeout=3)
    except Exception:
        pass


# ── Auto-patching ─────────────────────────────────────────────────────────────
# Silently patches OpenAI / Anthropic / Groq / Gemini / Mistral / Cohere
# so every API call is automatically tracked. No output produced here.

try:
    from server.utils.patcher import patch_all as _patch_all
    _patch_all()
except Exception as _e:
    log.debug("patcher.patch_all() failed: %s", _e)


# ── Dashboard ─────────────────────────────────────────────────────────────────
_dashboard_client = LLMOptimize()


class _Report:
    """
    Dual-mode report helper — exposed as ``llmoptimize.report``.

    Direct call — show cost & recommendations for calls tracked so far:
        llmoptimize.report()

    Context manager — dry-run mode (no real API calls, no cost):
        with llmoptimize.report:
            resp = client.chat.completions.create(model="gpt-4o", ...)
        # recommendations printed on exit; remove `with` to go live
    """

    def __call__(self, interactive: bool = True) -> None:
        _dashboard_client.report(interactive=interactive)

    def __enter__(self):
        try:
            from server.utils.patcher import enable_dry_run
            enable_dry_run()
        except Exception as e:
            log.debug("enable_dry_run failed: %s", e)
        return self

    def __exit__(self, *_):
        try:
            from server.utils.patcher import disable_dry_run
            disable_dry_run()
        except Exception as e:
            log.debug("disable_dry_run failed: %s", e)
        _dashboard_client.report(interactive=True)
        return False


report = _Report()


def track(
    model:             str,
    prompt_tokens:     int,
    completion_tokens: int = 0,
    provider:          str = "custom",
    prompt_preview:    str = "",
) -> dict:
    """Manually track a call — no real API needed."""
    _track_call(model, prompt_tokens, completion_tokens, provider, prompt_preview)
    return {"status": "tracked"}


def new_session() -> str:
    """Start a fresh session — clears local ID and server data."""
    global SESSION_ID, RUN_ID
    session_file = os.path.expanduser("~/.llmoptimize_session")
    try:
        os.remove(session_file)
    except OSError:
        pass
    SESSION_ID = str(uuid.uuid4())
    RUN_ID     = str(uuid.uuid4())
    try:
        with open(session_file, "w") as f:
            f.write(SESSION_ID)
    except OSError:
        pass
    try:
        req = urllib.request.Request(
            f"{SERVER_URL}/session/{SESSION_ID}",
            method  = "DELETE",
            headers = {"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=3)
    except Exception:
        pass
    try:
        from server.core.magic import reset_session
        reset_session()
    except Exception:
        pass
    _dashboard_client.session_id = RUN_ID
    return SESSION_ID


class _Task:
    """
    Named task session — resets tracking, optionally dry-runs, shows a
    labelled report when the block exits.

    Usage::

        # Live calls (real API, tracks actual tokens)
        with llmoptimize.task("rag-pipeline"):
            chunks  = embed_client.embeddings.create(...)
            summary = chat_client.chat.completions.create(...)

        # Dry-run (no real API calls, no cost, plan before you ship)
        with llmoptimize.task("cost-planning", dry_run=True):
            chat_client.chat.completions.create(model="gpt-4", ...)
    """

    def __init__(self, name: str, dry_run: bool = False):
        self._name    = name
        self._dry_run = dry_run

    def __enter__(self):
        new_session()                      # clean slate for this task
        if self._dry_run:
            try:
                from server.utils.patcher import enable_dry_run
                enable_dry_run()
            except Exception as e:
                log.debug("enable_dry_run failed: %s", e)
        return self

    def __exit__(self, *_):
        if self._dry_run:
            try:
                from server.utils.patcher import disable_dry_run
                disable_dry_run()
            except Exception as e:
                log.debug("disable_dry_run failed: %s", e)
        _dashboard_client.report(interactive=True, task_name=self._name)
        return False


def task(name: str, dry_run: bool = False) -> _Task:
    """
    Start a named task session.

    Resets all tracking, runs your code, then shows a report labelled
    with *name* on exit.  Pass ``dry_run=True`` to intercept calls
    without making real API requests.

    Examples::

        with llmoptimize.task("rag-pipeline"):
            ...  # real calls, tracked

        with llmoptimize.task("cost-planning", dry_run=True):
            ...  # zero cost, zero network
    """
    return _Task(name, dry_run)


__all__ = [
    "report", "track", "task", "new_session",
    "__version__", "SESSION_ID", "RUN_ID", "SERVER_URL",
]
