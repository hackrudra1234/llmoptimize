"""
LLMOptimize v3.2.2 - AI Cost Optimization SDK

llmoptimize is a READ-ONLY cost advisor.
It intercepts AI API calls silently, then tells you how to save money.
It never makes AI API calls itself — no API key needed for recommendations.
"""

__version__ = "3.4.0"
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

# ── Auto-patching ─────────────────────────────────────────────────────────────
# Always use the self-contained llmoptimize.patcher (works in all environments).
# Falls back to server.utils.patcher when running inside the full dev project.

from llmoptimize.patcher import (
    patch_all as _patch_all,
    enable_dry_run as _enable_dry_run,
    disable_dry_run as _disable_dry_run,
    reset_local_session as _reset_local_session,
    _record as _patcher_record,
    BudgetExceeded,
    StepBudgetExceeded,
)
_patch_all()


# ── Dashboard ─────────────────────────────────────────────────────────────────
_dashboard_client = LLMOptimize()


class _ReportContext:
    """
    Returned by ``report(...)`` — shows report immediately if not used as a
    context manager; enables dry-run mode if used with ``with``.
    """

    def __init__(self, interactive: bool = True):
        self._interactive = interactive
        self._entered = False

    def __enter__(self):
        self._entered = True
        _enable_dry_run()
        return self

    def __exit__(self, *_):
        _disable_dry_run()
        _dashboard_client.report(interactive=self._interactive)
        return False

    def __del__(self):
        # Called when the object is garbage-collected (i.e. used as a plain
        # function call rather than a context manager).
        if not self._entered:
            _dashboard_client.report(interactive=self._interactive)


class _Report:
    """
    Dual-mode report helper — exposed as ``llmoptimize.report``.

    Direct call — show cost & recommendations for calls tracked so far:
        llmoptimize.report()                      # interactive (browser option)
        llmoptimize.report(interactive=False)     # plain text report

    Context manager — dry-run mode (no real API calls, no cost):
        with llmoptimize.report:                          # interactive on exit
            resp = client.chat.completions.create(...)
        with llmoptimize.report():                        # same
            ...
        with llmoptimize.report(interactive=False):       # plain report on exit
            ...
    """

    def __call__(self, interactive: bool = True) -> _ReportContext:
        return _ReportContext(interactive)

    def __enter__(self):
        _enable_dry_run()
        return self

    def __exit__(self, *_):
        _disable_dry_run()
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
    _patcher_record(provider, model, prompt_tokens, completion_tokens, 0, None, prompt_preview)
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
    _reset_local_session()
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
        new_session()
        if self._dry_run:
            _enable_dry_run()
        return self

    def __exit__(self, *_):
        if self._dry_run:
            _disable_dry_run()
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


def _http_post(endpoint: str, payload: dict, timeout: int = 8) -> dict:
    """POST to the server. Returns parsed JSON or {'success': False, 'error': ...}."""
    try:
        body = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            f"{SERVER_URL}{endpoint}",
            data    = body,
            method  = "POST",
            headers = {
                "Content-Type": "application/json",
                "User-Agent":   f"llmoptimize/{__version__}",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def select_model(code: str) -> dict:
    """
    Get the optimal Groq model for a code/task string.

    Uses server-side GroqModelSelector (rule-based + cache, no API key needed).
    Returns model ID, complexity level, cost savings vs the heavy 70B model.

    Example::

        result = llmoptimize.select_model(\"\"\"
            Extract the user name from this JSON string.
        \"\"\")
        # result["selected_model"] -> "llama-3.1-8b-instant"
        # result["complexity_level"] -> "simple"
        # result["vs_heavy_model"]["savings_pct"] -> 84
    """
    return _http_post("/api/select-model", {"code": code})


def check_loop(actions: list) -> dict:
    """
    Check a list of agent action strings for loop/repetition patterns.

    Detects: exact repeats, circular patterns (A→B→C→A), alternating loops.
    No API key needed — server-side rule-based detection.

    Example::

        result = llmoptimize.check_loop([
            "search web for python docs",
            "read python docs",
            "search web for python docs",   # repeated!
            "read python docs",
        ])
        # result["loop_detected"] -> True
        # result["loops"][0]["pattern_type"] -> "exact_repeat"
    """
    if not isinstance(actions, list):
        actions = list(actions)
    return _http_post("/api/loop-check", {"actions": actions})


def rag(
    docs=None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    embedding_model: str = None,
    llm_model: str = None,
    query_type: str = None,
    silent: bool = False,
):
    """
    Context manager that analyses your RAG pipeline and prints recommendations.

    Wraps your full RAG setup — pass your loaded docs plus the config you're
    using. On exit, the server computes optimal chunk size, overlap, embedding
    model, and LLM model, then prints actionable recommendations.

    Example::

        loader = PyPDFLoader("docs/report.pdf")
        pages  = loader.load()

        with llmoptimize.rag(
            docs=pages,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="text-embedding-ada-002",
            llm_model="gpt-4",
        ):
            chunks      = splitter.split_documents(pages)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            result      = qa_chain.run("What is X?")
        # Recommendations print automatically on exit
    """
    from llmoptimize.rag import RAGContext
    return RAGContext(
        docs=docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        llm_model=llm_model,
        query_type=query_type,
        silent=silent,
    )


def analyze(prompt: str, model: str) -> dict:
    """
    Get a smart optimization recommendation for a specific prompt + model combo.

    Uses server-side heuristic + ML pipeline (no API key needed).
    Returns suggested cheaper model with confidence and savings estimate.

    Example::

        result = llmoptimize.analyze(
            prompt="Classify this support ticket as urgent or normal: ...",
            model="gpt-4o",
        )
        # result["recommendation"]["suggested_model"] -> "gpt-4o-mini"
        # result["recommendation"]["estimated_savings_percent"] -> 96
    """
    return _http_post("/api/recommend", {
        "prompt_preview": prompt[:200],
        "current_model":  model,
    })



def budget(max_usd: float, warn_only: bool = False):
    """
    Context manager that enforces a cumulative spend limit on patched AI API calls.

    Raises BudgetExceeded (or prints a warning if warn_only=True) as soon as
    accumulated spend within the block exceeds max_usd.

    Inside the block you can read .spent and .remaining at any time.

    Example::

        with llmoptimize.budget(max_usd=0.10):
            client.chat.completions.create(model="gpt-4", ...)

        # warn instead of raising:
        with llmoptimize.budget(max_usd=0.10, warn_only=True):
            ...
    """
    from llmoptimize.budget import BudgetContext
    return BudgetContext(max_usd, warn_only)


def step_budget(max_usd: float, warn_only: bool = False):
    """
    Context manager that enforces a per-call spend limit on patched AI API calls.

    Raises StepBudgetExceeded (or prints a warning if warn_only=True) if any
    single API call within the block costs more than max_usd.  Ideal for agent
    workflows where you want to cap the cost of individual reasoning or tool
    steps independently of total session spend.

    Example::

        with llmoptimize.step_budget(max_usd=0.005):
            # Each call must cost less than half a cent
            client.chat.completions.create(model="gpt-4o", ...)

        # warn instead of raising:
        with llmoptimize.step_budget(max_usd=0.005, warn_only=True):
            ...
    """
    from llmoptimize.budget import StepBudgetContext
    return StepBudgetContext(max_usd, warn_only)


def estimate(prompt: str, model=None, models: list = None, output_tokens: int = 100) -> dict:
    """
    Offline cost estimate — no server call, no API key needed.

    Single model:
        result = llmoptimize.estimate("Summarise this...", "gpt-4")
        # result["estimated_cost_usd"] -> 0.0042

    Multiple models (replaces compare()):
        result = llmoptimize.estimate("Summarise this...", models=["gpt-4", "gpt-4o-mini"])
        # result["rankings"] -> [{model, cost_usd, savings_pct}, ...]

    Uses the local pricing table (30+ models). Works fully offline.
    """
    from llmoptimize.patcher import _MODEL_COSTS, _calculate_cost

    input_tokens = max(1, len(prompt) // 4)

    # Resolve model list
    model_list = []
    if models:
        model_list = list(models)
    elif model:
        model_list = [model]
    else:
        return {"success": False, "error": "Provide model= or models="}

    results = []
    for m in model_list:
        pricing = _MODEL_COSTS.get(m, {"input": 0.001, "output": 0.002})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000
        results.append({
            "model": m,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_per_1k": pricing["input"],
            "output_cost_per_1k": pricing["output"],
            "estimated_cost_usd": round(cost, 8),
        })

    # Single model — flat result
    if len(results) == 1:
        r = results[0]
        r["success"] = True
        return r

    # Multiple models — ranked cheapest first
    results.sort(key=lambda x: x["estimated_cost_usd"])
    max_cost = results[-1]["estimated_cost_usd"] if results else 1
    for r in results:
        r["savings_pct"] = int((1 - r["estimated_cost_usd"] / max_cost) * 100) if max_cost else 0

    return {
        "success": True,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "rankings": results,
    }




def agent(
    max_usd: float = None,
    step_limit: float = None,
    warn_only: bool = False,
    max_steps: int = 50,
    silent: bool = False,
):
    """
    Unified agent pipeline monitor — one context manager for everything.

    Bundles: budget cap, per-step cap, loop detection, step classification,
    context growth alerts, framework detection, cost projection.
    All offline — no server calls, no API key needed.

    Example::

        with llmoptimize.agent(max_usd=0.50, step_limit=0.02):
            crew.kickoff()
        # Prints full analysis on exit

    Access stats inside the block::

        with llmoptimize.agent(max_usd=1.00) as a:
            client.chat.completions.create(...)
            print(a.spent, a.steps, a.remaining)
    """
    from llmoptimize.agent import AgentContext
    return AgentContext(
        max_usd=max_usd,
        step_limit=step_limit,
        warn_only=warn_only,
        max_steps=max_steps,
        silent=silent,
    )


__all__ = [
    "report", "track", "task", "new_session",
    "select_model", "check_loop", "analyze", "rag",
    "budget", "step_budget", "estimate", "agent",
    "BudgetExceeded", "StepBudgetExceeded",
    "__version__", "SESSION_ID", "RUN_ID", "SERVER_URL",
]
