"""
llmoptimize/agent.py — unified agent pipeline context manager.

Usage::

    with llmoptimize.agent(max_usd=0.50, step_limit=0.02):
        crew.kickoff()
    # Prints full agent analysis on exit: framework, steps, loops,
    # step types, cost projection, recommendations — all offline.
"""

from llmoptimize.patcher import (
    BudgetExceeded, StepBudgetExceeded, get_local_session,
    _MODEL_COSTS, _calculate_cost,
)


# ── Local step classification ────────────────────────────────────────────────

_STEP_RULES = [
    ("planning",     ["plan", "think", "strategy", "decide", "outline", "design",
                      "brainstorm", "approach", "consider", "evaluate options"]),
    ("tool_call",    ["search", "fetch", "call", "execute", "run", "query",
                      "lookup", "retrieve", "get", "download", "api"]),
    ("synthesis",    ["summarize", "combine", "merge", "compile", "aggregate",
                      "consolidate", "final answer", "conclude", "wrap up"]),
    ("reflection",   ["review", "check", "verify", "validate", "assess",
                      "reconsider", "rethink", "was that correct", "let me re"]),
    ("coding",       ["code", "function", "implement", "debug", "fix bug",
                      "write python", "javascript", "class ", "def ", "import "]),
    ("conversation", ["hello", "hi ", "thanks", "please", "help me",
                      "can you", "what is", "how do", "explain"]),
]


def _classify_step(prompt_preview: str) -> str:
    """Classify a step type from its prompt preview (offline, no server)."""
    if not prompt_preview:
        return "general"
    lower = prompt_preview.lower()
    for step_type, keywords in _STEP_RULES:
        if any(kw in lower for kw in keywords):
            return step_type
    return "general"


# ── Local loop detection ─────────────────────────────────────────────────────

def _detect_loops(previews: list) -> list:
    """Simple offline loop detection on prompt previews."""
    if len(previews) < 3:
        return []

    loops = []

    # 1. Exact repeats — same preview 3+ times
    from collections import Counter
    counts = Counter(p.strip().lower()[:80] for p in previews if p.strip())
    for text, count in counts.items():
        if count >= 3:
            loops.append({
                "type": "exact_repeat",
                "text": text[:60],
                "count": count,
                "severity": "warning" if count < 5 else "critical",
            })

    # 2. Alternating pattern — A B A B
    if len(previews) >= 4:
        normalized = [p.strip().lower()[:80] for p in previews]
        for i in range(len(normalized) - 3):
            if (normalized[i] == normalized[i + 2] and
                normalized[i + 1] == normalized[i + 3] and
                normalized[i] != normalized[i + 1]):
                loops.append({
                    "type": "alternating",
                    "steps": f"{i+1}-{i+4}",
                    "severity": "warning",
                })
                break  # report once

    return loops


# ── Agent context manager ────────────────────────────────────────────────────

class AgentContext:
    """
    Unified agent pipeline monitor.

    Bundles: budget cap, per-step cap, loop detection, step classification,
    context growth alerts, framework detection, cost projection.
    All offline — no server calls, no API key needed.

    Args:
        max_usd:    Total spend cap for the block (optional).
        step_limit: Per-call spend cap (optional).
        warn_only:  If True, print warnings instead of raising exceptions.
        max_steps:  Project cost for this many steps (default 50).
        silent:     If True, suppress the exit report.
    """

    def __init__(
        self,
        max_usd: float = None,
        step_limit: float = None,
        warn_only: bool = False,
        max_steps: int = 50,
        silent: bool = False,
    ):
        self.max_usd    = max_usd
        self.step_limit = step_limit
        self.warn_only  = warn_only
        self.max_steps  = max_steps
        self.silent     = silent

        self._start_cost  = 0.0
        self._start_calls = 0

    def __enter__(self):
        session = get_local_session()
        self._start_cost  = session.total_cost
        self._start_calls = len(session.calls)

        if self.max_usd is not None:
            if self.max_usd <= 0:
                raise ValueError("max_usd must be positive")
            session.set_budget(self.max_usd)

        if self.step_limit is not None:
            if self.step_limit <= 0:
                raise ValueError("step_limit must be positive")
            session.set_step_budget(self.step_limit)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        session = get_local_session()

        # Clear guardrails
        if self.max_usd is not None:
            session.clear_budget()
        if self.step_limit is not None:
            session.clear_step_budget()

        # Gather calls made within this block
        block_calls = session.calls[self._start_calls:]
        spent = session.total_cost - self._start_cost

        # Handle budget exceptions
        if exc_type in (BudgetExceeded, StepBudgetExceeded):
            if self.warn_only:
                print(f"\n[llmoptimize] Agent budget warning: {exc_val}\n")
                if not self.silent:
                    self._print_report(block_calls, spent)
                return True  # suppress
            if not self.silent:
                self._print_report(block_calls, spent)
            return False  # re-raise

        if exc_type is None and not self.silent:
            self._print_report(block_calls, spent)

        return False

    def _print_report(self, calls, spent):
        """Print the agent analysis report."""
        n = len(calls)
        if n == 0:
            print("\n[llmoptimize] Agent: no API calls tracked in this block.\n")
            return

        # Framework
        from llmoptimize.patcher import _detected_framework
        framework = _detected_framework or "unknown"

        # Step types
        step_types = {}
        step_type_costs = {}
        previews = []
        models_used = set()
        total_tokens = 0

        for call in calls:
            models_used.add(call.model)
            tokens = call.prompt_tokens + call.completion_tokens
            total_tokens += tokens

            # Get preview from session record if available
            preview = getattr(call, "prompt_preview", "") or ""
            previews.append(preview)
            st = _classify_step(preview)
            step_types[st] = step_types.get(st, 0) + 1
            step_type_costs[st] = step_type_costs.get(st, 0.0) + call.cost

        # Loop detection
        loops = _detect_loops(previews)

        # Cost stats
        cost_per_step = spent / n
        projected = cost_per_step * self.max_steps
        avg_tokens = total_tokens / n

        # Context growth
        token_list = [c.prompt_tokens + c.completion_tokens for c in calls]
        if len(token_list) >= 2 and token_list[0] > 0:
            growth = token_list[-1] / token_list[0]
        else:
            growth = 1.0

        # Budget status
        if self.max_usd:
            budget_str = f"${spent:.4f} / ${self.max_usd:.4f}"
        else:
            budget_str = f"${spent:.4f}"

        # Step type summary
        st_parts = [f"{k} ({v})" for k, v in sorted(step_types.items(), key=lambda x: -x[1])]

        # Model recommendations per step type
        recommendations = []
        for st, cost in sorted(step_type_costs.items(), key=lambda x: -x[1]):
            # Find which models are used for this step type
            for call in calls:
                preview = getattr(call, "prompt_preview", "") or ""
                if _classify_step(preview) == st and call.model in _MODEL_COSTS:
                    model_cost = _MODEL_COSTS[call.model]
                    # Check if a cheaper model exists
                    from llmoptimize.patcher import _ALTERNATIVES
                    if call.model in _ALTERNATIVES:
                        alt = _ALTERNATIVES[call.model][0]
                        alt_cost = _MODEL_COSTS.get(alt, {})
                        if alt_cost:
                            savings = int((1 - (alt_cost.get("input", 0) / max(model_cost["input"], 0.000001))) * 100)
                            if savings > 10:
                                recommendations.append(
                                    f"  {st} steps use {call.model} — switch to {alt} (save ~{savings}%)"
                                )
                    break

        # Print report
        print(f"\n{'='*56}")
        print(f"  AGENT ANALYSIS")
        print(f"{'='*56}")
        print(f"  Framework:       {framework}")
        print(f"  Steps:           {n}")
        print(f"  Models used:     {len(models_used)} ({', '.join(sorted(models_used))})")
        print(f"  Avg tokens/step: {avg_tokens:.0f}")
        print(f"  Context growth:  {growth:.1f}x")
        print(f"  Step types:      {', '.join(st_parts)}")

        if loops:
            for loop in loops:
                if loop["type"] == "exact_repeat":
                    print(f"  Loop detected:   [{loop['severity'].upper()}] "
                          f"'{loop['text'][:40]}...' repeated {loop['count']}x")
                elif loop["type"] == "alternating":
                    print(f"  Loop detected:   [WARNING] alternating pattern at steps {loop['steps']}")
        else:
            print(f"  Loops:           none detected")

        print(f"  ---")
        print(f"  Total cost:      {budget_str}")
        print(f"  Cost per step:   ${cost_per_step:.6f}")
        print(f"  Projected {self.max_steps}-step: ${projected:.4f}")

        if self.step_limit:
            print(f"  Step limit:      ${self.step_limit:.4f}/call")

        if recommendations:
            print(f"  ---")
            for rec in recommendations[:3]:
                print(rec)

        print(f"{'='*56}\n")

    # ── Live properties ──────────────────────────────────────────────────────

    @property
    def spent(self) -> float:
        """Total spend within this agent block so far."""
        return get_local_session().total_cost - self._start_cost

    @property
    def remaining(self) -> float:
        """Remaining budget (None if no max_usd set)."""
        if self.max_usd is None:
            return None
        return max(0.0, self.max_usd - self.spent)

    @property
    def steps(self) -> int:
        """Number of API calls made within this block so far."""
        return len(get_local_session().calls) - self._start_calls

    @property
    def result(self) -> dict:
        """Programmatic access to current agent stats."""
        session = get_local_session()
        calls = session.calls[self._start_calls:]
        n = len(calls)
        spent = self.spent

        step_types = {}
        for call in calls:
            preview = getattr(call, "prompt_preview", "") or ""
            st = _classify_step(preview)
            step_types[st] = step_types.get(st, 0) + 1

        return {
            "steps": n,
            "total_cost": round(spent, 6),
            "cost_per_step": round(spent / n, 6) if n else 0,
            "projected_cost": round((spent / n) * self.max_steps, 4) if n else 0,
            "step_types": step_types,
            "models": list(set(c.model for c in calls)),
        }
