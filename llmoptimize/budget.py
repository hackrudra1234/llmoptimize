"""
llmoptimize/budget.py — spend guardrail context manager.

Usage::

    # Raise BudgetExceeded if any patched API call pushes spend over $0.10:
    with llmoptimize.budget(max_usd=0.10):
        client.chat.completions.create(...)

    # Warn only (no exception):
    with llmoptimize.budget(max_usd=0.10, warn_only=True):
        client.chat.completions.create(...)

The check is real-time: BudgetExceeded is raised inside the API call that
pushes spend over the limit, not just on context exit.
"""

from llmoptimize.patcher import BudgetExceeded, get_local_session  # noqa: F401


class BudgetContext:
    """
    Context manager that enforces a per-session spend limit.

    Tracks only the calls made *within* this context block (uses a cost
    baseline so pre-existing session spend is excluded).

    Args:
        max_usd:   Maximum allowed spend in USD for the block.
        warn_only: If True, print a warning instead of raising BudgetExceeded.
    """

    def __init__(self, max_usd: float, warn_only: bool = False):
        if max_usd <= 0:
            raise ValueError("max_usd must be positive")
        self.max_usd   = max_usd
        self.warn_only = warn_only
        self._start_cost = 0.0

    def __enter__(self):
        session = get_local_session()
        self._start_cost = session.total_cost
        session.set_budget(self.max_usd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        session = get_local_session()
        spent   = session.total_cost - self._start_cost
        session.clear_budget()

        if exc_type is BudgetExceeded:
            if self.warn_only:
                print(
                    f"\n[llmoptimize] Budget warning: "
                    f"spent ${spent:.4f} — limit was ${self.max_usd:.4f}\n"
                )
                return True   # suppress the exception

            # Re-raise with a cleaner message
            return False

        if exc_type is None:
            # Normal exit — print summary
            status = "OK" if spent <= self.max_usd else "OVER"
            print(
                f"\n[llmoptimize] Budget [{status}]: "
                f"spent ${spent:.4f} / ${self.max_usd:.4f}\n"
            )

        return False

    # Allow programmatic access inside the block
    @property
    def spent(self) -> float:
        """Spend accumulated so far within this budget block."""
        return get_local_session().total_cost - self._start_cost

    @property
    def remaining(self) -> float:
        """Remaining budget."""
        return max(0.0, self.max_usd - self.spent)
