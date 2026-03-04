"""
LLMOptimize - Futuristic Interactive Client

FIXES vs previous version:
  1. Replaced 'import requests' with stdlib urllib — requests was removed
     from install_requires so importing it crashed on fresh installs
  2. reason text now wraps across multiple lines instead of truncating at 52
  3. _show_interactive_report passes full reason, no pre-truncation at [:50]
  4. SERVER_URL reads AIOPTIMIZE_SERVER_URL env var — was hardcoded
  5. LLMOptimize._get_session_id() now has OSError guard (read-only fs safe)
  6. Recommendations grouped by model type — chat models never shown for
     embedding-only sessions and vice versa
  7. _generate_html produces a real HTML report instead of a stub
"""

import os
import uuid
import json
import time
import sys
import textwrap
import urllib.request
from datetime import datetime

# Windows consoles default to cp1252 which can't render box-drawing / emoji.
# Reconfigure stdout/stderr to UTF-8 so the dashboard displays correctly.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 — best-effort

SERVER_URL = os.getenv("AIOPTIMIZE_SERVER_URL", "https://aioptimize.up.railway.app")
_SHARE_URL = SERVER_URL   # used in share text — stays in sync with env var

EMBEDDING_MODEL_PREFIXES = (
    "text-embedding",
    "embed-",
    "embedding-",
    "mistral-embed",
    "voyage-",        # Voyage AI: voyage-3, voyage-3-lite, voyage-code-3 etc.
    "jina-embedding", # Jina AI:   jina-embeddings-v3, jina-embeddings-v2-base-en
    "amazon.titan-embed",  # AWS Bedrock: amazon.titan-embed-text-v2:0
)

def _is_embedding_model(model: str) -> bool:
    m = model.lower()
    return any(m.startswith(p) or p in m for p in EMBEDDING_MODEL_PREFIXES)


# ── Terminal colors ───────────────────────────────────────────────────────────

def _tty() -> bool:
    return sys.stdout.isatty()

class Colors:
    PURPLE    = '\033[95m' if _tty() else ''
    BLUE      = '\033[94m' if _tty() else ''
    CYAN      = '\033[96m' if _tty() else ''
    GREEN     = '\033[92m' if _tty() else ''
    YELLOW    = '\033[93m' if _tty() else ''
    RED       = '\033[91m' if _tty() else ''
    BOLD      = '\033[1m'  if _tty() else ''
    UNDERLINE = '\033[4m'  if _tty() else ''
    END       = '\033[0m'  if _tty() else ''
    BG_PURPLE = '\033[45m' if _tty() else ''
    BG_GREEN  = '\033[42m' if _tty() else ''


# ── UI helpers ────────────────────────────────────────────────────────────────

def print_gradient_header(task_name: str = ""):
    subtitle = f"Task: {task_name}" if task_name else "Your AI Cost Optimization Summary"
    # pad subtitle to fit inside the box (inner width = 62)
    sub_pad  = max(0, 62 - len(subtitle))
    sub_line = f"║  {subtitle}" + " " * sub_pad + "║"
    print(f"""
{Colors.PURPLE}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🚀  L L M O P T I M I Z E   R E P O R T  🚀            ║
║                                                              ║
{sub_line}
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
{Colors.END}
    """)
    time.sleep(0.3)


def print_loading_animation(text="Analyzing your data"):
    sys.stdout.write(f"\n{Colors.CYAN}{text}")
    for _ in range(3):
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(0.3)
    sys.stdout.write(f" ✓{Colors.END}\n")
    time.sleep(0.2)


def print_stat_card(icon, label, value, subtext="", animate=True):
    print(f"\n{Colors.BOLD}{icon}  {label}{Colors.END}")
    if animate and _tty():
        if isinstance(value, float):
            steps = 20
            for i in range(steps + 1):
                current = (value * i) / steps
                sys.stdout.write(f"\r{Colors.GREEN}{Colors.BOLD}   ${current:.2f}{Colors.END}")
                sys.stdout.flush()
                time.sleep(0.02)
            print()
        elif isinstance(value, int):
            steps = min(20, value)
            for i in range(steps + 1):
                current = int((value * i) / steps) if steps else 0
                sys.stdout.write(f"\r{Colors.CYAN}{Colors.BOLD}   {current}{Colors.END}")
                sys.stdout.flush()
                time.sleep(0.02)
            print()
        else:
            print(f"{Colors.YELLOW}{Colors.BOLD}   {value}{Colors.END}")
    else:
        if isinstance(value, float):
            print(f"{Colors.GREEN}{Colors.BOLD}   ${value:.2f}{Colors.END}")
        elif isinstance(value, int):
            print(f"{Colors.CYAN}{Colors.BOLD}   {value}{Colors.END}")
        else:
            print(f"{Colors.YELLOW}{Colors.BOLD}   {value}{Colors.END}")

    if subtext:
        print(f"{Colors.END}   {subtext}")
    time.sleep(0.2)


def _build_why_lines(rec: dict) -> list:
    """
    Build Why lines entirely from rec payload fields — nothing hardcoded.

    Fields used:
      from_model, model, savings_pct, call_count,
      avg_in, avg_out, saving_per_1k, is_embedding, reason
    """
    from_model    = rec.get("from_model", "your current model")
    alt_model     = rec.get("model", "")
    pct           = rec.get("savings_pct", 0) or int(rec.get("savings_text", "0").replace("%", "") or 0)
    count         = rec.get("call_count", 0)
    avg_in        = rec.get("avg_in", 0)
    avg_out       = rec.get("avg_out", 0)
    saving_per_1k = rec.get("saving_per_1k", 0)
    is_embedding  = rec.get("is_embedding", False)
    reason        = rec.get("reason", "")

    lines = []

    # Line 1 — usage this session
    if count:
        times = f"{count} time{'s' if count != 1 else ''}"
        lines.append(f"You called {from_model} {times} this session")
    else:
        lines.append(f"Currently using {from_model}")

    # Line 2 — token size per call, in honest units (tokens, not words)
    if avg_in and is_embedding:
        lines.append(f"Each call embedded ~{avg_in} tokens of text")
    elif avg_in:
        lines.append(f"Each call used ~{avg_in} tokens in, ~{avg_out} tokens out")

    # Line 3 — saving derived from real pricing, expressed per 1,000 calls
    # (standard API metric — avoids projecting session count as daily volume)
    if saving_per_1k and saving_per_1k >= 0.0001:
        lines.append(
            f"{alt_model} costs {pct}% less — "
            f"saves ~${saving_per_1k:.4f} per 1,000 calls at this token size"
        )
    else:
        lines.append(f"{alt_model} costs {pct}% less for the same output")

    # Line 4 — full reason text when available (e.g. from hybrid ML/heuristic engine),
    # otherwise fall back to a generic quality note derived from savings %.
    if reason:
        lines.append(reason)
    elif pct >= 90:
        lines.append("Best for straightforward tasks — test on your workload first")
    elif pct >= 70:
        lines.append("Similar capability tier, lower price point")
    else:
        lines.append("Marginal quality difference at this savings level")

    return lines


def _build_action_line(rec: dict) -> str:
    """One-line code change the user needs to make."""
    from_model = rec.get("from_model", "")
    alt_model  = rec.get("model", "")
    if from_model and alt_model:
        return f'Change  model="{from_model}"  →  "{alt_model}"'
    return f'Switch to model="{alt_model}"'


def print_recommendation_card(rec_num: int, rec: dict):
    """
    Print a recommendation card with:
    - Model name + savings %
    - Why section (human-readable, not raw token counts)
    - How to fix (one actionable code line)
    """
    BOX_WIDTH   = 60
    INNER_WIDTH = BOX_WIDTH - 4

    model   = rec.get("model", "")
    savings = rec.get("savings_text", "")

    why_lines   = _build_why_lines(rec)
    action_line = _build_action_line(rec)

    def pad(text, extra=0):
        visible_len = len(text) + extra
        return " " * max(0, BOX_WIDTH - visible_len - 1)

    rows = [
        f"{Colors.PURPLE}╭{'─' * BOX_WIDTH}╮{Colors.END}",

        # Header
        (f"{Colors.PURPLE}│{Colors.END} {Colors.BOLD}#{rec_num} Recommendation"
         + pad(f"#{rec_num} Recommendation")
         + f"{Colors.PURPLE}│{Colors.END}"),

        f"{Colors.PURPLE}├{'─' * BOX_WIDTH}┤{Colors.END}",

        # Model name
        (f"{Colors.PURPLE}│{Colors.END} {Colors.CYAN}{Colors.BOLD}🎯 Switch to: {model}{Colors.END}"
         + pad(f"🎯 Switch to: {model}", 1)
         + f"{Colors.PURPLE}│{Colors.END}"),

        # Savings
        (f"{Colors.PURPLE}│{Colors.END} {Colors.GREEN}💰 Save {savings} on every call{Colors.END}"
         + pad(f"💰 Save {savings} on every call", 1)
         + f"{Colors.PURPLE}│{Colors.END}"),

        f"{Colors.PURPLE}│{Colors.END}" + " " * BOX_WIDTH + f"{Colors.PURPLE}│{Colors.END}",

        # Why header
        (f"{Colors.PURPLE}│{Colors.END} {Colors.YELLOW}💡 Why switch?{Colors.END}"
         + pad("💡 Why switch?", 1)
         + f"{Colors.PURPLE}│{Colors.END}"),
    ]

    # Why body lines
    for line in why_lines:
        wrapped = textwrap.wrap(line, width=INNER_WIDTH) or [""]
        for wline in wrapped:
            p = BOX_WIDTH - len(wline) - 3
            rows.append(
                f"{Colors.PURPLE}│{Colors.END}   {wline}" + " " * p + f"{Colors.PURPLE}│{Colors.END}"
            )

    rows.append(f"{Colors.PURPLE}│{Colors.END}" + " " * BOX_WIDTH + f"{Colors.PURPLE}│{Colors.END}")

    # Action header
    rows.append(
        f"{Colors.PURPLE}│{Colors.END} {Colors.BOLD}⚡ How to fix:{Colors.END}"
        + pad("⚡ How to fix:")
        + f"{Colors.PURPLE}│{Colors.END}"
    )

    # Action line — wrapped, monospace-style indent
    for aline in (textwrap.wrap(action_line, width=INNER_WIDTH) or [action_line]):
        p = BOX_WIDTH - len(aline) - 3
        rows.append(
            f"{Colors.PURPLE}│{Colors.END}   {Colors.CYAN}{aline}{Colors.END}" + " " * p + f"{Colors.PURPLE}│{Colors.END}"
        )

    rows.append(f"{Colors.PURPLE}╰{'─' * BOX_WIDTH}╯{Colors.END}")

    print("\n" + "\n".join(rows))
    time.sleep(0.3)


def print_divider():
    print(f"\n{Colors.PURPLE}{'─' * 60}{Colors.END}\n")


def print_savings_celebration(savings):
    if savings > 5:
        print(f"""
{Colors.GREEN}{Colors.BOLD}
    ✨ ═══════════════════════════════════════ ✨
    
         🎉  AMAZING SAVINGS DETECTED!  🎉
         
              You saved ${savings:.2f}!
              
    ✨ ═══════════════════════════════════════ ✨
{Colors.END}
        """)
        time.sleep(0.5)


def interactive_prompt():
    # Skip prompt when stdout is not a real terminal (scripts, CI, pipes)
    if not sys.stdout.isatty():
        return "4"
    print(f"\n{Colors.BOLD}What would you like to do?{Colors.END}\n")
    print(f"  {Colors.CYAN}[1]{Colors.END} 📊 Open full interactive report in browser")
    print(f"  {Colors.CYAN}[2]{Colors.END} 📤 Share your savings")
    print(f"  {Colors.CYAN}[3]{Colors.END} 💾 Export detailed report")
    print(f"  {Colors.CYAN}[4]{Colors.END} ✅ Done (continue coding)")
    try:
        choice = input(f"\n{Colors.YELLOW}Your choice (1-4): {Colors.END}")
        return choice.strip()
    except (EOFError, KeyboardInterrupt):
        return "4"


# ── HTTP helpers (stdlib only — no requests dependency) ───────────────────────

def _http_get(url: str, timeout: int = 10):
    """GET request using stdlib urllib. Returns parsed JSON or None."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "llmoptimize-dashboard/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass
    return None


def _http_post(url: str, payload: dict, timeout: int = 5):
    """POST request using stdlib urllib. Returns parsed JSON or None."""
    try:
        body = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url,
            data    = body,
            method  = "POST",
            headers = {
                "Content-Type": "application/json",
                "User-Agent":   "llmoptimize-dashboard/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass
    return None


# ── Main client ───────────────────────────────────────────────────────────────

class LLMOptimize:
    def __init__(self):
        # Use RUN_ID from __init__ — matches what _track_call sends.
        # Falls back to reading the session file if imported standalone.
        try:
            from llmoptimize import RUN_ID as _run_id
            self.session_id = _run_id
        except (ImportError, AttributeError):
            self.session_id = self._get_session_id()

    def _get_session_id(self) -> str:
        """Get or create session ID. FIX: OSError guard for read-only filesystems."""
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

    def track(self, model, prompt_tokens, completion_tokens):
        """Track an API call."""
        return _http_post(
            f"{SERVER_URL}/track",
            {
                "model":             model,
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": completion_tokens,
                "session_id":        self.session_id,
            },
        )

    def report(self, interactive=True, task_name: str = ""):
        """
        Generate terminal report.

        Priority:
        1. Local MagicSession (in-memory, instant — populated by the patcher
           as real API calls happen in the same process)
        2. Server fetch (used when called from a different process or after
           a with llmoptimize.report(): dry-run block)
        """
        if interactive:
            print_gradient_header(task_name)

        # ── 1. Try local in-memory session first ─────────────────────────
        data = self._local_session_data()

        # ── 2. Fall back to server if local is empty ──────────────────────
        if not data:
            if interactive:
                print_loading_animation("Fetching your usage data")
            data = _http_get(f"{SERVER_URL}/session/{self.session_id}")

        if not data or data.get("total_calls", 0) == 0:
            print(f"{Colors.YELLOW}No calls recorded in this session yet.{Colors.END}")
            print(f"  Add  import llmoptimize  at the top of your script and run it.")
            print(f"  Start fresh anytime:  {Colors.BOLD}llmoptimize.new_session(){Colors.END}")
            return

        if interactive:
            self._show_interactive_report(data, task_name=task_name)
        else:
            self._show_basic_report(data)

    def _local_session_data(self) -> dict:
        """
        Build a report dict from the in-memory MagicSession (magic.py).
        Same shape as the server /session/{id} response.
        Returns empty dict if no calls recorded yet.
        """
        try:
            from server.core.magic import get_session
            from server.core.calculator import (
                calculate_cost, get_alternatives, is_embedding_model,
                REASONING_MODELS,
            )
            session = get_session()
            if not session.calls:
                return {}

            total_calls = len(session.calls)
            total_cost  = session.total_cost

            # aggregate by model
            model_agg = {}
            for call in session.calls:
                m = call.model
                if m not in model_agg:
                    model_agg[m] = {"count": 0, "prompt_tokens": 0, "completion_tokens": 0}
                model_agg[m]["count"]             += 1
                model_agg[m]["prompt_tokens"]     += call.prompt_tokens
                model_agg[m]["completion_tokens"] += call.completion_tokens

            # build recommendations (mirrors server logic)
            all_candidates = []
            seen = set()
            top = sorted(model_agg.items(), key=lambda x: -x[1]["count"])[:3]
            for model, stats in top:
                count   = stats["count"]
                avg_in  = max(1, stats["prompt_tokens"]     // count)
                avg_out =        stats["completion_tokens"] // count
                model_is_embedding  = is_embedding_model(model)
                model_is_reasoning  = model in REASONING_MODELS
                current_cost        = calculate_cost(model, avg_in, avg_out)

                for alt_info in get_alternatives(model):
                    alt_model = alt_info["model"]
                    if is_embedding_model(alt_model) != model_is_embedding:
                        continue
                    if alt_model in seen:
                        continue
                    seen.add(alt_model)
                    alt_cost = calculate_cost(alt_model, avg_in, avg_out)
                    if alt_cost >= current_cost:
                        continue
                    pct           = int((current_cost - alt_cost) / current_cost * 100)
                    saving_per_1k = round((current_cost - alt_cost) * 1000, 6)
                    tier_drop     = alt_info.get("tier_drop", 0)
                    if tier_drop == 0:
                        quality_note = "Same capability tier — comparable results at lower cost"
                    else:
                        quality_note = (
                            "Handles most tasks equally well at lower cost — "
                            "test on your workload before switching"
                        )
                    all_candidates.append({
                        "model":         alt_model,
                        "from_model":    model,
                        "savings_pct":   pct,
                        "savings_text":  f"{pct}%",
                        "avg_in":        avg_in,
                        "avg_out":       avg_out,
                        "call_count":    count,
                        "saving_per_1k": saving_per_1k,
                        "is_embedding":  model_is_embedding,
                        "is_reasoning":  model_is_reasoning,
                        "tier_drop":     tier_drop,
                        "reason": (
                            f"{quality_note}. "
                            f"{alt_model} is {pct}% cheaper at ~{avg_in} tokens/call "
                            f"(saves ${saving_per_1k:.4f} per 1,000 calls)"
                        ),
                    })

            recommendations = sorted(all_candidates, key=lambda x: -x["savings_pct"])[:3]

            # ── Hybrid enrichment: ask server for ML/heuristic recommendations ──
            # Best-effort — falls back to static recs silently if server is down.
            try:
                enriched = []
                for model, stats in top[:2]:
                    sample_preview = next(
                        (c.prompt_preview for c in session.calls
                         if c.model == model and c.prompt_preview),
                        "",
                    )
                    hybrid = _http_post(
                        f"{SERVER_URL}/api/recommend",
                        {"current_model": model, "prompt_preview": sample_preview},
                        timeout=2,
                    )
                    if (
                        hybrid
                        and hybrid.get("success")
                        and hybrid.get("recommendation", {}).get("should_switch")
                    ):
                        hrec      = hybrid["recommendation"]
                        alt_model = hrec.get("suggested_model", "")
                        if alt_model and alt_model != model:
                            count   = stats["count"]
                            avg_in  = max(1, stats["prompt_tokens"] // count)
                            avg_out = stats["completion_tokens"] // count
                            cur_c   = calculate_cost(model, avg_in, avg_out)
                            alt_c   = calculate_cost(alt_model, avg_in, avg_out)
                            pct     = hrec.get(
                                "estimated_savings_percent",
                                int((cur_c - alt_c) / cur_c * 100) if cur_c > 0 else 0,
                            )
                            enriched.append({
                                "model":         alt_model,
                                "from_model":    model,
                                "savings_pct":   pct,
                                "savings_text":  f"{pct}%",
                                "avg_in":        avg_in,
                                "avg_out":       avg_out,
                                "call_count":    count,
                                "saving_per_1k": round((cur_c - alt_c) * 1000, 6),
                                "is_embedding":  is_embedding_model(model),
                                "reason":        hrec.get(
                                    "reasoning",
                                    f"Switch from {model} to {alt_model} — {pct}% cheaper",
                                ),
                                "confidence":    hrec.get("confidence", 0.7),
                            })
                if enriched:
                    seen_hybrid = {r["model"] for r in enriched}
                    for r in recommendations:
                        if r["model"] not in seen_hybrid:
                            enriched.append(r)
                    recommendations = sorted(enriched, key=lambda x: -x.get("savings_pct", 0))[:3]
            except Exception:
                pass  # static recs already set above — use them as-is

            # ── Category breakdown ─────────────────────────────────────────────
            categories: dict = {}
            for model, stats in model_agg.items():
                count   = stats["count"]
                avg_in  = max(1, stats["prompt_tokens"] // count)
                avg_out = stats["completion_tokens"]    // count
                cost    = calculate_cost(model, avg_in, avg_out) * count
                if model in REASONING_MODELS:
                    cat = "reasoning"
                elif is_embedding_model(model):
                    cat = "embedding"
                else:
                    cat = "chat"
                entry = categories.setdefault(cat, {"calls": 0, "cost": 0.0, "models": []})
                entry["calls"] += count
                entry["cost"]   = round(entry["cost"] + cost, 6)
                if model not in entry["models"]:
                    entry["models"].append(model)
            # attach best alternative per category for the summary line
            for cat, entry in categories.items():
                best = next(
                    (r for r in recommendations
                     if (cat == "embedding" and     r.get("is_embedding") and not r.get("is_reasoning"))
                     or (cat == "reasoning" and     r.get("is_reasoning"))
                     or (cat == "chat"      and not r.get("is_embedding") and not r.get("is_reasoning"))),
                    None,
                )
                entry["best_alt"] = best["model"]       if best else None
                entry["best_pct"] = best["savings_pct"] if best else 0

            # Only count savings once per from_model — prevents double-counting when
            # multiple recommendations target the same source model.
            best_per_from = {}
            for r in recommendations:
                fm = r.get("from_model", "")
                if fm and r.get("avg_in"):
                    sv = (calculate_cost(fm, r["avg_in"], r["avg_out"]) -
                          calculate_cost(r["model"], r["avg_in"], r["avg_out"]))
                    if fm not in best_per_from or sv > best_per_from[fm]:
                        best_per_from[fm] = sv
            total_savings   = sum(best_per_from.values()) if best_per_from else 0.0
            total_savings   = max(0.0, min(total_savings, total_cost))
            avg_savings_pct = int(total_savings / total_cost * 100) if total_cost > 0 else 0

            return {
                "total_calls":     total_calls,
                "total_cost":      round(total_cost, 4),
                "total_savings":   round(total_savings, 4),
                "avg_savings_pct": avg_savings_pct,
                "recommendations": recommendations,
                "categories":      categories,
            }

        except Exception:
            return {}

    def _show_interactive_report(self, data, task_name: str = ""):
        print_divider()
        print(f"{Colors.BOLD}{Colors.BLUE}📊 YOUR USAGE SUMMARY{Colors.END}\n")

        print_stat_card("🚀", "Total API Calls Tracked",
                        data.get("total_calls", 0), "Optimized and analyzed")
        print_stat_card("💰", "Total Cost",
                        data.get("total_cost", 0.0), "Amount spent on AI API calls")

        savings = data.get("total_savings", 0.0)
        print_savings_celebration(savings)
        print_stat_card("💎", "Potential Savings", savings,
                        f"That's {data.get('avg_savings_pct', 0)}% less than you could have spent!")

        # ── Usage by type ─────────────────────────────────────────────────────
        categories = data.get("categories", {})
        if categories:
            print_divider()
            if task_name:
                print(f"{Colors.BOLD}{Colors.PURPLE}📌 Task: {task_name}{Colors.END}\n")
            print(f"{Colors.BOLD}{Colors.BLUE}📋 USAGE BY TYPE{Colors.END}\n")
            cat_icons  = {"embedding": "📚", "chat": "💬", "reasoning": "🧠"}
            cat_labels = {"embedding": "Embedding", "chat": "Chat", "reasoning": "Reasoning"}
            for cat in ("chat", "embedding", "reasoning"):
                icon  = cat_icons[cat]
                label = cat_labels[cat]
                if cat in categories:
                    entry     = categories[cat]
                    calls_str = f"{entry['calls']} call{'s' if entry['calls'] != 1 else ''}"
                    cost_str  = f"${entry['cost']:.4f}"
                    best_alt  = entry.get("best_alt")
                    best_pct  = entry.get("best_pct", 0)
                    if best_alt:
                        alt_str = (
                            f"→ {Colors.CYAN}{best_alt}{Colors.END}"
                            f" (saves {Colors.GREEN}{best_pct}%{Colors.END})"
                        )
                        print(f"  {icon} {Colors.BOLD}{label:12}{Colors.END}"
                              f"  {calls_str:15}  {cost_str:10}  {alt_str}")
                    else:
                        print(f"  {icon} {Colors.BOLD}{label:12}{Colors.END}"
                              f"  {calls_str:15}  {cost_str}")
                else:
                    print(f"  {icon} {Colors.BOLD}{label:12}{Colors.END}"
                          f"  {Colors.YELLOW}— not used this session{Colors.END}")
            print()

        print_divider()
        print(f"{Colors.BOLD}{Colors.BLUE}💡 PERSONALIZED RECOMMENDATIONS{Colors.END}\n")

        recommendations = data.get("recommendations", [])

        # FIX: group by model type so embedding tasks don't show chat model recs
        # Classify by from_model (what the user currently uses) — not the alt model.
        # e.g. rec {from_model: text-embedding-3-large, model: text-embedding-3-small}
        # should appear under Embedding, not Chat.
        def _is_emb_rec(r):
            src_model = r.get("from_model") or r.get("model", "")
            return _is_embedding_model(src_model)

        embedding_recs = [r for r in recommendations if     _is_emb_rec(r)]
        chat_recs      = [r for r in recommendations if not _is_emb_rec(r)]

        # Decide which group is primary based on what the session mostly used
        # If ALL top-model recs are embedding → only show embedding section
        all_recs = embedding_recs or chat_recs
        if embedding_recs and chat_recs:
            # Mixed session — show both sections with headers
            print(f"{Colors.BOLD}── Embedding Models ──{Colors.END}")
            for i, rec in enumerate(embedding_recs[:2], 1):
                print_recommendation_card(i, rec)
            print(f"{Colors.BOLD}── Chat Models ──{Colors.END}")
            for i, rec in enumerate(chat_recs[:2], 1):
                print_recommendation_card(i, rec)
        else:
            for i, rec in enumerate(all_recs[:3], 1):
                print_recommendation_card(i, rec)

        print_divider()

        choice = interactive_prompt()
        if choice == "1":
            print(f"\n{Colors.GREEN}Opening full report in browser...{Colors.END}")
            self._open_html_report(data)
        elif choice == "2":
            print(f"\n{Colors.GREEN}Share link:{Colors.END}")
            print(f"💬 I saved ${savings:.2f} with @LLMOptimize! 🚀 {_SHARE_URL}")
        elif choice == "3":
            fname = f"llmoptimize_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            self._export_text_report(data, fname)
            print(f"\n{Colors.GREEN}Report exported to: {fname}{Colors.END}")
        else:
            print(f"\n{Colors.GREEN}Happy coding! Keep optimizing! 🚀{Colors.END}\n")

    def _show_basic_report(self, data):
        print("\n💰 COST REPORT")
        print(f"Total Calls:       {data.get('total_calls', 0)}")
        print(f"Total Cost:        ${data.get('total_cost', 0):.4f}")
        print(f"Potential Savings: ${data.get('total_savings', 0):.4f} ({data.get('avg_savings_pct', 0)}%)")
        print("\n💡 RECOMMENDATIONS")
        for rec in data.get("recommendations", [])[:3]:
            print(f"  ✓ Switch to {rec['model']} — {rec.get('reason', '')}")
        print()

    def _open_html_report(self, data):
        import tempfile, webbrowser
        html = self._generate_html(data)
        # FIX: specify utf-8 encoding explicitly — Windows defaults to cp1252
        # which cannot encode emojis, causing UnicodeEncodeError on write
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html",
                                         encoding="utf-8") as f:
            f.write(html)
            temp_path = f.name
        webbrowser.open("file://" + temp_path)
        print(f"\n✨ Report opened in browser!")

    def _export_text_report(self, data, filename: str):
        lines = [
            "LLMOptimize Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
            f"Total Calls:       {data.get('total_calls', 0)}",
            f"Total Cost:        ${data.get('total_cost', 0):.4f}",
            f"Potential Savings: ${data.get('total_savings', 0):.4f} ({data.get('avg_savings_pct', 0)}%)",
            "",
            "RECOMMENDATIONS",
            "-" * 50,
        ]
        for i, rec in enumerate(data.get("recommendations", [])[:3], 1):
            lines.append(f"#{i} {rec.get('model', '')} — save {rec.get('savings_text', '')}")
            lines.append(f"   {rec.get('reason', '')}")
            lines.append("")
        with open(filename, "w") as f:
            f.write("\n".join(lines))

    def _generate_html(self, data):
        """Full HTML report — mirrors all detail shown in the terminal interactive report."""

        def _esc(s):
            return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # ── Category table ────────────────────────────────────────────────────
        categories = data.get("categories", {})
        cat_icons  = {"embedding": "📚", "chat": "💬", "reasoning": "🧠"}
        cat_labels = {"embedding": "Embedding", "chat": "Chat", "reasoning": "Reasoning"}
        cat_rows   = ""
        for cat in ("chat", "embedding", "reasoning"):
            icon  = cat_icons[cat]
            label = cat_labels[cat]
            if cat in categories:
                entry    = categories[cat]
                calls    = entry["calls"]
                cost     = entry["cost"]
                best_alt = entry.get("best_alt", "")
                best_pct = entry.get("best_pct", 0)
                alt_cell = (
                    f'<span class="alt-model">{_esc(best_alt)}</span>'
                    f' <span class="savings-pct">saves {best_pct}%</span>'
                    if best_alt else "—"
                )
                cat_rows += (
                    f'<tr><td>{icon} {label}</td>'
                    f'<td>{calls} call{"s" if calls != 1 else ""}</td>'
                    f'<td>${cost:.4f}</td>'
                    f'<td>{alt_cell}</td></tr>'
                )
            else:
                cat_rows += (
                    f'<tr class="unused-cat"><td>{icon} {label}</td>'
                    f'<td colspan="3">— not used this session</td></tr>'
                )
        cats_html = (
            f'<h2>📋 Usage by Type</h2>'
            f'<table class="cat-table">'
            f'<thead><tr><th>Type</th><th>Calls</th><th>Cost</th><th>Best Alternative</th></tr></thead>'
            f'<tbody>{cat_rows}</tbody></table>'
        ) if categories else ""

        recs_html = ""
        for i, rec in enumerate(data.get("recommendations", [])[:3], 1):
            from_model    = _esc(rec.get("from_model", ""))
            alt_model     = _esc(rec.get("model", ""))
            savings_text  = _esc(rec.get("savings_text", ""))
            reason        = _esc(rec.get("reason", ""))
            avg_in        = rec.get("avg_in", 0)
            avg_out       = rec.get("avg_out", 0)
            call_count    = rec.get("call_count", 0)
            saving_per_1k = rec.get("saving_per_1k", 0)
            is_embedding  = rec.get("is_embedding", False)

            usage_line = (
                f"~{avg_in} tokens/call (embedding)"
                if is_embedding
                else f"~{avg_in} tokens in, ~{avg_out} tokens out"
            )
            calls_line = (
                f"{call_count} call{'s' if call_count != 1 else ''} this session"
                if call_count else "—"
            )
            cost_line = (
                f"saves ~${saving_per_1k:.4f} per 1,000 calls at this token size"
                if saving_per_1k and saving_per_1k >= 0.0001
                else f"{savings_text} cheaper per call"
            )

            recs_html += f"""
        <div class="card">
          <div class="card-header">
            <span class="rec-num">#{i}</span>
            <span class="model-badge">{alt_model}</span>
            <span class="savings-badge">💰 Save {savings_text}</span>
          </div>
          <div class="card-body">
            <div class="section">
              <div class="section-title">🔄 Change</div>
              <code class="code-line">model=&quot;{from_model}&quot; &nbsp;→&nbsp; &quot;{alt_model}&quot;</code>
            </div>
            <div class="section">
              <div class="section-title">📊 Your usage</div>
              <div class="detail-row"><span class="label">Calls this session</span><span>{calls_line}</span></div>
              <div class="detail-row"><span class="label">Token size</span><span>{usage_line}</span></div>
              <div class="detail-row"><span class="label">Cost impact</span><span>{_esc(cost_line)}</span></div>
            </div>
            <div class="section">
              <div class="section-title">💡 Why switch?</div>
              <p class="reason-text">{reason}</p>
            </div>
          </div>
        </div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LLMOptimize Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body   {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0f0f1a; color: #e0e0ff; padding: 2rem; line-height: 1.5; }}
  h1     {{ color: #a78bfa; font-size: 1.8rem; margin-bottom: 0.25rem; }}
  .ts    {{ color: #6b7280; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  /* ── stats row ── */
  .stats {{ display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }}
  .stat  {{ background: #1a1a2e; border: 1px solid #4c1d95; border-radius: 10px;
            padding: 1rem 1.25rem; flex: 1; min-width: 130px; }}
  .stat-value {{ font-size: 1.8rem; font-weight: 700; color: #34d399; }}
  .stat-label {{ font-size: 0.8rem; color: #9ca3af; margin-top: 0.2rem; }}
  /* ── section heading ── */
  h2 {{ color: #c4b5fd; font-size: 1.1rem; margin: 1.5rem 0 0.75rem; }}
  /* ── recommendation cards ── */
  .card {{ background: #1a1a2e; border: 1px solid #4c1d95; border-radius: 10px;
           margin-bottom: 1rem; overflow: hidden; }}
  .card-header {{ display: flex; align-items: center; gap: 0.75rem;
                  background: #231a3a; padding: 0.75rem 1.25rem; flex-wrap: wrap; }}
  .rec-num      {{ color: #7c3aed; font-weight: 700; font-size: 1rem; }}
  .model-badge  {{ background: #2e1065; color: #c4b5fd; border-radius: 6px;
                   padding: 0.2rem 0.6rem; font-weight: 600; font-size: 0.9rem; }}
  .savings-badge{{ background: #064e3b; color: #34d399; border-radius: 6px;
                   padding: 0.2rem 0.6rem; font-weight: 600; font-size: 0.85rem; margin-left: auto; }}
  .card-body    {{ padding: 1rem 1.25rem; }}
  .section      {{ margin-bottom: 1rem; }}
  .section-title{{ font-size: 0.78rem; font-weight: 600; color: #9ca3af;
                   text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem; }}
  .code-line    {{ display: inline-block; background: #0f172a; color: #7dd3fc;
                   font-family: "Fira Code", "Cascadia Code", monospace; font-size: 0.88rem;
                   padding: 0.3rem 0.6rem; border-radius: 5px; border: 1px solid #1e3a5f; }}
  .detail-row   {{ display: flex; justify-content: space-between; font-size: 0.88rem;
                   padding: 0.2rem 0; border-bottom: 1px solid #1e1b4b; }}
  .detail-row:last-child {{ border-bottom: none; }}
  .label        {{ color: #9ca3af; }}
  .reason-text  {{ font-size: 0.9rem; color: #d1d5db; }}
  /* ── category table ── */
  .cat-table    {{ width: 100%; border-collapse: collapse; margin-bottom: 1.5rem;
                   font-size: 0.9rem; }}
  .cat-table th {{ text-align: left; color: #9ca3af; font-weight: 600;
                   padding: 0.4rem 0.75rem; border-bottom: 1px solid #4c1d95; }}
  .cat-table td {{ padding: 0.5rem 0.75rem; border-bottom: 1px solid #1e1b4b; }}
  .cat-table tr:last-child td {{ border-bottom: none; }}
  .unused-cat td {{ color: #4b5563; font-style: italic; }}
  .alt-model    {{ color: #7dd3fc; font-weight: 600; }}
  .savings-pct  {{ color: #34d399; font-size: 0.82rem; }}
</style>
</head>
<body>
<h1>🚀 LLMOptimize Report</h1>
<p class="ts">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

<div class="stats">
  <div class="stat"><div class="stat-value">{data.get('total_calls', 0)}</div><div class="stat-label">Total Calls</div></div>
  <div class="stat"><div class="stat-value">${data.get('total_cost', 0):.4f}</div><div class="stat-label">Total Cost</div></div>
  <div class="stat"><div class="stat-value">${data.get('total_savings', 0):.4f}</div><div class="stat-label">Potential Savings</div></div>
  <div class="stat"><div class="stat-value">{data.get('avg_savings_pct', 0)}%</div><div class="stat-label">Savings Rate</div></div>
</div>

{cats_html}
<h2>💡 Recommendations</h2>
{recs_html}
</body>
</html>"""