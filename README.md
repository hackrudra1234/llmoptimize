# LLMOptimize

> **Cut your AI API costs — automatically.**
> One import. Zero config. No API key required.

```bash
pip install llmoptimize
```

---

## What It Does

LLMOptimize silently watches every AI API call your code makes and tells you how to save money — without touching your prompts or responses.

- **Zero setup** — just `import llmoptimize`
- **Works offline** — local pricing for 30+ models, no server needed
- **6 providers** — OpenAI, Anthropic, Groq, Gemini, Mistral, Cohere (sync + async)
- **Auto-detects frameworks** — LangChain, CrewAI, LlamaIndex, AutoGen, Haystack

---

## Quickstart

```python
import llmoptimize

import openai
client = openai.OpenAI()

response = client.chat.completions.create(
    model    = "gpt-4",
    messages = [{"role": "user", "content": "Summarize this article..."}],
)

llmoptimize.report()
```

That's it. Two lines added, full cost report with savings recommendations.

---

## The 6 Functions You Need

| Function | What it does | Needs server? |
|---|---|---|
| `.report()` | Cost report with recommendations | No |
| `.agent()` | Full agent pipeline monitor | No |
| `.rag()` | RAG pipeline optimizer | Yes |
| `.budget()` | Spend cap (raises on overspend) | No |
| `.estimate()` | Pre-call cost check (1 or many models) | No |
| `.analyze()` | Instant model recommendation | Yes |

Everything else (`step_budget`, `classify_step`, `check_loop`, context growth alerts, framework detection, secret redaction) is built into `agent()` automatically.

---

## `llmoptimize.agent()` — Agent Pipeline Monitor

One context manager that handles everything for agent workflows:
budget cap, per-step cap, loop detection, step classification, context growth alerts, framework detection, and cost projection. **All offline.**

```python
import llmoptimize

with llmoptimize.agent(max_usd=0.50, step_limit=0.02):
    crew.kickoff()

# Prints on exit:
# ========================================================
#   AGENT ANALYSIS
# ========================================================
#   Framework:       crewai
#   Steps:           12
#   Models used:     2 (gpt-4, gpt-4o-mini)
#   Avg tokens/step: 420
#   Context growth:  1.8x
#   Step types:      planning (3), tool_call (5), synthesis (4)
#   Loops:           none detected
#   ---
#   Total cost:      $0.0340 / $0.5000
#   Cost per step:   $0.002833
#   Projected 50-step: $0.1417
#   Step limit:      $0.0200/call
#   ---
#   tool_call steps use gpt-4 — switch to gpt-4o-mini (save ~97%)
# ========================================================
```

### What it bundles automatically

| Feature | How it works |
|---|---|
| Budget cap | `max_usd=0.50` — raises `BudgetExceeded` when total spend crosses limit |
| Per-step cap | `step_limit=0.02` — raises `StepBudgetExceeded` if any single call exceeds |
| Loop detection | Catches exact repeats (3+) and alternating patterns (A-B-A-B) |
| Step classification | Labels each step: planning, tool_call, synthesis, reflection, coding |
| Context growth | Warns when tokens double between consecutive calls |
| Framework detection | Auto-detects LangChain, CrewAI, etc. from call stack |
| Cost projection | Projects cost to 50 steps (configurable via `max_steps=`) |
| Model recommendations | Shows cheaper alternatives for expensive step types |

### Options

```python
# Monitor only (no budget limits):
with llmoptimize.agent():
    ...

# Warn instead of raising on budget overspend:
with llmoptimize.agent(max_usd=0.50, warn_only=True):
    ...

# Access stats inside the block:
with llmoptimize.agent(max_usd=1.00) as a:
    client.chat.completions.create(...)
    print(a.spent, a.steps, a.remaining)
    print(a.result)  # full stats dict

# Suppress printed report, use programmatically:
with llmoptimize.agent(silent=True) as a:
    ...
print(a.result["cost_per_step"])
```

---

## `llmoptimize.estimate()` — Cost Estimate (Offline)

Check what a prompt will cost **before** calling the API. Works fully offline using local pricing data. No API key needed.

```python
# Single model
result = llmoptimize.estimate("Summarise this 10-page report...", "gpt-4")
# result["estimated_cost_usd"]  -> 0.00792
# result["input_tokens"]        -> 194

# Compare multiple models at once
result = llmoptimize.estimate(
    "Classify this support ticket",
    models=["gpt-4", "gpt-4o", "gpt-4o-mini", "claude-3-haiku"],
)
for r in result["rankings"]:
    print(r["model"], f"${r['estimated_cost_usd']:.6f}", f"saves {r['savings_pct']}%")
# gpt-4o-mini   $0.000000   saves 99%
# claude-3-haiku $0.000001  saves 98%
# gpt-4o        $0.000016   saves 74%
# gpt-4         $0.000062   saves 0%
```

---

## `llmoptimize.budget()` — Spend Cap

Real-time spend limit — raises `BudgetExceeded` the instant a call pushes cost over the cap.

```python
from llmoptimize import BudgetExceeded

with llmoptimize.budget(max_usd=0.10):
    client.chat.completions.create(model="gpt-4", messages=[...])

# Warn instead of raising:
with llmoptimize.budget(max_usd=0.10, warn_only=True):
    ...

# Check spend inside the block:
with llmoptimize.budget(max_usd=1.00) as b:
    client.chat.completions.create(...)
    print(b.spent, b.remaining)
```

For agent workflows, prefer `llmoptimize.agent(max_usd=...)` which includes budget + everything else.

---

## `llmoptimize.rag()` — RAG Pipeline Optimizer

Wrap your RAG pipeline — get recommendations for chunk size, overlap, embedding model, and LLM.

```python
with llmoptimize.rag(
    docs=pages,
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="text-embedding-ada-002",
    llm_model="gpt-4",
):
    chunks = splitter.split_documents(pages)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    result = qa_chain.run("What is the revenue?")

# Prints: chunk size recommendation, embedding savings, LLM alternative
```

---

## Dry-Run Mode

Test your full code without spending a dollar. Mock responses, real recommendations.

```python
with llmoptimize.report:
    client.chat.completions.create(model="gpt-4", messages=[...])
# Report prints on exit — zero cost

with llmoptimize.task("my-pipeline", dry_run=True):
    ...  # named + dry-run
```

---

## CLI Audit

```bash
llmoptimize audit mycode.py        # static analysis + cost estimate
llmoptimize audit mycode.py --quiet  # one-line summary
```

---

## Supported Providers

| Provider | Library | Sync | Async | Embeddings |
|---|---|---|---|---|
| OpenAI | `openai` | yes | yes | yes |
| Anthropic | `anthropic` | yes | yes | — |
| Groq | `groq` | yes | yes | — |
| Google Gemini | `google-generativeai` | yes | yes | — |
| Mistral | `mistralai` | yes | yes | — |
| Cohere | `cohere` | yes | yes | yes |

All frameworks built on these (LangChain, CrewAI, LlamaIndex, etc.) are tracked automatically.

---

## Full API Reference

```python
import llmoptimize

# ── See your costs ───────────────────────────────────────────
llmoptimize.report()                       # full report
llmoptimize.report(interactive=False)      # plain text

# ── Agent pipeline (the main one) ────────────────────────────
with llmoptimize.agent(max_usd=0.50, step_limit=0.02):
    ...  # budget + loops + steps + alerts + framework detection

# ── RAG pipeline ─────────────────────────────────────────────
with llmoptimize.rag(docs=pages, chunk_size=1000, llm_model="gpt-4"):
    ...  # chunk/embed/LLM recommendations

# ── Budget (standalone) ──────────────────────────────────────
with llmoptimize.budget(max_usd=0.10):
    ...  # raises BudgetExceeded on overspend

# ── Cost planning (offline) ──────────────────────────────────
llmoptimize.estimate("prompt", "gpt-4")                    # single model
llmoptimize.estimate("prompt", models=["gpt-4", "gpt-4o"]) # compare

# ── Recommendation ───────────────────────────────────────────
llmoptimize.analyze("prompt", "gpt-4")    # get cheaper alternative

# ── Utilities ────────────────────────────────────────────────
llmoptimize.new_session()                 # reset tracking
llmoptimize.track(model, prompt_tokens, completion_tokens)  # manual
```

---

## Privacy

| Data | Stored locally | Sent to server |
|---|---|---|
| Your prompt text | Never | Never |
| Token counts | Yes | Yes (anonymised) |
| Model names | Yes | Yes |
| API keys | Never stored | Never sent |

The first 200 chars of prompts are used for classification only. **Secrets are auto-redacted** before leaving your machine.

```bash
export AIOPTIMIZE_SERVER_URL=""   # disable server tracking entirely
```

---

## FAQ

**Do I need to configure anything?**
No. `import llmoptimize` is all the setup required.

**Will it slow down my app?**
No. Tracking is fire-and-forget, never blocks the critical path.

**Does it work offline?**
Yes. `report()`, `agent()`, `budget()`, and `estimate()` all work without a server connection. Only `analyze()`, `rag()`, `select_model()`, and `check_loop()` use the server.

**Can I use it without an API key?**
Yes — use `with llmoptimize.report:` for dry-run mode. Mock responses, zero cost, full recommendations.

**What's the difference between `agent()` and `budget()`?**
`budget()` is just a spend cap. `agent()` includes the spend cap *plus* loop detection, step classification, context growth alerts, framework detection, cost projection, and model recommendations — all in one.

---

*LLMOptimize v3.4.0 — spend less, build more.*
