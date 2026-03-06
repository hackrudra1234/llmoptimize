# LLMOptimize

> **Cut your AI API costs — automatically.**
> One import. Zero config. No API key required.

```bash
pip install llmoptimize
```

---

## What It Does

LLMOptimize is a **complete AI cost optimization SDK** that silently watches every AI API call your code makes and surfaces actionable savings — without ever touching your prompts or responses.

- **Zero setup** — just `import llmoptimize`
- **No API key needed** for recommendations
- **Never reads your prompt text** — only token counts and model names
- **Works with OpenAI, Anthropic, Groq** and any framework built on them (LangChain, CrewAI, LlamaIndex, etc.)

**What it analyzes:**

| Feature | Description |
|---|---|
| 💰 Cost tracking | Real token usage → exact cost per call |
| 💡 Model recommendations | Heuristic + ML engine finds cheaper alternatives |
| 🔁 Loop detection | Catches agent loops before they drain your budget |
| 📚 RAG pattern detection | Identifies RAG pipelines and embedding savings |
| ⚡ Cache opportunities | Finds repeated prompts that should be cached |
| 🧠 ML model | Learns from your usage and improves over time |
| 🤖 Agent workflow | Multi-step tracking, context growth, step analytics |
| 📏 Context optimizer | Detects context window growth, compression tips |
| 🛡️ Security guardrails | Flags if API keys or sensitive data appear in prompts |
| 💸 Budget guardrail | Raises an exception the moment spend exceeds your limit |
| 🔍 Cost estimator | Pre-call USD estimate for any prompt + model (no API call) |
| ⚖️ Model comparator | Side-by-side cost table across any model list |

---

## Quickstart (2 lines)

```python
import llmoptimize          # ← add this at the top

import openai
client = openai.OpenAI()

response = client.chat.completions.create(
    model    = "gpt-4",
    messages = [{"role": "user", "content": "Summarize this article..."}],
)
print(response.choices[0].message.content)   # your real output, unchanged

llmoptimize.report()        # ← add this at the end
```

That's it. Your code runs exactly as before. At the end you'll see a full cost + optimization report.

---

## What the Report Shows

Running `llmoptimize.report()` produces a full interactive terminal report covering:

```
╔══════════════════════════════════════════════════════════════╗
║     🚀  L L M O P T I M I Z E   R E P O R T  🚀            ║
╚══════════════════════════════════════════════════════════════╝

📊 YOUR USAGE SUMMARY
  🚀 Total API Calls Tracked    3
  💰 Total Cost                 $0.0041
  💎 Potential Savings          $0.0039  (94% less!)

📋 USAGE BY TYPE
  💬 Chat        2 calls    $0.0040    → gpt-4o-mini (saves 94%)
  📚 Embedding   1 call     $0.0001    → text-embedding-3-small (saves 80%)

💡 PERSONALIZED RECOMMENDATIONS
  ╭────────────────────────────────────────────────────╮
  │ #1 Switch to: gpt-4o-mini   💰 Save 94%           │
  │ Why: You called gpt-4 2x — gpt-4o-mini costs 94%  │
  │      less, saves ~$0.18 per 1,000 calls            │
  │ Fix: model="gpt-4"  →  "gpt-4o-mini"              │
  ╰────────────────────────────────────────────────────╯

🤖 AGENT WORKFLOW ANALYSIS
  Steps tracked:         3
  Models used:           2 (multi-model workflow)
  Avg tokens/step:       420
  Context growth:        1.2x
  💡 Multi-model workflow — heuristic engine can recommend
     the cheapest model per step automatically

⚡ CACHING OPPORTUNITIES
  Found 1 repeated prompt pattern — caching could save ~$0.0008

🧠 ML MODEL STATUS
  ML collecting training data (12 samples — activates after 50+)
  Track more calls to unlock ML-powered model selection

📏 CONTEXT WINDOW OPTIMIZER
  Prompt sizes stable (avg 380 tokens) — context is well-managed

🔧 SDK UTILITIES
  llmoptimize.select_model(code)      → pick cheapest Groq model
  llmoptimize.check_loop(actions)     → detect agent loops
  llmoptimize.analyze(prompt, model)  → instant recommendation
```

---

## Dry-Run Mode — Plan Costs Before Spending

Test your full code flow and get savings advice **before spending a dollar**.
Wrap your code with `with llmoptimize.report:` — real API calls are intercepted,
mock responses are returned so your code runs fully, and the report prints on exit.

```python
import llmoptimize
import openai

client = openai.OpenAI(api_key="anything")   # not used in dry-run

with llmoptimize.report:
    # No real API calls — mock responses returned automatically
    client.embeddings.create(
        model = "text-embedding-3-large",
        input = ["RAG systems retrieve relevant documents."],
    )
    client.chat.completions.create(
        model    = "gpt-4",
        messages = [{"role": "user", "content": "Summarize this."}],
    )
# Report prints automatically when the block exits
```

When you're ready to go live, just remove the `with llmoptimize.report:` line — your code is already correct.

---

## Named Task Sessions

Use `llmoptimize.task()` to get a separate labelled report per pipeline stage.
Each block gets a clean slate, its own label, and optional dry-run mode.

```python
import llmoptimize
import openai

client = openai.OpenAI()

# Track real costs per stage
with llmoptimize.task("rag-pipeline"):
    chunks  = client.embeddings.create(model="text-embedding-3-large", input=["..."])
    summary = client.chat.completions.create(model="gpt-4", messages=[...])

# Plan costs before shipping — no real API calls
with llmoptimize.task("cost-planning", dry_run=True):
    client.chat.completions.create(model="gpt-4", messages=[...])
```

---

## SDK Utility Functions

These work anywhere in your code — no extra setup, no API key.

### Instant Model Recommendation

```python
result = llmoptimize.analyze(
    prompt = "Classify this support ticket as urgent or normal: ...",
    model  = "gpt-4o",
)

# result["recommendation"]["suggested_model"]           → "gpt-4o-mini"
# result["recommendation"]["estimated_savings_percent"] → 96
# result["recommendation"]["reasoning"]                 → "Classification task — cheaper models maintain 95%+ accuracy"
```

### Smart Model Selector for Code Tasks

Automatically picks the cheapest Groq model that can handle your code complexity.
Saves up to **84% vs always using the 70B model**.

```python
result = llmoptimize.select_model("""
    Extract the user name and email from this JSON string.
    Return as a Python dict.
""")

# result["selected_model"]                  → "llama-3.1-8b-instant"
# result["complexity_level"]                → "simple"
# result["model_info"]["best_for"]          → "Simple scripts, single API calls, basic logic"
# result["vs_heavy_model"]["savings_pct"]   → 84
# result["vs_heavy_model"]["message"]       → "Simple task — llama-3.1-8b-instant is 84% cheaper..."
```

| Complexity | Model Selected | Use Case |
|---|---|---|
| `simple` | `llama-3.1-8b-instant` | Single API calls, basic logic, extraction |
| `medium` | `openai/gpt-oss-20b` | Multi-step workflows, moderate complexity |
| `complex` | `llama-3.3-70b-versatile` | Complex agents, advanced reasoning |

### Agent Loop Detection

Catches repetitive agent behavior **before it drains your budget**.

```python
result = llmoptimize.check_loop([
    "search web for python docs",
    "read python docs",
    "search web for python docs",   # repeated!
    "read python docs",
    "search web for python docs",   # repeated again!
])

# result["loop_detected"]              → True
# result["loops"][0]["pattern_type"]   → "exact_repeat"
# result["loops"][0]["severity"]       → "warning"
# result["loops"][0]["recommendation"] → "Add a stop condition..."
# result["message"]                    → "⚠️ 1 loop pattern detected..."
```

Detects:
- **Exact repeats** — same action 3+ times
- **Circular patterns** — A→B→C→A→B→C
- **Alternating loops** — A→B→A→B
- **Semantic loops** — different words, same intent (AI-powered)

---

### RAG Pipeline Analyzer

Get recommendations for your full RAG setup — chunk size, overlap, embedding model, and LLM — by wrapping your pipeline with `llmoptimize.rag()`.

```python
import llmoptimize
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

loader = PyPDFLoader("docs/report.pdf")
pages  = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

with llmoptimize.rag(
    docs=pages,                               # your loaded documents
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="text-embedding-ada-002",
    llm_model="gpt-4",
):
    chunks      = splitter.split_documents(pages)
    vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
    result      = qa_chain.run("What is the revenue?")

# Prints automatically on exit:
#
# 📄  Documents:  12 docs, avg 8,432 chars
# ✂️   Chunking:   use 700 instead of 1000 (medium docs)
#                overlap 200 is wasteful — try 100-150
# 🔢  Embedding:  switch to text-embedding-3-small (80% cheaper)
# 🤖  LLM:        switch to gpt-3.5-turbo (97% cheaper for RAG Q&A)
# 💰  Est. monthly savings: $97.80
```

Access the raw result programmatically:

```python
with llmoptimize.rag(docs=pages, chunk_size=1000, llm_model="gpt-4") as r:
    ...  # your pipeline

r.result["chunking"]["recommended_size"]              # → 700
r.result["embedding"]["savings_percent"]              # → 80
r.result["llm"]["suggested_model"]                    # → "gpt-3.5-turbo"
r.result["overall"]["estimated_monthly_savings_usd"]  # → 97.8
```

**Accepts any doc format** — LangChain `Document` objects, plain strings, or dicts with `text` / `content` / `page_content` keys.

| Parameter | Description |
|---|---|
| `docs` | Your loaded documents |
| `chunk_size` | Your splitter's chunk_size |
| `chunk_overlap` | Your splitter's chunk_overlap |
| `embedding_model` | e.g. `"text-embedding-ada-002"` |
| `llm_model` | e.g. `"gpt-4"` |
| `query_type` | `"qa"` / `"summarization"` / `"extraction"` (optional) |
| `silent=True` | Suppress printed output, access via `.result` |

---

### Budget Guardrail

Enforce a per-block spend limit — `BudgetExceeded` is raised the instant a patched call
pushes spend over the cap. Use `warn_only=True` to log instead of raising.

```python
from llmoptimize import BudgetExceeded

try:
    with llmoptimize.budget(max_usd=0.05):
        client.chat.completions.create(model="gpt-4", messages=[...])
        client.chat.completions.create(model="gpt-4", messages=[...])   # may raise here
except BudgetExceeded as e:
    print(e)   # "Budget of $0.0500 exceeded (spent $0.0620 on this session)"

# Warn instead of raising:
with llmoptimize.budget(max_usd=0.10, warn_only=True):
    client.chat.completions.create(model="gpt-4", messages=[...])

# Check spend inside the block:
with llmoptimize.budget(max_usd=1.00) as b:
    client.chat.completions.create(...)
    print(b.spent, b.remaining)   # live USD figures
```

The check is **real-time** — it fires inside the API call that crosses the limit,
not just at exit. Only calls made *within* the block count toward the limit.

---

### Pre-Call Cost Estimate

Get the USD cost of a prompt + model combo before making the real call.
Uses the server pricing database (60+ models). No API key needed.

```python
result = llmoptimize.estimate(
    prompt        = "Summarise this 10-page report...",
    model         = "gpt-4",
    output_tokens = 200,          # expected output length (default 100)
)

# result["estimated_cost_usd"]  -> 0.00792
# result["input_tokens"]        -> 194
# result["input_cost_per_1k"]   -> 0.03
# result["output_cost_per_1k"]  -> 0.06
```

---

### Multi-Model Cost Comparison

Compare the cost of the same prompt across any list of models. No real API calls made.

`sort_by="cost"` (default) — cheapest-first, pure pricing.
`sort_by="value"` — best cost-per-quality-tier first. A tier-3 model at 3× the price of a tier-1 model gets the same value score — rewarding models that are both cheaper *and* more capable.

```python
# Cost-only ranking (default)
result = llmoptimize.compare(
    prompt        = "Classify this support ticket as urgent or normal.",
    models        = ["gpt-4", "gpt-4o", "gpt-4o-mini", "claude-3-haiku"],
    output_tokens = 50,
)
for r in result["rankings"]:
    print(r["model"], r["cost_usd"], r["quality_tier"])
# claude-3-haiku  0.00000125  tier=2
# gpt-4o-mini     0.00000125  tier=2
# gpt-4o          0.00001625  tier=3
# gpt-4           0.00006200  tier=4

# Value ranking — balances cost and capability
result = llmoptimize.compare(
    prompt        = "Classify this support ticket as urgent or normal.",
    models        = ["gpt-4", "gpt-4o", "gpt-4o-mini", "claude-3-haiku"],
    sort_by       = "value",
)
for r in result["rankings"]:
    print(r["model"], r["value_score"], r["value_savings_vs_worst_pct"])
# gpt-4o-mini  0.0000000625  94%
# gpt-4o       0.0000054167  99%  ← strong model, reasonable value
# claude-3-haiku ...
```

Quality tiers: `1` = lightweight · `2` = capable · `3` = strong · `4` = frontier

---

## LangChain & CrewAI

No changes needed to your agents or chains. Just add `import llmoptimize` at the top — all LLM calls inside chains and agents are tracked automatically.

```python
import llmoptimize       # ← one line at the top

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

llm   = ChatOpenAI(model="gpt-4")
chain = LLMChain(llm=llm, prompt=my_prompt)
chain.invoke({"input": "..."})

llmoptimize.report()    # see exactly what the chain spent and how to cut it
```

```python
import llmoptimize       # ← one line at the top

from crewai import Agent, Task, Crew

researcher = Agent(role="Researcher", llm="gpt-4", ...)
crew = Crew(agents=[researcher], tasks=[...])
crew.kickoff()

llmoptimize.report()    # full agent workflow analysis included
```

---

## CLI — Audit a File Before Running It

No code changes needed. Point it at any Python file:

```bash
llmoptimize audit mycode.py
```

```
╔════════════════════════════════════════════════════════════════╗
║                   🤖 AI CODE AUDIT REPORT                     ║
╚════════════════════════════════════════════════════════════════╝

📄 File: mycode.py

📊 SUMMARY
   API calls found:    7
   Issues detected:    4
   Models used:        gpt-4, claude-3-opus

   Est. monthly cost:  $342  (at 1,000 runs/month)
   Potential savings:  $298  (87%)

🔍 RECOMMENDATIONS

🔴 Line 42 — claude-3-opus
   Switch to: claude-3-5-haiku  |  saves 95%
   Why: Classification task — claude-3-5-haiku is 18x cheaper
        with comparable accuracy for structured output.
```

**Options:**

```bash
llmoptimize audit mycode.py             # full report
llmoptimize audit mycode.py --quiet     # one-line summary
llmoptimize audit mycode.py --force     # skip cache, always re-analyze
llmoptimize stats                       # show cache statistics
llmoptimize clear-cache                 # clear cached results
```

---

## Supported Providers

`import llmoptimize` automatically patches every AI library you have installed.

| Provider | Library | Chat | Embeddings |
|---|---|---|---|
| OpenAI | `openai` | ✅ | ✅ |
| Anthropic | `anthropic` | ✅ | — |
| Groq | `groq` | ✅ | — |

Pricing data for 60+ models: OpenAI, Anthropic, Groq, Gemini, Mistral, Cohere, Voyage AI, Jina AI, AWS Bedrock.

---

## How Recommendations Work

Recommendations use a **3-layer engine** — never just the cheapest model:

```
1. Heuristic engine (5ms)
   ↓ Keyword-based task detection (classification → gpt-4o-mini, etc.)

2. ML model (10ms)
   ↓ Trained on real accept/reject decisions from all SDK users
   ↓ Learns which models work best per prompt category + complexity

3. Crowd-sourced patterns (instant)
   ↓ Global anonymised data: which model won for this task type?
```

Capability tiers are respected — you'll never see a recommendation that drops more than one quality tier:

| Tier | Examples |
|---|---|
| Frontier | `gpt-4`, `claude-3-opus`, `o1` |
| Strong | `gpt-4o`, `claude-3-5-sonnet`, `gemini-1.5-pro` |
| Capable | `gpt-4o-mini`, `claude-3-haiku`, `gemini-1.5-flash` |
| Lightweight | `gemini-1.5-flash-8b`, `llama-3.1-8b-instant` |

---

## Session Management

```python
llmoptimize.new_session()              # clear tracking, start fresh
llmoptimize.report(interactive=False)  # no menu prompt — useful in scripts
```

> **Jupyter / VS Code Interactive Window note:**
> The Python kernel stays alive between cells, so `llmoptimize` accumulates
> calls across all cells. Call `llmoptimize.new_session()` before each test run:
>
> ```python
> import llmoptimize
> llmoptimize.new_session()   # ← reset before testing a new model
>
> resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
> llmoptimize.report()
> ```
>
> Regular `.py` scripts reset automatically on each run.

---

## Manual Tracking

For custom or self-hosted models not auto-patched:

```python
llmoptimize.track(
    model             = "my-custom-model",
    prompt_tokens     = 400,
    completion_tokens = 120,
    provider          = "custom",
)

llmoptimize.report()
```

---

## Free Tier & License

LLMOptimize includes **500 free tracked calls** per machine.

### Activate a paid license

```bash
llmoptimize activate llmopt-xxxxxxxxxxxx
# ✅ License activated!  Plan: starter  |  500 calls/month
#    Valid through: 2026-04
```

### Remove a license

```bash
llmoptimize deactivate
# ✅ License removed. Free tier limits restored.
```

### For servers / containers

```bash
export AIOPTIMIZE_LICENSE_KEY="llmopt-xxxxxxxxxxxx"
```

---

## Privacy

| Data | Stored locally | Sent to server |
|---|---|---|
| Your prompt text | Never | Never |
| Token counts | Yes | Yes (anonymised) |
| Model names | Yes | Yes |
| Cost figures | Yes | Yes |
| API keys | Never stored | Never sent |

Only the **first 100 chars of your prompt** are optionally sent for category classification (e.g. "classification" vs "summarization") — never the full text, never stored.

To disable server tracking entirely:

```bash
export AIOPTIMIZE_SERVER_URL=""
```

---

## Full API Reference

```python
# ── Core ──────────────────────────────────────────────────────
import llmoptimize

llmoptimize.report()                   # interactive report
llmoptimize.report(interactive=False)  # plain text, no menu
llmoptimize.new_session()              # reset all tracking
llmoptimize.track(model, prompt_tokens, completion_tokens)

# ── Dry-run / named sessions ──────────────────────────────────
with llmoptimize.report:               # dry-run + report on exit
    ...

with llmoptimize.report(interactive=False):
    ...

with llmoptimize.task("pipeline-name"):       # named session, live calls
    ...

with llmoptimize.task("plan", dry_run=True):  # dry-run + labelled report
    ...

# ── Smart utilities ───────────────────────────────────────────
result = llmoptimize.analyze(prompt, model)   # instant recommendation
result = llmoptimize.select_model(code)       # pick cheapest Groq model
result = llmoptimize.check_loop(actions)      # detect agent loops
result = llmoptimize.estimate(prompt, model)  # pre-call cost estimate
result = llmoptimize.compare(prompt, models)  # side-by-side cost table

# ── Budget guardrail ───────────────────────────────────────────
with llmoptimize.budget(max_usd=0.10):        # raises BudgetExceeded if over
    ...
with llmoptimize.budget(0.10, warn_only=True): # warns instead of raising
    ...

# ── RAG pipeline ──────────────────────────────────────────────
with llmoptimize.rag(
    docs=pages,
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="text-embedding-ada-002",
    llm_model="gpt-4",
) as r:
    ...  # your RAG pipeline
# r.result -> full recommendations dict
```

---

## FAQ

**Do I need to configure anything?**
No. `import llmoptimize` is all the setup required.

**Will it slow down my app?**
No. Tracking happens after your response is returned and never blocks the critical path. All server calls are fire-and-forget.

**What if the recommendation server is unreachable?**
It falls back to local pricing data instantly. Your app is never affected.

**Does it work with LangChain / LlamaIndex / CrewAI?**
Yes — they all use the underlying OpenAI/Anthropic/Groq SDKs which are patched automatically.

**Does it work with streaming?**
Yes. Token counts are recorded from the final usage block after streaming completes.

**Can I use it without an API key at all?**
Yes — use `dry_run=True` or `with llmoptimize.report:`. Your code runs end-to-end with mock responses. No API key, no cost, full recommendations.

**What's the difference between `task()` and `report()`?**
`task("name")` resets the session first and labels the output. `report()` shows everything tracked since the last reset. Use `task()` when benchmarking specific pipeline stages.

**What does `select_model()` do?**
It sends your code/task description to the server's GroqModelSelector — a hybrid rule-based + caching engine that picks the cheapest Groq model capable of handling the complexity. No API key needed.

**What does `check_loop()` do?**
It sends a list of action strings to the server's LoopDetector, which uses AI-powered + rule-based detection for exact repeats, circular patterns (A→B→C→A), alternating loops, and semantic loops (different words, same intent). Returns which steps are looping and a recommendation to fix it.

**What does `budget()` do?**
It enforces a real-time spend limit. As soon as any patched call inside the block pushes cost above `max_usd`, a `BudgetExceeded` exception is raised immediately — not on exit. Use `warn_only=True` to log a warning instead. `.spent` and `.remaining` are readable inside the block.

**What does `estimate()` do?**
It returns a USD cost breakdown for a prompt + model before you make the real call. Uses the server's 60-model pricing database. Useful for sanity-checking expensive prompts or building cost-aware routing. No API key needed.

**What does `compare()` do?**
It calculates the cost of a prompt across a list of models (no real API calls) and returns a ranked table. Default `sort_by="cost"` sorts cheapest-first. Use `sort_by="value"` for quality-aware ranking — each model's cost is divided by its capability tier (1–4), so a powerful model at a fair price beats a cheap but weak one. Each result includes `quality_tier` and `value_score`.

**What does `llmoptimize.rag()` do?**
It's a context manager you wrap around your RAG pipeline. Pass your loaded docs and your splitter/model config — on exit it sends the stats to the server's `RAGPipelineAnalyzer` which recommends the optimal chunk size, overlap, embedding model, and LLM for your document sizes and query type. No changes to your pipeline code needed.

---

*LLMOptimize v3.4.0 — spend less, build more.*
