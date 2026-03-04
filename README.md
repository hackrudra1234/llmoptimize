# LLMOptimize

> **Cut your AI API costs — automatically.**
> One import. Zero config. No API key required.

```bash
pip install llmoptimize
```

---

## What It Does

LLMOptimize watches every AI API call your code makes and tells you which cheaper model to switch to — and **why**, in plain English.

- **No API key needed** to get recommendations
- **Never touches your prompts or responses** — read-only
- **Works with OpenAI, Anthropic, Groq, Gemini, Mistral, Cohere**
- **Zero setup** — just import it

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

That's it. Run your code normally. At the end you'll see:

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🚀  L L M O P T I M I Z E   R E P O R T  🚀            ║
║                                                              ║
║  Your AI Cost Optimization Summary                           ║
╚══════════════════════════════════════════════════════════════╝

📊 YOUR USAGE SUMMARY

🚀  Total API Calls Tracked
   1
   Optimized and analyzed

💰  Total Cost
   $0.0036
   Amount spent on AI API calls

💎  Potential Savings
   $0.0034
   That's 94% less than you could have spent!

────────────────────────────────────────────────────────────

📋 USAGE BY TYPE

  💬 Chat      1 call    $0.0036    → gpt-4o-mini (saves 94%)

────────────────────────────────────────────────────────────

💡 PERSONALIZED RECOMMENDATIONS

╭────────────────────────────────────────────────────────────╮
│ #1 Recommendation                                          │
├────────────────────────────────────────────────────────────┤
│ 🎯 Switch to: gpt-4o-mini                                  │
│ 💰 Save 94% on every call                                  │
│                                                            │
│ 💡 Why switch?                                             │
│   You called gpt-4 1 time this session                     │
│   gpt-4o-mini costs 94% less — saves ~$0.18 per 1,000      │
│   calls at this token size                                 │
│                                                            │
│ ⚡ How to fix:                                              │
│   Change  model="gpt-4"  →  "gpt-4o-mini"                  │
╰────────────────────────────────────────────────────────────╯
```

---

## Don't Have an API Key Yet? Use Dry-Run

Test your code flow and get cost advice **before spending a dollar**.
Wrap your code with `with llmoptimize.report:` — it intercepts calls,
returns mock responses so your code runs fully, and shows the report on exit.

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

## Track a Specific Task

Use `llmoptimize.task()` to get a separate report per pipeline stage.
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

Each block prints its own labelled report:

```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🚀  L L M O P T I M I Z E   R E P O R T  🚀            ║
║                                                              ║
║  Task: rag-pipeline                                          ║
╚══════════════════════════════════════════════════════════════╝

📌 Task: rag-pipeline

📋 USAGE BY TYPE

  📚 Embedding     1 call     $0.0000    → text-embedding-3-small (saves 80%)
  💬 Chat          1 call     $0.0005    → gpt-4o-mini (saves 99%)
  🧠 Reasoning     — not used this session
```

---

## CLI — Audit a File Before Running It

No code changes needed. Point it at any Python file and get instant advice:

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
   Why: You're using claude-3-opus ($90/1M tokens). For ticket
   classification claude-3-5-haiku costs $4.80/1M — same accuracy,
   18x cheaper.
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

`import llmoptimize` automatically patches every AI library you have installed. Nothing else needed.

| Provider | Library | Chat | Embeddings |
|---|---|---|---|
| OpenAI | `openai` | ✅ | ✅ |
| Anthropic | `anthropic` | ✅ | — |
| Groq | `groq` | ✅ | — |
| Google Gemini | `google-generativeai` | ✅ | — |
| Mistral | `mistralai` | ✅ | — |
| Cohere | `cohere` | ✅ | ✅ |

Pricing data for 60+ models including OpenAI, Anthropic, Groq, Gemini, Mistral, Cohere, Voyage AI, Jina AI, and AWS Bedrock.

---

## How Recommendations Work

Recommendations are **never just the cheapest model**. The engine checks capability tiers so you only see alternatives that deliver comparable results:

| Tier | Examples |
|---|---|
| Frontier | `gpt-4`, `claude-3-opus`, `o1` |
| Strong | `gpt-4o`, `claude-3-5-sonnet`, `gemini-1.5-pro` |
| Capable | `gpt-4o-mini`, `claude-3-haiku`, `gemini-1.5-flash` |
| Lightweight | `gemini-1.5-flash-8b`, `llama-3.1-8b-instant` |

It only recommends models **at most one tier below** what you're using — never a dramatic quality drop.

**How reasoning is generated:**
1. Pricing tables identify which model to switch to and savings %
2. AI analysis (Groq, on our server — no key needed from you) explains *why* in plain English
3. If the server is unreachable, cached reasoning from previous sessions is used
4. Final fallback: friendly plain-English text computed from pricing data

---

## Free Tier & License

LLMOptimize includes **500 free tracked calls** per machine.

```
🎉 Upgrade to continue:
   llmoptimize activate YOUR_LICENSE_KEY
```

### Activate a paid license

```bash
llmoptimize activate llmopt-xxxxxxxxxxxx
# ✅ License activated!  Plan: starter  |  500 calls/month
#    Valid through: 2026-04
```

The key is validated online and stored locally at `~/.aioptimize/license.json`.
No environment variables needed. Works for all future sessions on this machine.

### Remove a license

```bash
llmoptimize deactivate
# Remove license llmopt-xxxx...? (y/N): y
# ✅ License removed. Free tier limits restored.
```

### For servers / containers

```bash
export AIOPTIMIZE_LICENSE_KEY="llmopt-xxxxxxxxxxxx"
```

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

## Session Management

```python
llmoptimize.new_session()              # clear tracking, start fresh
llmoptimize.report(interactive=False)  # no menu prompt — useful in scripts
```

---

## Privacy

| Data | Stored locally | Sent to server |
|---|---|---|
| Your prompt text | Never | Never |
| Token counts | Yes | Yes |
| Model names | Yes | Yes |
| Cost figures | Yes | Yes |
| API keys | Never stored | Never sent |

Prompt text never leaves your machine. To disable server tracking entirely:

```bash
export AIOPTIMIZE_SERVER_URL=""
```

---

## FAQ

**Do I need to configure anything?**
No. `import llmoptimize` is all the setup required.

**Will it slow down my app?**
No. Tracking happens after your response is returned and never blocks the critical path.

**What if the recommendation server is unreachable?**
It falls back to local pricing data instantly. Your app is never affected.

**Does it work with LangChain / LlamaIndex?**
Yes — both use the underlying OpenAI/Anthropic SDKs which are patched automatically.

**Does it work with streaming?**
Yes. Token counts are recorded from the final usage block after streaming completes.

**Can I use it without an API key at all?**
Yes — use `dry_run=True` or `with llmoptimize.report:`. Your code runs end-to-end with mock responses. No API key, no cost, full recommendations.

**What's the difference between `task()` and `report()`?**
`task("name")` resets the session first (clean slate) and labels the output. `report()` shows everything tracked since the last reset. Use `task()` when benchmarking specific pipeline stages.

---

*LLMOptimize v3.2.3 — spend less, build more.*
