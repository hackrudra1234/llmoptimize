"""
llmoptimize — Test All 3 Usage Modes
=====================================

Run this file to see all 3 modes in action:

    python test_embedding.py

Or test the CLI audit separately:

    llmoptimize audit test_embedding.py

No real API key needed for Mode 1 and Mode 2.
Mode 3 (live tracking) needs a real OPENAI_API_KEY.
"""

import llmoptimize
import openai

# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — DRY-RUN  (no API key needed, no real calls, no cost)
#
# Use this to plan costs BEFORE shipping.
# Your code runs end-to-end with mock responses.
# Remove the `with llmoptimize.report:` line to go live.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("MODE 1: Dry-run (with llmoptimize.report)")
print("="*60)

client = openai.OpenAI(api_key="not-needed-in-dry-run")

with llmoptimize.report:
    # Expensive embedding — will recommend a cheaper alternative
    client.embeddings.create(
        model = "text-embedding-3-large",
        input = [
            "RAG systems retrieve relevant documents before generation.",
            "Embeddings turn text into vectors for similarity search.",
            "Costs can be reduced by switching to smaller models.",
        ],
    )

    # Expensive chat — will recommend gpt-4o-mini
    client.chat.completions.create(
        model    = "gpt-4",
        messages = [{"role": "user", "content": "Summarise the key points."}],
    )

    client.chat.completions.create(
        model    = "gpt-4o",
        messages = [{"role": "user", "content": "Translate this to French."}],
    )

# Report printed automatically when the block exits ↑


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — TASK DRY-RUN  (named session, isolated report per task)
#
# Use this to benchmark specific pipeline stages.
# dry_run=True means no real API calls.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("MODE 2: Task dry-run (llmoptimize.task)")
print("="*60)

with llmoptimize.task("rag-pipeline", dry_run=True):
    # Step 1 — embed documents
    client.embeddings.create(
        model = "text-embedding-ada-002",
        input = ["Document chunk 1", "Document chunk 2", "Document chunk 3"],
    )

    # Step 2 — generate answer from retrieved context
    client.chat.completions.create(
        model    = "gpt-4",
        messages = [
            {"role": "system",  "content": "Answer based on the provided context."},
            {"role": "user",    "content": "What is the capital of France?"},
        ],
    )

# Report printed automatically, labelled "Task: rag-pipeline" ↑


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 — LIVE TRACKING  (real API calls, needs OPENAI_API_KEY)
#
# Comment this section out if you don't have a key.
# llmoptimize silently watches real calls and reports at the end.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("MODE 3: Live tracking (real calls)")
print("="*60)

import os
if os.getenv("OPENAI_API_KEY"):
    live_client = openai.OpenAI()   # reads OPENAI_API_KEY automatically

    response = live_client.chat.completions.create(
        model    = "gpt-4o",
        messages = [{"role": "user", "content": "Say hello in one word."}],
    )
    print("Live response:", response.choices[0].message.content)

    llmoptimize.report()   # show cost + recommendations for real calls above

else:
    print("⚠  OPENAI_API_KEY not set — skipping live tracking demo.")
    print("   Set it and re-run to test Mode 3.\n")
