"""
RAG pipeline context manager for llmoptimize.

Wraps your RAG setup to give full visibility into:
- Document structure (count, size distribution)
- Chunk configuration (size, overlap efficiency)
- Embedding model cost optimization
- LLM model recommendation for RAG queries

Usage:
    loader = PyPDFLoader("docs/report.pdf")
    pages  = loader.load()

    with llmoptimize.rag(
        docs=pages,
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="text-embedding-ada-002",
        llm_model="gpt-4",
    ):
        chunks     = splitter.split_documents(pages)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        result     = qa_chain.run("What is X?")
    # Recommendations print automatically on exit
"""

import json
import urllib.request


class RAGContext:
    """
    Context manager that analyses your RAG pipeline configuration and
    prints actionable recommendations on exit.

    Accepts any doc format: LangChain Document objects, plain strings, or
    dicts with "text" / "content" / "page_content" keys.
    """

    def __init__(
        self,
        docs=None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        embedding_model: str = None,
        llm_model: str = None,
        query_type: str = None,   # "qa", "summarization", "extraction"
        silent: bool = False,
    ):
        self.docs            = docs
        self.chunk_size      = chunk_size
        self.chunk_overlap   = chunk_overlap
        self.embedding_model = embedding_model
        self.llm_model       = llm_model
        self.query_type      = query_type
        self.silent          = silent
        self._result         = None

    # ── Doc stats ─────────────────────────────────────────────────────────────

    def _compute_doc_stats(self) -> dict:
        if not self.docs:
            return {"doc_count": 0, "total_chars": 0, "avg_doc_chars": 0}

        sizes = []
        for doc in self.docs:
            if hasattr(doc, "page_content"):          # LangChain Document
                sizes.append(len(doc.page_content))
            elif isinstance(doc, str):
                sizes.append(len(doc))
            elif isinstance(doc, dict):
                text = (doc.get("text") or doc.get("content")
                        or doc.get("page_content", ""))
                sizes.append(len(text))

        if not sizes:
            return {"doc_count": 0, "total_chars": 0, "avg_doc_chars": 0}

        return {
            "doc_count":    len(sizes),
            "total_chars":  sum(sizes),
            "avg_doc_chars": sum(sizes) // len(sizes),
            "min_doc_chars": min(sizes),
            "max_doc_chars": max(sizes),
        }

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._analyze()
        return False

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _analyze(self):
        from llmoptimize import SERVER_URL, __version__

        payload = {
            **self._compute_doc_stats(),
            "chunk_size":      self.chunk_size,
            "chunk_overlap":   self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "llm_model":       self.llm_model,
            "query_type":      self.query_type,
        }

        try:
            body = json.dumps(payload).encode("utf-8")
            req  = urllib.request.Request(
                f"{SERVER_URL}/api/rag-analyze",
                data    = body,
                method  = "POST",
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent":   f"llmoptimize/{__version__}",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                self._result = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            self._result = {"success": False, "error": str(exc)}

        if not self.silent:
            self._print_recommendations()

    # ── Display ───────────────────────────────────────────────────────────────

    def _print_recommendations(self):
        r = self._result
        if not r or not r.get("success"):
            print(f"[llmoptimize] RAG analyze failed: {r.get('error', 'unknown error')}")
            return

        B  = "\033[1m"
        G  = "\033[92m"
        Y  = "\033[93m"
        C  = "\033[96m"
        E  = "\033[0m"

        print(f"\n{B}{'─'*56}{E}")
        print(f"{B}{C}  llmoptimize — RAG Pipeline Recommendations{E}")
        print(f"{B}{'─'*56}{E}")

        # Documents
        docs = r.get("doc_stats", {})
        if docs and docs.get("doc_count"):
            print(f"\n{B}📄  Documents{E}")
            print(f"   Count:       {docs['doc_count']}")
            print(f"   Avg size:    {docs['avg_doc_chars']:,} chars")
            print(f"   Total:       {docs['total_chars']:,} chars")

        # Chunking
        chunk = r.get("chunking")
        if chunk:
            icon = "✅" if chunk.get("status") == "ok" else "⚠️ "
            print(f"\n{B}✂️   Chunking{E}")
            print(f"   Current size:    {chunk.get('current_size', '—')}")
            rec = chunk.get("recommended_size")
            if rec:
                print(f"   Recommended:     {G}{rec}{E}")
            print(f"   {icon} {chunk.get('message', '')}")
            if chunk.get("overlap_issue"):
                print(f"   {Y}Overlap: {chunk['overlap_issue']}{E}")

        # Embedding
        emb = r.get("embedding")
        if emb:
            print(f"\n{B}🔢  Embedding Model{E}")
            print(f"   Current:     {emb.get('current_model', '—')}")
            if emb.get("suggested_model"):
                print(f"   Switch to:   {G}{emb['suggested_model']}{E}"
                      f"  ({emb.get('savings_percent', 0)}% cheaper)")
            else:
                print(f"   {G}Already optimal{E}")

        # LLM
        llm = r.get("llm")
        if llm:
            print(f"\n{B}🤖  LLM Model{E}")
            print(f"   Current:     {llm.get('current_model', '—')}")
            if llm.get("should_switch") and llm.get("suggested_model"):
                print(f"   Switch to:   {G}{llm['suggested_model']}{E}"
                      f"  ({llm.get('savings_percent', 0)}% cheaper)")
                print(f"   Reason:      {llm.get('reason', '')}")
            else:
                print(f"   {G}Good choice for this query type{E}")

        # Overall
        overall = r.get("overall", {})
        savings  = overall.get("estimated_monthly_savings_usd", 0)
        if savings:
            print(f"\n{B}{G}💰  Est. monthly savings if all tips applied: ${savings:.2f}{E}")

        print(f"{B}{'─'*56}{E}\n")

    # ── Access result programmatically ────────────────────────────────────────

    @property
    def result(self) -> dict:
        """Raw server response — available after the context block exits."""
        return self._result
