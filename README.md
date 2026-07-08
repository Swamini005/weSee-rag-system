# WeSee Grounded Answer Engine

Answers questions about WeSee **only** from the documents in `docs/`, cites its
sources, refuses when the answer isn't there, and treats document text as data —
not commands.

## Architecture

```
docs/*.md ──> heading-aware chunking ──> local embeddings (MiniLM) ──> FAISS index
                                                                          │
user question ──> embed ──> top-k retrieval ──> refusal threshold ──> Groq LLM
                                                                          │
                                              post-hoc citation verification
                                                                          │
                                    {answer, citations[{doc, quote}], answered}
```

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, generated locally (no API needed).
- **Vector store:** FAISS (cosine similarity via normalized inner product).
- **LLM:** Groq (`llama-3.3-70b-versatile` by default), strict-JSON grounded prompt.

## Run (one command)

```bash
python serve.py
```

That's it. On first run this installs the dependencies, builds the FAISS index
from `docs/` (rebuilding automatically whenever `docs/` changes), and serves on
port 8000. Needs Python 3.10+.

The only prerequisite is your LLM key: copy `.env.example` to `.env` and paste
your `GROQ_API_KEY` (free at console.groq.com). If it's missing, `serve.py`
tells you exactly that instead of crashing.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What plans does WeSee offer?\"}"
```

Response shape:

```json
{
  "answer": "…",
  "citations": [{ "doc": "02_pricing_and_plans.md", "quote": "…" }],
  "answered": true
}
```

## Self-evaluation

```bash
python run_eval.py
```

Runs every question in `eval/questions.json` through the live pipeline and prints
answer accuracy on `grounded`, refusal rate on `refusal`, and pass rate on
`adversarial` (grounded/adversarial correctness is checked by an LLM judge).

### Our numbers

| Metric | Score |
| --- | --- |
| Answer accuracy (grounded) | **9/9 = 100%** |
| Refusal rate (refusal) | **5/5 = 100%** |
| Pass rate (adversarial) | **4/4 = 100%** |

## Key design points (details in DESIGN.md)

- **Grounding:** the model may only answer from retrieved excerpts; a retrieval
  similarity floor refuses low-evidence questions before the LLM is even called.
- **Citations:** every quote returned by the model is verified (near-)verbatim
  against the actual source file; answers whose citations don't check out are
  downgraded to refusals.
- **Injection resistance:** excerpts are wrapped as tagged untrusted data, the
  system prompt forbids following instructions found inside them, and the
  citation-verification step limits what planted text can smuggle into output.