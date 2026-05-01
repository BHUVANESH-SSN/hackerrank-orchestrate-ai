# Multi-Domain Support Triage Agent

A terminal-based multi-agent support triage system for the HackerRank Orchestrate challenge. Handles support tickets across **HackerRank**, **Claude (Anthropic)**, and **Visa** domains.

## Architecture

```
START
  │
  ▼
[intake]        → Parse ticket, extract signals (deterministic)
  │
  ▼
[classifier]    → Domain + product area classification (Claude API)
  │
  ▼
[risk]          → REPLY | ESCALATE decision (deterministic)
  │
  ├── ESCALATE ──→ [escalation] ──→ [output] ──→ END
  │
  └── REPLY ────→ [retrieval]
                      │
                      ▼
                  [synthesis]     (Claude API, grounded)
                      │
                      ▼
                  [faithfulness]  (Claude API, safety gate)
                      │
                  ├── pass ──→ [output] ──→ END
                  └── fail ──→ [escalation] ──→ END
```

## Tech Stack

| Layer | Library |
|---|---|
| LLM | Anthropic Claude (`claude-sonnet-4-20250514`) |
| Orchestration | LangGraph |
| Vector DB | ChromaDB |
| Embeddings | `all-MiniLM-L6-v2` |
| Sparse Search | BM25 (rank_bm25) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Terminal UI | Rich |
| Validation | Pydantic v2 |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env → add your ANTHROPIC_API_KEY

# 3. Build corpus (one-time, ~2-5 minutes)
python scripts/build_corpus.py

# 4a. Interactive mode
python main.py

# 4b. Batch mode (generates output.csv + log.txt)
python batch.py
```

## Project Structure

```
code/
├── agents/              # Agent nodes (intake, classifier, risk, etc.)
├── graph/               # LangGraph state machine
├── rag/                 # RAG pipeline (ChromaDB, BM25, cross-encoder)
├── models/              # Pydantic v2 schemas
├── config/              # Settings (pydantic-settings)
├── ui/                  # Rich terminal interface
├── utils/               # Logger, safety utilities
├── scripts/             # Corpus build script
├── main.py              # Interactive entry point
├── batch.py             # CSV batch processor
└── requirements.txt
```

## Output Format

The batch processor generates `output.csv` with columns:
- `issue` — original ticket text
- `subject` — ticket subject
- `company` — source company
- `response` — generated response (grounded in corpus)
- `product_area` — classified product area
- `status` — `Replied` or `Escalated`
- `request_type` — `product_issue`, `feature_request`, `bug`, or `invalid`
- `justification` — reasoning for the decision
