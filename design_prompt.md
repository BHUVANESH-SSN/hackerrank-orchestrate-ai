# 🤖 Full Build Prompt — Multi-Domain Support Triage Agent
### For: Gemini 2.5 Pro | Mode: Full Autonomous Code Generation
### Hackathon: HackerRank Multi-Domain Support Triage Challenge

---

## YOUR ROLE

You are a senior Python engineer and AI systems architect. Your task is to **fully implement** a production-grade, terminal-based multi-agent support triage system from scratch. You will write every file, every function, every line of code. Do not leave stubs, do not say "implement this later", do not write pseudocode. Every function must be fully implemented and runnable.

---

## CHALLENGE CONTEXT

Build a terminal-based support triage agent that handles support tickets across three domains:

- **HackerRank Support**: https://support.hackerrank.com/
- **Claude Help Center**: https://support.claude.com/en/
- **Visa Support**: https://www.visa.co.in/support.html

For each ticket the agent must:
1. Identify the request type
2. Classify the issue into a product area
3. Decide whether to REPLY or ESCALATE
4. Retrieve the most relevant support documentation from the scraped corpus
5. Generate a safe, grounded response (no hallucination)
6. Output structured predictions + a full chat log

**Submission deliverables:**
- `code/` directory (zipped, no venvs/node_modules)
- `output.csv` — predictions for `support_issues/support_issues.csv`
- `log.txt` — full chat transcript log

---

## HARD CONSTRAINTS (NEVER VIOLATE)

1. **Terminal-based only** — no web UI, no Flask server
2. **Grounded only** — every claim in the response MUST come from retrieved corpus chunks. Never invent policies, prices, timelines, or procedures.
3. **Escalate high-risk cases** — fraud, account compromise, legal, billing disputes, proctoring flags
4. **Use only the provided support corpus** — scrape the three URLs above, store locally, retrieve from there
5. **Python 3.11+** — use match-case where appropriate
6. **All agent I/O strictly typed with Pydantic v2**

---

## ARCHITECTURE OVERVIEW

Build a **multi-agent pipeline** using **LangGraph** as the state machine orchestrator. Each agent is a node. Routing (REPLY vs ESCALATE) is done via conditional edges.

```
START
  │
  ▼
[intake_node]           → parse ticket, extract signals
  │
  ▼
[classifier_node]       → domain + product_area classification
  │
  ▼
[risk_node]             → REPLY | ESCALATE | PARTIAL_REPLY decision
  │
  ├── ESCALATE ──────→ [escalation_node] ──→ [output_node] ──→ END
  │
  └── REPLY ─────────→ [retrieval_node]
                              │
                              ▼
                        [synthesis_node]
                              │
                              ▼
                        [faithfulness_node]   ← safety gate
                              │
                        ├── pass ──────────→ [output_node] ──→ END
                        └── fail ──────────→ [escalation_node] ──→ END
```

---

## TECH STACK

| Layer | Library | Version |
|---|---|---|
| LLM | `anthropic` Python SDK | latest |
| Model | `claude-sonnet-4-20250514` | — |
| Agent orchestration | `langgraph` | latest |
| Vector DB | `chromadb` | latest |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | latest |
| Sparse search | `rank_bm25` | latest |
| Re-ranker | `sentence-transformers` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) | latest |
| Web scraping | `trafilatura` | latest |
| Terminal UI | `rich` | latest |
| Data validation | `pydantic` v2 | latest |
| CSV | `pandas` | latest |
| Logging | `structlog` | latest |
| Config | `pydantic-settings` | latest |
| Tokenization | `tiktoken` | latest |
| HTTP | `httpx` | latest |
| Env | `python-dotenv` | latest |

---

## COMPLETE DIRECTORY STRUCTURE

Generate **every file** listed below:

```
code/
├── agents/
│   ├── __init__.py
│   ├── intake.py
│   ├── classifier.py
│   ├── risk.py
│   ├── retrieval.py
│   ├── synthesis.py
│   ├── escalation.py
│   └── output_formatter.py
├── graph/
│   ├── __init__.py
│   ├── pipeline.py          # LangGraph state machine
│   └── state.py             # GraphState TypedDict
├── rag/
│   ├── __init__.py
│   ├── ingest.py            # scrape → chunk → embed → store
│   ├── retriever.py         # hybrid BM25 + vector + cross-encoder rerank
│   └── corpus_store.py      # ChromaDB wrapper with 3 collections
├── models/
│   ├── __init__.py
│   └── schemas.py           # All Pydantic v2 models
├── config/
│   ├── __init__.py
│   └── settings.py          # pydantic-settings config
├── ui/
│   ├── __init__.py
│   └── terminal.py          # Rich terminal interface
├── utils/
│   ├── __init__.py
│   ├── logger.py            # structlog setup → log.txt
│   └── safety.py            # PII detection, risk keyword lists
├── scripts/
│   └── build_corpus.py      # One-time corpus ingestion runner
├── main.py                  # Interactive terminal entry point
├── batch.py                 # CSV batch processor → output.csv
├── requirements.txt
└── .env.example
```

---

## FILE-BY-FILE IMPLEMENTATION SPECIFICATION

### `requirements.txt`

```
anthropic>=0.40.0
langgraph>=0.2.0
langchain-core>=0.3.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
rank-bm25>=0.2.2
trafilatura>=1.12.0
rich>=13.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
pandas>=2.0.0
structlog>=24.0.0
tiktoken>=0.7.0
httpx>=0.27.0
python-dotenv>=1.0.0
torch>=2.0.0
```

---

### `.env.example`

```
ANTHROPIC_API_KEY=your_key_here
CHROMA_PERSIST_DIR=./data/chroma
CORPUS_DIR=./data/corpus
LOG_FILE=./log.txt
OUTPUT_CSV=./output.csv
MAX_RETRIEVAL_CHUNKS=5
RERANK_TOP_K=3
CHUNK_SIZE_TOKENS=400
CHUNK_OVERLAP_TOKENS=50
RISK_THRESHOLD=6
```

---

### `config/settings.py`

Use `pydantic-settings` `BaseSettings`. Load from `.env`. Fields:
- `anthropic_api_key: str`
- `chroma_persist_dir: str = "./data/chroma"`
- `corpus_dir: str = "./data/corpus"`
- `log_file: str = "./log.txt"`
- `output_csv: str = "./output.csv"`
- `max_retrieval_chunks: int = 5`
- `rerank_top_k: int = 3`
- `chunk_size_tokens: int = 400`
- `chunk_overlap_tokens: int = 50`
- `risk_threshold: int = 6`
- `model: str = "claude-sonnet-4-20250514"`

Expose a singleton `settings = Settings()`.

---

### `models/schemas.py`

Define all Pydantic v2 models with full field validation. Implement ALL of these:

```python
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class Domain(str, Enum):
    HACKERRANK = "hackerrank"
    CLAUDE = "claude"
    VISA = "visa"
    UNKNOWN = "unknown"

class ProductArea(str, Enum):
    # HackerRank
    ASSESSMENTS = "assessments"
    CODING_CHALLENGES = "coding_challenges"
    PROCTORING = "proctoring"
    HR_BILLING = "hr_billing"
    HR_ACCOUNT = "hr_account"
    HR_BUGS = "hr_bugs"
    HR_GENERAL = "hr_general"
    # Claude
    CLAUDE_API = "claude_api"
    CLAUDE_UI = "claude_ui"
    CLAUDE_BILLING = "claude_billing"
    CLAUDE_SAFETY = "claude_safety"
    CLAUDE_ACCOUNT = "claude_account"
    CLAUDE_GENERAL = "claude_general"
    # Visa
    VISA_FRAUD = "visa_fraud"
    VISA_DISPUTES = "visa_disputes"
    VISA_CARD_SERVICES = "visa_card_services"
    VISA_MERCHANT = "visa_merchant"
    VISA_ELIGIBILITY = "visa_eligibility"
    VISA_GENERAL = "visa_general"
    # Fallback
    UNKNOWN = "unknown"

class Action(str, Enum):
    REPLY = "reply"
    ESCALATE = "escalate"
    PARTIAL_REPLY = "partial_reply"

class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IntakeResult(BaseModel):
    raw_text: str
    ticket_summary: str = Field(description="1-sentence distillation of the issue")
    urgency_signals: list[str] = Field(default_factory=list)
    pii_detected: bool = False
    sentiment: str = "neutral"
    keywords: list[str] = Field(default_factory=list)

class DomainResult(BaseModel):
    domain: Domain
    product_area: ProductArea
    classification_confidence: float = Field(ge=0.0, le=1.0)
    classification_reasoning: str

class RiskResult(BaseModel):
    action: Action
    risk_score: int = Field(ge=0, le=10)
    urgency: UrgencyLevel
    escalation_reason: Optional[str] = None
    escalation_team: Optional[str] = None

class RetrievedChunk(BaseModel):
    chunk_text: str
    source_url: str
    source_title: str
    relevance_score: float
    domain: Domain

class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    query_used: str
    total_candidates: int

class SynthesisResult(BaseModel):
    response_text: str
    sources: list[str]
    is_grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)

class FaithfulnessResult(BaseModel):
    is_faithful: bool
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    reasoning: str

class TicketOutput(BaseModel):
    ticket_id: str
    original_text: str
    domain: Domain
    product_area: ProductArea
    action: Action
    risk_score: int
    response: str
    sources: list[str]
    escalation_reason: Optional[str] = None
    escalation_team: Optional[str] = None
    faithfulness_score: Optional[float] = None
    processing_time_ms: int = 0
```

---

### `graph/state.py`

```python
from typing import TypedDict, Optional, Any
from models.schemas import *

class GraphState(TypedDict):
    # Input
    ticket_id: str
    raw_ticket: str
    
    # Agent outputs — all Optional so they can be None before that node runs
    intake: Optional[IntakeResult]
    domain_result: Optional[DomainResult]
    risk_result: Optional[RiskResult]
    retrieval_result: Optional[RetrievalResult]
    synthesis_result: Optional[SynthesisResult]
    faithfulness_result: Optional[FaithfulnessResult]
    final_output: Optional[TicketOutput]
    
    # Metadata
    error: Optional[str]
    start_time_ms: int
```

---

### `utils/safety.py`

Implement the following fully:

**`ESCALATION_TRIGGERS` dict** — maps trigger phrase patterns to `(risk_score, escalation_team, reason)`:

```python
ESCALATION_TRIGGERS = {
    # Visa — always escalate
    r"(fraud|unauthorized.*(charge|transaction)|stolen card|card compromised|someone used my card)": 
        (10, "Visa Fraud & Security Team", "Potential fraud or unauthorized transaction detected"),
    r"(dispute|chargeback|transaction dispute)":
        (8, "Visa Disputes Resolution Team", "Transaction dispute requires human review"),
    
    # All domains — account security
    r"(account.*hacked|account.*compromised|someone.*logged in|unauthorized.*access|can.?t log.*in|locked.*out|account.*suspended|account.*banned)":
        (9, "Account Security Team", "Account security incident"),
    
    # HackerRank — proctoring
    r"(proctoring.*wrong|false.*positive|incorrectly.*flagged|cheating.*flag|assessment.*disqualif|unfairly.*penaliz)":
        (9, "HackerRank Integrity & Appeals Team", "Proctoring false positive — affects candidate career"),
    
    # Legal/compliance
    r"(legal|lawsuit|attorney|GDPR|data breach|subpoena|court order|regulatory)":
        (10, "Legal & Compliance Team", "Legal or regulatory matter"),
    
    # PII / data exposure
    r"(my data.*exposed|data.*leak|personal.*information.*shared)":
        (10, "Data Protection Officer", "Potential data exposure incident"),
    
    # Billing disputes (specific amounts)
    r"(charged.*\$[\d]+|incorrect.*charge|overcharged|double.?charged|refund.*\$[\d]+)":
        (7, "Billing Support Team", "Specific billing dispute requires human verification"),
}
```

**`detect_pii(text: str) -> bool`** — regex-based: detect credit card numbers (Luhn pattern), email addresses, phone numbers, SSN patterns. Return True if any found.

**`get_risk_score(text: str) -> tuple[int, str, str]`** — iterate ESCALATION_TRIGGERS, return highest risk match as `(score, team, reason)` or `(2, None, None)` if no match.

**`redact_pii(text: str) -> str`** — replace card numbers, SSNs with `[REDACTED]`.

---

### `utils/logger.py`

Set up `structlog` to:
- Write JSON structured logs to `log.txt` (from settings)  
- Also write human-readable output to stdout via `rich`
- Expose `get_logger(name: str)` function
- Each log entry includes: `timestamp`, `agent`, `ticket_id`, `event`, `data`

Implement a `ChatTranscriptLogger` class that:
- Accumulates a readable conversation-style log per ticket
- Has method `log_agent_step(agent_name, input_summary, output_summary, ticket_id)`
- Has method `save(path)` to write the full `log.txt`

---

### `rag/corpus_store.py`

Implement `CorpusStore` class:

```python
class CorpusStore:
    def __init__(self, persist_dir: str)
    def get_or_create_collection(self, domain: Domain) -> chromadb.Collection
    def add_chunks(self, domain: Domain, chunks: list[dict]) -> None
        # chunk dict: {id, text, embedding, source_url, source_title, domain}
    def query_dense(self, domain: Domain, query_embedding: list[float], n_results: int) -> list[dict]
    def collection_exists(self, domain: Domain) -> bool
    def get_all_chunks(self, domain: Domain) -> list[dict]
    def chunk_count(self, domain: Domain) -> int
```

Use ChromaDB's persistent client. One collection per domain named `support_{domain}`. Store metadata: `source_url`, `source_title`, `domain`, `chunk_index`.

---

### `rag/ingest.py`

Implement the full corpus ingestion pipeline. This is a **one-time offline script**.

**`scrape_url(url: str) -> list[dict]`**:
- Use `trafilatura` to fetch and extract clean text from a URL
- Also follow links on the page that are within the same domain (max depth 2, max 50 pages per domain)
- Use `httpx` for HTTP with `User-Agent: Mozilla/5.0` header
- Return list of `{url, title, text}` dicts
- Cache scraped pages to `CORPUS_DIR/domain_name/` as `.txt` files so re-runs don't re-scrape

**Domain seed URLs to scrape:**
```python
DOMAIN_SEEDS = {
    Domain.HACKERRANK: [
        "https://support.hackerrank.com/hc/en-us",
        "https://support.hackerrank.com/hc/en-us/categories",
    ],
    Domain.CLAUDE: [
        "https://support.claude.com/en/",
        "https://support.claude.com/en/collections",
    ],
    Domain.VISA: [
        "https://www.visa.co.in/support.html",
        "https://www.visa.co.in/support/consumer/",
        "https://www.visa.co.in/support/small-business/",
    ],
}
```

**`chunk_text(text: str, source_url: str, source_title: str, domain: Domain, chunk_size: int, overlap: int) -> list[dict]`**:
- Use `tiktoken` with `cl100k_base` encoding to count tokens
- Sliding window chunking: `chunk_size` tokens, `overlap` tokens overlap
- Return list of chunk dicts with `id` (uuid), `text`, `source_url`, `source_title`, `domain`

**`embed_chunks(chunks: list[dict]) -> list[dict]`**:
- Load `all-MiniLM-L6-v2` via `sentence_transformers.SentenceTransformer`
- Batch encode all chunk texts
- Add `embedding` field to each chunk dict
- Return updated list

**`ingest_domain(domain: Domain, store: CorpusStore) -> int`**:
- Orchestrate: scrape → chunk → embed → store
- Return number of chunks stored
- Skip if collection already has chunks (idempotent)

**`build_corpus() -> None`**:
- Run `ingest_domain` for all three domains
- Print progress with `rich.progress`

---

### `rag/retriever.py`

Implement `HybridRetriever` class:

```python
class HybridRetriever:
    def __init__(self, store: CorpusStore)
    
    def _build_bm25_index(self, domain: Domain) -> BM25Okapi
        # Build BM25 index from all chunks in domain collection
        # Cache in memory per domain
    
    def _dense_search(self, domain: Domain, query: str, top_k: int) -> list[RetrievedChunk]
        # Embed query with all-MiniLM-L6-v2
        # Query ChromaDB collection
        # Return top_k chunks with relevance scores
    
    def _sparse_search(self, domain: Domain, query: str, top_k: int) -> list[RetrievedChunk]
        # Tokenize query
        # Score with BM25
        # Return top_k chunks
    
    def _reciprocal_rank_fusion(
        self, 
        dense_results: list[RetrievedChunk], 
        sparse_results: list[RetrievedChunk],
        k: int = 60
    ) -> list[RetrievedChunk]
        # RRF formula: score = sum(1 / (k + rank_i))
        # Merge and deduplicate by chunk text
        # Return combined ranked list
    
    def _rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]
        # Use cross-encoder/ms-marco-MiniLM-L-6-v2
        # Score (query, chunk_text) pairs
        # Return top_k by cross-encoder score
    
    def retrieve(self, domain: Domain, query: str, top_k: int = 5) -> RetrievalResult
        # Full hybrid pipeline:
        # 1. Dense search (top 20)
        # 2. Sparse BM25 search (top 20)
        # 3. RRF fusion
        # 4. Cross-encoder rerank → top_k
        # Return RetrievalResult
```

---

### `agents/intake.py`

Implement `run_intake(state: GraphState) -> dict`:

This is a **deterministic** agent — do NOT call Claude API. Use regex + heuristics only (fast, free, no latency).

```python
def run_intake(state: GraphState) -> dict:
    text = state["raw_ticket"]
    
    # 1. Detect urgency signal keywords
    urgency_keywords = [
        "urgent", "immediately", "asap", "emergency", "critical",
        "fraud", "stolen", "hacked", "compromised", "locked out",
        "can't access", "lost access", "unauthorized", "dispute",
        "refund", "overcharged", "false positive", "disqualified",
        "legal", "lawsuit"
    ]
    found_signals = [kw for kw in urgency_keywords if kw.lower() in text.lower()]
    
    # 2. Detect PII
    pii = detect_pii(text)
    
    # 3. Simple sentiment (keyword-based)
    negative_words = ["angry", "frustrated", "terrible", "awful", "worst", "unacceptable", "disgusting"]
    positive_words = ["thank", "great", "love", "excellent", "helpful", "appreciate"]
    sentiment = "negative" if any(w in text.lower() for w in negative_words) else \
                "positive" if any(w in text.lower() for w in positive_words) else "neutral"
    
    # 4. Extract keywords (simple noun phrases, domain terms)
    # Use basic regex tokenization — no external NLP libs needed
    keywords = list(set(re.findall(r'\b[A-Za-z]{4,}\b', text)))[:20]
    
    # 5. 1-sentence summary — truncate/clean
    summary = text.strip().split('.')[0][:200] if text else "No content"
    
    return {
        "intake": IntakeResult(
            raw_text=text,
            ticket_summary=summary,
            urgency_signals=found_signals,
            pii_detected=pii,
            sentiment=sentiment,
            keywords=keywords
        )
    }
```

---

### `agents/classifier.py`

Implement `run_classifier(state: GraphState) -> dict`:

Call Claude API with a **zero-shot structured classification prompt**. Parse JSON response.

**System prompt:**
```
You are a support ticket classifier. Given a support ticket, you must classify it into exactly one domain and product area.

Domains and their product areas:
- hackerrank: assessments, coding_challenges, proctoring, hr_billing, hr_account, hr_bugs, hr_general
- claude: claude_api, claude_ui, claude_billing, claude_safety, claude_account, claude_general  
- visa: visa_fraud, visa_disputes, visa_card_services, visa_merchant, visa_eligibility, visa_general

Respond ONLY with valid JSON matching this schema:
{
  "domain": "<domain>",
  "product_area": "<product_area>",
  "classification_confidence": <0.0-1.0>,
  "classification_reasoning": "<one sentence>"
}

Rules:
- If the ticket mentions card, transaction, payment, bank, CVV, PIN → domain is "visa"
- If the ticket mentions assessment, test, coding challenge, proctoring, candidate, recruiter, HackerRank → domain is "hackerrank"  
- If the ticket mentions Claude, Anthropic, AI assistant, API key, model, claude.ai → domain is "claude"
- Product area must be from the exact list above
```

**User message:** `state["intake"].ticket_summary + "\n\nFull text: " + state["raw_ticket"][:500]`

Parse the JSON response. Handle malformed JSON gracefully — fall back to `Domain.UNKNOWN`. Return updated state dict with `domain_result` key.

Also implement a **keyword fallback** for classification that runs if API confidence < 0.5:
```python
DOMAIN_KEYWORDS = {
    Domain.HACKERRANK: ["hackerrank", "assessment", "proctoring", "coding test", "recruiter", "candidate", "challenge", "submission"],
    Domain.CLAUDE: ["claude", "anthropic", "claude.ai", "api key", "model", "prompt", "conversation"],
    Domain.VISA: ["visa", "card", "transaction", "payment", "cvv", "pin", "atm", "merchant", "dispute", "chargeback"],
}
```

---

### `agents/risk.py`

Implement `run_risk(state: GraphState) -> dict`:

**Deterministic** — no Claude API call. Use `utils/safety.py` logic.

```python
def run_risk(state: GraphState) -> dict:
    text = state["raw_ticket"]
    intake = state["intake"]
    domain = state["domain_result"].domain
    product_area = state["domain_result"].product_area
    
    # 1. Check hard escalation triggers from safety.py
    risk_score, team, reason = get_risk_score(text)
    
    # 2. Domain-specific overrides
    if product_area == ProductArea.VISA_FRAUD:
        risk_score = max(risk_score, 10)
        team = team or "Visa Fraud & Security Team"
        reason = reason or "Fraud-classified ticket"
    
    if product_area == ProductArea.PROCTORING:
        risk_score = max(risk_score, 8)
        team = team or "HackerRank Integrity & Appeals Team"
        reason = reason or "Proctoring dispute — affects candidate career"
    
    if product_area in [ProductArea.CLAUDE_ACCOUNT, ProductArea.HR_ACCOUNT]:
        risk_score = max(risk_score, 7)
    
    # 3. PII detected → flag
    if intake.pii_detected:
        risk_score = max(risk_score, 8)
        team = team or "Data Protection Team"
        reason = reason or "PII detected in ticket — handle with care"
    
    # 4. Decision
    threshold = settings.risk_threshold  # default 6
    
    if risk_score >= threshold:
        action = Action.ESCALATE
        urgency = UrgencyLevel.CRITICAL if risk_score >= 9 else UrgencyLevel.HIGH
    else:
        action = Action.REPLY
        urgency = UrgencyLevel.LOW if risk_score <= 2 else UrgencyLevel.MEDIUM
    
    return {
        "risk_result": RiskResult(
            action=action,
            risk_score=risk_score,
            urgency=urgency,
            escalation_reason=reason,
            escalation_team=team
        )
    }
```

---

### `agents/retrieval.py`

Implement `run_retrieval(state: GraphState) -> dict`:

```python
def run_retrieval(state: GraphState) -> dict:
    domain = state["domain_result"].domain
    intake = state["intake"]
    
    # Build a rich retrieval query from intake
    query = f"{intake.ticket_summary} {' '.join(intake.keywords[:10])}"
    
    # Run hybrid retrieval
    retriever = get_retriever()  # singleton
    result = retriever.retrieve(
        domain=domain,
        query=query,
        top_k=settings.rerank_top_k
    )
    
    # If no results found, escalate
    if not result.chunks:
        state["risk_result"] = RiskResult(
            action=Action.ESCALATE,
            risk_score=7,
            urgency=UrgencyLevel.HIGH,
            escalation_reason="No relevant documentation found in corpus for this query",
            escalation_team="Tier 2 Support"
        )
    
    return {"retrieval_result": result}
```

Use a module-level singleton `_retriever: Optional[HybridRetriever] = None` with a `get_retriever()` factory that initializes once.

---

### `agents/synthesis.py`

Implement `run_synthesis(state: GraphState) -> dict`:

Call Claude API with retrieved chunks as grounded context.

**System prompt** (strict grounding):
```
You are a support agent for {domain_name}. 

CRITICAL RULES:
1. Answer ONLY using information from the provided context sections below.
2. If the context does not contain enough information to fully answer the question, explicitly say: "I don't have enough information in our support documentation to fully answer this. I recommend contacting our support team directly."
3. NEVER invent policies, prices, time limits, procedures, or any facts not in the context.
4. Always be helpful, empathetic, and professional.
5. Keep your response concise and structured (use bullet points where helpful).
6. At the end, cite the sources you used as: "Sources: [URL1], [URL2]"

Context:
{context_block}
```

Build `context_block` as:
```
--- Source 1: {chunk.source_title} ({chunk.source_url}) ---
{chunk.chunk_text}

--- Source 2: ...
```

**User message:** `state["raw_ticket"]`

Parse response, extract sources from response text. Return `SynthesisResult`.

---

### `agents/escalation.py`

Implement `run_escalation(state: GraphState) -> dict`:

Generate a professional, empathetic escalation response. Do NOT call LLM — use template.

```python
ESCALATION_TEMPLATES = {
    "Visa Fraud & Security Team": """
Thank you for contacting Visa Support. We take security concerns very seriously.

Your case has been flagged as a high-priority security matter and is being escalated to our **Fraud & Security Team** for immediate review.

**Next steps:**
- A security specialist will contact you within 24 hours
- Please do not attempt further transactions until resolved  
- If you believe your card is compromised, call the number on the back of your card immediately to freeze it

We apologize for any inconvenience and appreciate your patience while we investigate.
""",
    # ... templates for all escalation teams
    "default": """
Thank you for reaching out to our support team.

Your request has been reviewed and requires attention from a specialist on our team. Your case has been escalated to **{team}** for further assistance.

**Reason for escalation:** {reason}

A team member will follow up with you as soon as possible. We appreciate your patience and apologize for any inconvenience.
"""
}
```

Fill in team and reason from `state["risk_result"]`. Return final `TicketOutput` with `action=ESCALATE`.

---

### `agents/output_formatter.py`

Implement `run_output_formatter(state: GraphState) -> dict`:

Assemble the final `TicketOutput` from all state fields. Calculate `processing_time_ms`. Return final state update.

Also implement:
```python
def format_csv_row(output: TicketOutput) -> dict:
    """Convert TicketOutput to a flat dict for CSV output"""
    return {
        "ticket_id": output.ticket_id,
        "domain": output.domain.value,
        "product_area": output.product_area.value,
        "action": output.action.value,
        "risk_score": output.risk_score,
        "response": output.response,
        "sources": " | ".join(output.sources),
        "escalation_reason": output.escalation_reason or "",
        "escalation_team": output.escalation_team or "",
        "faithfulness_score": output.faithfulness_score or "",
    }
```

---

### `agents/faithfulness_check` (inside `agents/synthesis.py` or separate file)

Implement `run_faithfulness_check(state: GraphState) -> dict`:

Call Claude API with a **binary faithfulness verification prompt**:

```
System: You are a faithfulness checker. Your job is to verify that a support response does not contain any claims that are not grounded in the provided context.

Context provided to the agent:
{context}

Agent's response:
{response}

Question: Does the agent's response contain ANY factual claims, policies, procedures, prices, or timelines that are NOT supported by the context above?

Respond ONLY with valid JSON:
{
  "is_faithful": true/false,
  "faithfulness_score": 0.0-1.0,
  "reasoning": "one sentence"
}
```

If `is_faithful == False` → set `risk_result.action = ESCALATE` in state to trigger escalation branch.

---

### `graph/pipeline.py`

Wire the full LangGraph state machine. This is the most critical file.

```python
from langgraph.graph import StateGraph, END
from graph.state import GraphState
from agents.intake import run_intake
from agents.classifier import run_classifier
from agents.risk import run_risk
from agents.retrieval import run_retrieval
from agents.synthesis import run_synthesis, run_faithfulness_check
from agents.escalation import run_escalation
from agents.output_formatter import run_output_formatter

def route_after_risk(state: GraphState) -> str:
    """Conditional edge: after risk assessment, decide path"""
    action = state["risk_result"].action
    if action == Action.ESCALATE:
        return "escalation"
    else:
        return "retrieval"

def route_after_faithfulness(state: GraphState) -> str:
    """Conditional edge: after faithfulness check, decide path"""
    if state["faithfulness_result"].is_faithful:
        return "output"
    else:
        return "escalation"

def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("intake", run_intake)
    graph.add_node("classifier", run_classifier)
    graph.add_node("risk", run_risk)
    graph.add_node("retrieval", run_retrieval)
    graph.add_node("synthesis", run_synthesis)
    graph.add_node("faithfulness", run_faithfulness_check)
    graph.add_node("escalation", run_escalation)
    graph.add_node("output", run_output_formatter)
    
    # Linear edges
    graph.set_entry_point("intake")
    graph.add_edge("intake", "classifier")
    graph.add_edge("classifier", "risk")
    graph.add_edge("retrieval", "synthesis")
    graph.add_edge("synthesis", "faithfulness")
    graph.add_edge("escalation", "output")
    graph.add_edge("output", END)
    
    # Conditional edges
    graph.add_conditional_edges(
        "risk",
        route_after_risk,
        {"escalation": "escalation", "retrieval": "retrieval"}
    )
    graph.add_conditional_edges(
        "faithfulness",
        route_after_faithfulness,
        {"output": "output", "escalation": "escalation"}
    )
    
    return graph.compile()

# Singleton compiled graph
_app = None
def get_pipeline():
    global _app
    if _app is None:
        _app = build_graph()
    return _app
```

---

### `ui/terminal.py`

Implement rich terminal interface using `rich`. Functions:

**`print_banner()`** — print a styled banner with ASCII art title "Support Triage Agent" and the three domain URLs.

**`print_ticket_result(output: TicketOutput)`** — print a rich `Panel` with:
- Title: `[bold] Ticket {id} | {domain} | {action} [/bold]`
- Color: green for REPLY, red for ESCALATE, yellow for PARTIAL_REPLY
- Sections: Domain & Product Area, Risk Score (with color bar), Response text, Sources

**`print_processing_step(step_name: str, status: str)`** — print a spinner/step indicator during processing.

**`print_corpus_stats(stats: dict)`** — table of domain → chunk count.

**`interactive_loop(pipeline)`** — main REPL loop:
```
while True:
    ticket = prompt user for input (rich prompt)
    if ticket in ["exit", "quit", "q"]:
        break
    result = run_pipeline(ticket, pipeline)
    print_ticket_result(result)
```

---

### `scripts/build_corpus.py`

Standalone script. When run directly (`python scripts/build_corpus.py`), it:
1. Initializes `CorpusStore`
2. Calls `build_corpus()` from `rag/ingest.py`
3. Prints a rich table showing domain → pages scraped → chunks stored
4. Also writes a `data/corpus_manifest.json` with scrape timestamps

If corpus already built (ChromaDB collections exist and have data), print stats and exit without re-scraping.

Add a `--force` flag to re-scrape even if data exists.

---

### `main.py`

Entry point for interactive terminal use:

```python
#!/usr/bin/env python3
"""
Multi-Domain Support Triage Agent
Interactive Terminal Mode
"""
import sys
import time
from rich.console import Console
from ui.terminal import print_banner, interactive_loop, print_corpus_stats
from graph.pipeline import get_pipeline
from rag.corpus_store import CorpusStore
from config.settings import settings
from utils.logger import ChatTranscriptLogger

console = Console()

def check_corpus_ready() -> bool:
    """Check if corpus has been built. If not, prompt user to run build_corpus.py"""
    store = CorpusStore(settings.chroma_persist_dir)
    from models.schemas import Domain
    for domain in [Domain.HACKERRANK, Domain.CLAUDE, Domain.VISA]:
        if not store.collection_exists(domain) or store.chunk_count(domain) == 0:
            return False
    return True

def main():
    print_banner()
    
    if not check_corpus_ready():
        console.print("[yellow]⚠️  Corpus not found. Run: python scripts/build_corpus.py[/yellow]")
        console.print("[dim]This scrapes the support pages and builds the local vector database.[/dim]")
        sys.exit(1)
    
    # Show corpus stats
    store = CorpusStore(settings.chroma_persist_dir)
    from models.schemas import Domain
    stats = {d.value: store.chunk_count(d) for d in [Domain.HACKERRANK, Domain.CLAUDE, Domain.VISA]}
    print_corpus_stats(stats)
    
    pipeline = get_pipeline()
    transcript = ChatTranscriptLogger()
    
    interactive_loop(pipeline, transcript)
    
    transcript.save(settings.log_file)
    console.print(f"\n[green]✅ Session log saved to {settings.log_file}[/green]")

if __name__ == "__main__":
    main()
```

---

### `batch.py`

CSV batch processor. Reads `support_issues/support_issues.csv`, runs every ticket through the pipeline, writes `output.csv`.

```python
#!/usr/bin/env python3
"""
Batch processor: reads support_issues.csv → writes output.csv + log.txt
"""
import pandas as pd
import time
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from graph.pipeline import get_pipeline
from graph.state import GraphState
from agents.output_formatter import format_csv_row
from models.schemas import TicketOutput
from config.settings import settings
from utils.logger import ChatTranscriptLogger

def run_pipeline_on_ticket(pipeline, ticket_id: str, ticket_text: str) -> TicketOutput:
    """Run the full pipeline on a single ticket"""
    import time
    state: GraphState = {
        "ticket_id": ticket_id,
        "raw_ticket": ticket_text,
        "intake": None,
        "domain_result": None,
        "risk_result": None,
        "retrieval_result": None,
        "synthesis_result": None,
        "faithfulness_result": None,
        "final_output": None,
        "error": None,
        "start_time_ms": int(time.time() * 1000),
    }
    result = pipeline.invoke(state)
    return result["final_output"]

def main():
    # Load input CSV
    # Expected columns: ticket_id (or index), text/issue/description
    # Handle multiple possible column names gracefully
    
    df = pd.read_csv("support_issues/support_issues.csv")
    
    # Auto-detect text column
    text_col = None
    for col in ["text", "issue", "description", "message", "content", "body", "ticket"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[-1]  # fallback: last column
    
    # Auto-detect id column
    id_col = None
    for col in ["ticket_id", "id", "ID", "ticket_number"]:
        if col in df.columns:
            id_col = col
            break
    
    pipeline = get_pipeline()
    transcript = ChatTranscriptLogger()
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing tickets...", total=len(df))
        
        for idx, row in df.iterrows():
            ticket_id = str(row[id_col]) if id_col else str(idx)
            ticket_text = str(row[text_col])
            
            try:
                output = run_pipeline_on_ticket(pipeline, ticket_id, ticket_text)
                results.append(format_csv_row(output))
                transcript.log_agent_step(
                    "batch_processor", 
                    f"Ticket {ticket_id}", 
                    f"{output.action.value} | {output.domain.value}",
                    ticket_id
                )
            except Exception as e:
                results.append({
                    "ticket_id": ticket_id,
                    "domain": "unknown",
                    "product_area": "unknown",
                    "action": "escalate",
                    "risk_score": 10,
                    "response": f"System error — escalating for manual review. Error: {str(e)[:100]}",
                    "sources": "",
                    "escalation_reason": "Pipeline error",
                    "escalation_team": "Tier 2 Support",
                    "faithfulness_score": "",
                })
            
            progress.advance(task)
    
    # Write output CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv(settings.output_csv, index=False)
    print(f"✅ Output written to {settings.output_csv} ({len(results)} tickets)")
    
    # Save transcript
    transcript.save(settings.log_file)
    print(f"✅ Log saved to {settings.log_file}")

if __name__ == "__main__":
    main()
```

---

## ESCALATION ROUTING TABLE

Implement this routing logic exactly in `agents/escalation.py`:

| Product Area | Escalation Team | Response Template |
|---|---|---|
| `visa_fraud` | Visa Fraud & Security Team | Fraud response with card freeze instructions |
| `visa_disputes` | Visa Disputes Resolution Team | Dispute submission guidance |
| `proctoring` | HackerRank Integrity & Appeals Team | Appeals process explanation |
| `hr_account` / `claude_account` | Account Security Team | Account recovery steps |
| Legal/GDPR | Legal & Compliance Team | Data protection officer contact |
| PII detected | Data Protection Team | Privacy team routing |
| No corpus match | Tier 2 Support | General escalation |
| Billing dispute | Billing Support Team | Billing team routing |

---

## FAITHFULNESS ENFORCEMENT

The synthesis agent must follow these rules enforced in code:

1. **Context injection**: Always pass retrieved chunks as explicit `<context>` blocks in the system prompt — never rely on Claude's training knowledge
2. **Source citation**: Response must end with "Sources: [url1, url2]" — parsed and validated
3. **Faithfulness gate**: After generation, run `run_faithfulness_check` — if `is_faithful == False`, automatically escalate instead of returning the unfaithful response
4. **Confidence threshold**: If `faithfulness_score < 0.7`, flag in output even if technically faithful

---

## LOG.TXT FORMAT

The `log.txt` must be human-readable and show the full reasoning chain per ticket. Format:

```
================================================================================
TICKET: {ticket_id} | {timestamp}
================================================================================
INPUT: {raw_ticket}

[INTAKE AGENT]
  Summary: {ticket_summary}
  Urgency Signals: {signals}
  PII Detected: {bool}
  Sentiment: {sentiment}

[CLASSIFIER AGENT]  
  Domain: {domain} (confidence: {confidence})
  Product Area: {product_area}
  Reasoning: {reasoning}

[RISK AGENT]
  Risk Score: {score}/10
  Action: {action}
  Escalation Reason: {reason or "N/A"}

[RETRIEVAL AGENT]  (if REPLY path)
  Query: {query}
  Retrieved {n} chunks from {domain} corpus
  Top source: {source_url} (score: {score})

[SYNTHESIS AGENT]  (if REPLY path)
  Response generated. Faithfulness check: {pass/fail}

[FINAL OUTPUT]
  Action: {action}
  Response: {response}
  Sources: {sources}

================================================================================
```

---

## ERROR HANDLING

Every agent node must be wrapped in try/except. On exception:
- Log the error to structlog
- Set `state["error"] = str(e)`
- Return an escalation action with reason "System error — routing to human agent"
- Never crash the pipeline; always produce an output

---

## RUNNING INSTRUCTIONS

Generate a `README.md` in `code/` with:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Build corpus (one-time, ~5-10 minutes)
python scripts/build_corpus.py

# 4a. Interactive mode
python main.py

# 4b. Batch mode (generates output.csv + log.txt)
python batch.py
```

---

## QUALITY REQUIREMENTS

- **No stub functions** — every function must be fully implemented
- **No `pass` bodies** — all methods must have real code
- **No TODO comments** — everything is done
- **Typed** — every function has full type annotations
- **Pydantic v2 strict mode** — all models use `model_config = ConfigDict(strict=True)`  
- **Graceful degradation** — if a scrape fails, skip that URL and continue
- **Idempotent ingestion** — running `build_corpus.py` twice does not duplicate chunks
- **BM25 index built lazily** — only when first query hits that domain

---

## WHAT NOT TO DO

- ❌ Do not use Flask, FastAPI, Streamlit, or any web framework
- ❌ Do not use OpenAI, Cohere, or any non-Anthropic LLM for generation
- ❌ Do not hardcode support answers — always retrieve from corpus
- ❌ Do not log raw PII to log.txt — redact it
- ❌ Do not hallucinate corpus content if retrieval returns empty
- ❌ Do not make the faithfulness check optional — it is always mandatory
- ❌ Do not use `asyncio` for the main pipeline (LangGraph handles it)
- ❌ Do not skip error handling

---

## OUTPUT INSTRUCTIONS

Generate all files in order:
1. `requirements.txt`
2. `.env.example`
3. `config/settings.py`
4. `models/schemas.py`
5. `graph/state.py`
6. `utils/safety.py`
7. `utils/logger.py`
8. `rag/corpus_store.py`
9. `rag/ingest.py`
10. `rag/retriever.py`
11. `agents/intake.py`
12. `agents/classifier.py`
13. `agents/risk.py`
14. `agents/retrieval.py`
15. `agents/synthesis.py` (includes faithfulness check)
16. `agents/escalation.py`
17. `agents/output_formatter.py`
18. `graph/pipeline.py`
19. `ui/terminal.py`
20. `scripts/build_corpus.py`
21. `main.py`
22. `batch.py`
23. `README.md`

Write each file completely. Do not truncate. Do not say "rest of implementation follows the same pattern."

Begin now with `requirements.txt`.