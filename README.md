<br/>
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/65/HackerRank_logo.png" alt="HackerRank" width="200"/>
</p>

<h1 align="center">Orchestrate '26: Enterprise Multi-Agent AI Triage System</h1>

<p align="center">
  <strong>Production-Ready LangGraph Triage Architecture for Tier-1 Support Automation</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/LangGraph-0.0.35-lightgray?logo=chainlink" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-orange" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/Llama_3.1-8B_Instant-black" alt="Meta Llama"/>
  <img src="https://img.shields.io/badge/Groq-LPU_Inference-green?logo=lightning" alt="Groq"/>
  <img src="https://img.shields.io/badge/FastAPI-Deployment-teal?logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Hackathon-HackerRank-brightgreen" alt="Hackathon"/>
</p>

---

<p align="center">
  <a href="https://github.com/BHUVANESH-SSN/hackerrank-orchestrate-ai.git"><strong>GitHub Repository:</strong> BHUVANESH-SSN/hackerrank-orchestrate-ai</a>
</p>

## Overview

This repository contains an Enterprise-grade Multi-Agent System designed for exactly one purpose: accurately triaging, resolving, and routing Tier-1 Customer Support tickets with zero hallucinations. Built for the **HackerRank Orchestrate May '26 Hackathon**, this pipeline safely processes queries across **HackerRank, Claude, and Visa** domains.

Unlike standard LLM chains, this architecture uses a **LangGraph state machine** to force deterministic safety checks. The system routes incoming messages, checks databases structurally, and employs a rigorous multi-step defense system to catch AI hallucinations before they ever reach the user.

### 💡 The "Middleware Plugin" Philosophy
This project is engineered as an **Invisible Autonomous Middleware**. It does not feature a customer-facing bot UI. Instead, it seamlessly intercepts Webhooks from a CRM (like Zendesk or Salesforce), evaluates emails against the company's knowledge base, and either:
1. **Replies Autonomously:** If the ticket is simple and safe.
2. **Escalates to Humans:** If the issue is complex, dangerous, or requires explicit employee review.

Because the system acts purely as an API gateway plugin between the CRM and the human support team, the logic is **100% Company-Agnostic**. By simply swapping out the vector database documents, this exact identical codebase can confidently triage tickets for HackerRank, Airbnb, Google, or any other enterprise.

---

## Enterprise Features

### Core Capabilities
- **LangGraph Orchestration:** A cyclic state machine mapping multi-agent routing (Intake -> Classification -> Risk -> Retrieval -> Synthesis -> Output).
- **Strict Faithfulness Gate:** The Synthesizer LLM self-critiques its own drafted response against the retrieved ChromaDB chunks. If a hallucination is detected, the `Faithfulness Gate` triggers an immediate human escalation.
- **Groq-Powered LPU Inference:** Uses the ultra-fast, highly contextual `llama-3.1-8b-instant` running on Groq hardware, achieving parsing speeds of `> 10,000 tokens/sec`.
- **Deterministic Output Conformity:** The system outputs strict `status, product_area, response, justification, request_type` schemas parsing 100% cleanly into pandas DataFrames.

### "Beyond the Basics" (Advanced Add-ons)
- **Conversational Memory Checkpointing:** Powered by LangGraph's native `MemorySaver`, the pipeline saves inter-turn history tracking using thread IDs.
- **Native Multilingual Zero-Shot:** The synthesis agent automatically detects inbound ticket languages (e.g., Spanish, Japanese) and natively translates the RAG output while preserving technical jargon.
- **Sentiment & Churn Risk Detection:** Uses lightweight mathematical sentiment mapping. If `churn_risk=true` or an angry sentiment is identified, the system bypasses RAG and routes directly to the **"Customer Retention" tier**.

---

## Project Architecture

The architecture is composed of 7 discrete pipeline states:

1. **Intake Agent:** Parses PII and creates the standardized schema.
2. **Classifier Agent:** Maps the core intent into strict domain bounds (`hr_billing`, `visa_merchant`, `claude_api`).
3. **Risk Agent:** Executes non-LLM heuristic safety checks (`utils/safety.py`) and specific churn detection.
4. **Retrieval Agent:** Queries the `ChromaDB` localized vector store utilizing chunked Hackathon documentation with `all-MiniLM-L6-v2` dense embeddings.
5. **Synthesis Agent:** Drafts the professional email response directly citing retrieved sources.
6. **Faithfulness Evaluator:** Cross-examines the synthesis to guarantee absolute truth. Flags hallucinatory statements.
7. **Output Formatter:** Condenses the state loop into the final 5-column CSV target trace.

---

## Enterprise Blueprint Implementations

To prove scalability beyond a terminal pipeline, we have included three production blueprints inside `code/enterprise/`:
* `fastapi_server.py`: Demonstrates deploying the LangGraph system inside an API utilizing `BackgroundTasks` to handle concurrent Zendesk webhooks asynchronously.
* `semantic_cache.py`: Intercepts recurring FAQ user intents using `Cosine Similarity` vector math to instantly return answered prompts, preventing massive LLM token waste on repeat queries.
* `scripts/human_approval.py`: A LangGraph `interrupt_before=["output"]` checkpoint blueprint demonstrating how Human-In-The-Loop approval gateways function for IT managers.
* `scripts/observability_dashboard.py`: Parses the pipeline output to render a Datadog-esque terminal visualization of API latency, token tracking, scaling anomalies, and escalation health ratios.

---

## Getting Started

### 1. Installation

```bash
cd code
# Setup virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables
Add your Groq API key to `code/.env`:
```
GROQ_API_KEY=gsk_your_key_here
MODEL=llama-3.1-8b-instant
```

### 3. Run the Core Hackathon Pipeline

**Batch Processing (Primary Hackathon Requirement):**
Parses `sample_support_tickets.csv` and generates the compliant `output.csv`.
```bash
python batch.py
```

**Rich Interactive Terminal (For Judge Testing):**
Launch a gorgeous command line UI to type dynamic queries directly into the graph.
```bash
python main.py
```

### 4. Run the Enterprise Dashboards
Test the Observability UI matrix:
```bash
python scripts/observability_dashboard.py
```

---

<p align="center">
  <i>Developed specifically for the Orchestrate '26 Competition. Code engineered for stability, scale, and compliance.</i>
</p>