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
  <img src="https://img.shields.io/badge/AWS_Bedrock-Claude_3.5-purple" alt="AWS Bedrock"/>
  <img src="https://img.shields.io/badge/Groq-Qwen3_&_Llama-green?logo=lightning" alt="Groq"/>
  <img src="https://img.shields.io/badge/Hackathon-HackerRank-brightgreen" alt="Hackathon"/>
</p>

---

<p align="center">
  <a href="https://github.com/BHUVANESH-SSN/hackerrank-orchestrate-ai.git"><strong>GitHub Repository:</strong> BHUVANESH-SSN/hackerrank-orchestrate-ai</a>
</p>

## Overview

This repository contains an Enterprise-grade Multi-Agent System designed for exactly one purpose: accurately triaging, resolving, and routing Tier-1 Customer Support tickets with zero hallucinations. Built for the **HackerRank Orchestrate May '26 Hackathon**, this pipeline safely processes queries across the HackerRank, Claude, and Visa domains.

Unlike standard LLM chains, this architecture uses a **LangGraph state machine** to force deterministic safety checks. The system routes incoming messages, retrieves documents via a highly-tuned Hybrid RAG pipeline, and employs a rigorous dual-model defense system to catch AI hallucinations before they ever reach the user.

---

## System Design & Core Concepts

![Enterprise Architecture Flowchart](./hacker_rank_agentic_ai.png)

To achieve an enterprise-ready standard, the system relies on several core architectural pillars:

### 1. Multi-Agent State Machine (LangGraph)
The pipeline is orchestrated using LangGraph, breaking down the triage process into distinct, specialized agents that pass a shared state. 
- **Intake & Classifier Agent:** Quickly parses incoming tickets, sanitizes PII, and maps the core intent into strict domain bounds (e.g., `hr_billing`, `visa_merchant`). It uses Groq (Llama 3.1) for ultra-fast, low-latency classification with a keyword-based fallback mechanism to ensure 100% uptime even during API rate limits.
- **Risk & Safety Agent:** A deterministic, non-LLM safety layer (`utils/safety.py`). It scans tickets for high-severity triggers (e.g., fraud, legal threats, security vulnerabilities, or PII leaks). High-risk tickets are instantly escalated to human teams (bypassing the AI response generation entirely), guaranteeing that sensitive issues are handled by real people.

### 2. Precision Hybrid RAG Pipeline
To prevent "No relevant documents found" errors on borderline tickets, the retrieval pipeline is highly tuned:
- **Dual-Index Search:** Combines BM25 (sparse keyword search) and ChromaDB (dense vector embeddings via `all-MiniLM-L6-v2`) to recall the top 20 candidate documents.
- **Cross-Encoder Reranking:** The candidates are passed through an MS-MARCO Cross-Encoder. The system dynamically re-ranks the documents and selects the absolute best `top_k=6` chunks to provide maximum, highly-relevant context to the synthesis model.

### 3. The Dual-Model Synthesis & Faithfulness Gate
Response generation is handled by **AWS Bedrock** (with an automatic fallback to **Groq**), utilizing a "Maker-Checker" architecture to eliminate hallucinations:
- **The Maker (Claude 3.5 Sonnet / Qwen3-32b):** Synthesizes a professional, empathetic email response using *only* the provided `top_k` documents. 
- **The Checker (Claude 3.5 Haiku / Qwen3-32b):** Acts as a strict "Faithfulness Evaluator". It cross-examines the drafted response against the retrieved source documents. If it detects any unverified claims or hallucinations, the ticket is instantly flagged as a `Faithfulness check failed` and escalated to a human.

---

## Evaluator Guide: How to Run the System

This repository is built to be easily reproducible by the HackerRank evaluation team. Follow these steps to initialize the environment, build the vector database, and run the pipeline.

### 1. Prerequisites & Installation

Ensure you have Python 3.12+ installed.

```bash
# Clone the repository and navigate to the code directory
cd code

# Create and activate a virtual environment
python -m venv venv

# Mac/Linux
source venv/bin/activate
# Windows
# venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

The system requires API keys for both **Groq** (for fast classification) and **AWS Bedrock** (for high-reasoning synthesis). 

Create a `.env` file inside the `code/` directory:

```env
GROQ_API_KEY=your_groq_api_key_here

```
*(Note: If AWS credentials are not provided, the system will gracefully fall back to using Groq for all generation tasks).*

### 3. Build the Knowledge Base (Corpus)

Before running the agents, you must ingest the markdown documentation into the localized ChromaDB and BM25 indexes.

```bash
python scripts/build_corpus.py
```
*This will parse the `data/` folder and generate the vector embeddings in `data/chroma/`.*

### 4. Run the Evaluation Batch

To process the `support_tickets.csv` file and generate the required output for the hackathon evaluation:

```bash
python batch.py
```
*The script will process each ticket sequentially (with built-in API rate-limit delays) and output the final results to `support_tickets/output.csv`.*

### 5. Interactive Testing (Optional)

If you would like to test the agent interactively via a terminal UI, run:

```bash
python main.py
```
*You can type custom support queries and watch the multi-agent graph classify, retrieve, and synthesize responses in real-time.*

---

<p align="center">
  <i>Developed specifically for the Orchestrate '26 Competition. Code engineered for stability, safety, and strict evaluation compliance.</i>
</p>