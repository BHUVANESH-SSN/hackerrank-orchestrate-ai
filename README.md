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
  <img src="https://img.shields.io/badge/GPT--OSS--120B-Groq-black" alt="GPT-OSS-120B"/>
  <img src="https://img.shields.io/badge/Groq-LPU_Inference-green?logo=lightning" alt="Groq"/>
  <img src="https://img.shields.io/badge/FastAPI-Deployment-teal?logo=fastapi" alt="FastAPI"/>
</p>

---

**GitHub Repository:** [https://github.com/BHUVANESH-SSN/hackerrank-orchestrate-ai.git](https://github.com/BHUVANESH-SSN/hackerrank-orchestrate-ai.git)

---

## Overview

This repository contains an Enterprise-grade Multi-Agent System designed for accurately triaging, resolving, and routing Tier-1 Customer Support tickets with zero hallucinations. Built for the HackerRank Orchestrate May '26 Hackathon, this pipeline safely processes queries across the HackerRank, Claude, and Visa domains.

Unlike standard LLM chains, this architecture uses a LangGraph state machine to force deterministic safety checks. The system routes incoming messages, retrieves documents via a highly-tuned Hybrid RAG pipeline, and employs a rigorous dual-model defense system to catch AI hallucinations before they ever reach the user.

---

## System Design and Architecture

![Enterprise Architecture Flowchart](./hacker_rank_agentic_ai.png)

This system is designed as an Autonomous Support Middleware capable of integrating with existing support channels such as Zendesk, Salesforce, or direct email. The core design philosophy centers on Safety-First Agentic RAG. A state-machine approach ensures that every ticket passes through a rigorous sequence of validation gates before a response is generated.

---

## The Agentic Ecosystem (LangGraph State Machine)

The system is orchestrated using LangGraph, treating the triage process as a directed graph where each node represents a specialized agent. This enables conditional branching and deterministic error handling.

### 1. Intake and Normalization Agent
- **Role:** Data Cleaning.
- **Function:** Sanitizes the raw ticket body, removes metadata noise, and extracts key entities. 

### 2. Multi-Domain Classifier Agent
- **Role:** Intelligent Routing.
- **Function:** Utilizes Llama-3.1-8B for low-latency intent detection. It maps tickets to one of three domains: HackerRank, Claude, or Visa.

### 3. Risk and Safety Agent
- **Role:** Policy Enforcement.
- **Function:** A deterministic engine scanning for high-severity triggers such as PII leakage, security vulnerabilities, or legal threats. High-risk tickets are immediately diverted to human specialists.

### 4. Hybrid Retrieval Agent
- **Role:** Context Provision.
- **Function:** Implements a multi-stage Hybrid Search pipeline combining dense vector search (ChromaDB) and sparse keyword search (BM25), followed by Cross-Encoder reranking for maximum precision.

### 5. Synthesis Agent (The Maker)
- **Role:** Professional Response Generation.
- **Function:** Uses GPT-OSS-120B to draft grounded responses using only the provided context.

### 6. Faithfulness Agent (The Checker)
- **Role:** Validation.
- **Function:** Cross-examines the drafted response against source documents to ensure 100% factual accuracy and zero hallucinations.

---

## Technical Stack

- **Orchestration:** LangGraph
- **LLM Inference:** Groq LPU (Llama 3.1 & GPT-OSS-120B)
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace all-MiniLM-L6-v2
- **Reranker:** HuggingFace ms-marco-MiniLM-L-6-v2
- **API Framework:** FastAPI

---

## Evaluator Guide: System Execution

### 1. Prerequisites and Installation

Ensure Python 3.12+ is installed.

```bash
# Navigate to the code directory
cd code

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the `code/` directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Knowledge Base Initialization

```bash
python scripts/build_corpus.py
```

### 4. Running the Batch Evaluation

```bash
python batch.py
```

### 5. Interactive UI (Optional)

```bash
python main.py
```

---

<p align="center">
  <i>Developed for the HackerRank Orchestrate '26 Competition. Focused on Safety, Grounding, and Professional Reliability.</i>
</p>
