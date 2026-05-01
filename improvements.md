# Production-Grade Improvement Plan (AWS Stack)

With a $150 AWS budget, we can transition this hackathon prototype into a highly scalable, enterprise-grade production support triage system. This plan maps out the improvements module by module, fully utilizing AWS services to optimize for efficiency, scale, and performance without overspending.

---

## 1. Intelligence & Orchestration (Agents & Graph)
Currently, the LangGraph system relies on local state and Groq (which hit its rate limit). 

**Production Improvements:**
- **Amazon Bedrock for LLMs**: Since we have AWS credits, replace Groq with Amazon Bedrock. 
  - Use **Claude 3.5 Sonnet** via Bedrock for complex reasoning (synthesis and faithfulness checking).
  - Use **Claude 3 Haiku** for faster, cheaper classification & routing.
- **AWS Step Functions / LangGraph Cloud**: Instead of running LangGraph in a single python process, deploy the state machine. We can host the LangGraph API on **AWS Fargate** (Serverless Containers).
- **Asynchronous Processing**: Introduce **Amazon SQS (Simple Queue Service)** to queue incoming support tickets, parsing them through the graph sequentially or concurrently without missing a payload if traffic spikes.

## 2. Retrieval-Augmented Generation (RAG) & Vector Database
We currently use a local `ChromaDB` directory, BM25, and local HuggingFace embedding models (`all-MiniLM-L6-v2`), which don't scale across multiple instances.

**Production Improvements:**
- **Amazon OpenSearch Serverless (Vector Engine)**: Migrate ChromaDB to OpenSearch. This gives us horizontally scalable KNN vector search combined with world-class keyword (BM25) search natively out of the box.
- **Amazon Bedrock Titan Embeddings v2**: Replace the local `.pt` pytorch models with `amazon.titan-embed-text-v2:0` for enterprise-grade text embeddings, reducing container size drastically (no need to install PyTorch/CUDA libraries).
- **Amazon S3 Document Store**: Instead of keeping raw corpus markdown files in `./data`, sync them to an S3 bucket.
- **Automated Ingestion**: Set up an **AWS Lambda function** triggered by S3 uploads to automatically chunk, embed, and index new support documentation into OpenSearch.

## 3. Deployment & Infrastructure
The application currently runs via a terminal command `python main.py`.

**Production Improvements:**
- **Amazon API Gateway + AWS Lambda**: Expose the triage agent as a REST API (e.g., `/v1/triage`).
- **AWS DynamoDB for State**: LangGraph can persist state (checkpointers). Use DynamoDB to store the chat/ticket history so multiple turns of conversation can be supported for the same ticket.
- **AWS ECS (Fargate) for Batch Jobs**: For bulk processing (like the batch processor), spin up a Fargate container, process the CSV from S3, save the output CSV to S3, and shut down automatically, costing pennies.

## 4. Safety & Observability
Currently, logging is written to a local `log.txt` via structlog, and safety is regex-based.

**Production Improvements:**
- **Amazon CloudWatch**: Route all `structlog` output to CloudWatch Logs. Setup CloudWatch Dashboards to monitor escalation rates, failure rates, and PII detection events.
- **Amazon Macie**: For deep PII/PHI redaction if tickets contain extremely sensitive user data.
- **Bedrock Guardrails**: Replace manual python regex logic with native Amazon Bedrock Guardrails, which can automatically redact PII from prompts, block jailbreaks, and filter toxic inputs before it even hits the LLM.

---

## 💡 Recommended $150 AWS Architecture Roadmap

### Phase 1: Lift & Shift API (Low Cost, High Reward) - Estimated $10-$20/month
1. Swap `Groq` for `boto3` and call Claude 3 Haiku/Sonnet via **Amazon Bedrock**.
2. Run your existing LangGraph code inside **AWS Lambda**.
3. Expose via **API Gateway**.
4. Result: No more 429 Rate Limits, Highly Available.

### Phase 2: RAG Modernization - Estimated $30-$50/month
1. Move the local `/data` folder to **Amazon S3**.
2. Replace local ChromaDB clustering with **Amazon OpenSearch Serverless** (you pay only for what you use).
3. Use Bedrock Titan Embeddings instead of downloading PyTorch locally.
4. Result: Zero-maintenance vector search with native Hybrid (Dense + Sparse) querying.

### Phase 3: Analytics & Safety - Estimated $5-$10/month
1. Push all metrics (Escalation %, Average Response Time, Grounding Score) to **CloudWatch/X-Ray**.
2. Configure **Bedrock Guardrails** to drop hallucinated statements strictly.

**Total spend will be extremely low** in a hackathon / proof-of-concept environment, and perfectly suited to burn down the $150 efficiently while demonstrating tier-1 enterprise software engineering principles.
