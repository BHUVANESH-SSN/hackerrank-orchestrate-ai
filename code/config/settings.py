"""
Application settings loaded from environment variables via pydantic-settings.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration — reads from .env file."""

    groq_api_key: str = Field(default="", description="Groq API key")
    aws_access_key_id: str = Field(default="", description="AWS Access Key ID for Bedrock")
    aws_secret_access_key: str = Field(default="", description="AWS Secret Access Key for Bedrock")
    aws_region: str = Field(default="us-east-1", description="AWS region for Bedrock")
    chroma_persist_dir: str = Field(default="./data/chroma", description="ChromaDB persistence directory")
    corpus_dir: str = Field(default="../data", description="Path to scraped corpus data")
    log_file: str = Field(default="./log.txt", description="Path to chat transcript log")
    output_csv: str = Field(default="../support_tickets/output.csv", description="Path to output CSV")
    max_retrieval_chunks: int = Field(default=5, description="Max chunks to retrieve")
    rerank_top_k: int = Field(default=3, description="Top-K after cross-encoder reranking")
    chunk_size_tokens: int = Field(default=400, description="Chunk size in tokens")
    chunk_overlap_tokens: int = Field(default=50, description="Chunk overlap in tokens")
    risk_threshold: int = Field(default=6, description="Risk score threshold for escalation")
    model: str = Field(default="llama-3.1-8b-instant", description="Groq model name")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
