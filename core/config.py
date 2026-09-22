"""项目配置管理：从 .env 加载，全局单例。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class Settings:
    # LLM
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_api_base: str = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_DEFAULT_MODEL", "qwen-flash"))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1")))
    item_model: str = field(default_factory=lambda: os.getenv("ITEM_MODEL", "qwen-flash"))
    kg_model: str = field(default_factory=lambda: os.getenv("KG_MODEL", "qwen-flash"))

    # Embedding
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-v4"))

    # Reranker：DashScope gte-rerank-v2（cross-encoder，无需GPU，等价 BGE-Reranker）
    rerank_model: str = field(default_factory=lambda: os.getenv("RERANK_MODEL", "gte-rerank-v2"))

    # Milvus
    milvus_url: str = field(default_factory=lambda: os.getenv("MILVUS_URL", "http://localhost:19530"))
    chunks_collection: str = field(default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "kb_chunks"))
    entity_collection: str = field(default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "kb_entity_names"))
    item_collection: str = field(default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "kb_item_names"))
    milvus_metric: str = field(default_factory=lambda: os.getenv("MILVUS_METRIC_TYPE", "COSINE"))
    milvus_min_score: float = field(default_factory=lambda: float(os.getenv("MILVUS_MIN_COSINE_SCORE", "0.75")))

    # Neo4j
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_username: str = field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))
    neo4j_database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))

    # MongoDB
    mongo_url: str = field(default_factory=lambda: os.getenv("MONGO_URL", "mongodb://localhost:27017"))
    mongo_db: str = field(default_factory=lambda: os.getenv("MONGO_DB_NAME", "kb001"))

    # MinIO
    minio_endpoint: str = field(default_factory=lambda: os.getenv("MINIO_ENDPOINT", "localhost:9000"))
    minio_access_key: str = field(default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
    minio_secret_key: str = field(default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "minioadmin"))
    minio_bucket: str = field(default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "knowledge-base"))

    # MCP
    mcp_url: str = field(default_factory=lambda: os.getenv("MCP_DASHSCOPE_BASE_URL", ""))

    # 文档切分
    max_content_length: int = 2000
    min_content_length: int = 500

    # 检索
    embedding_search_limit: int = 10
    rrf_k: int = 60
    rrf_max_results: int = 10


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
