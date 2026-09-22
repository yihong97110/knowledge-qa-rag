"""嵌入工具

基于 DashScope OpenAI 兼容的 text-embedding-v4 API，提供：
- get_client()    返回 OpenAI 客户端单例
- embed()         批量生成嵌入向量
- embed_one()     单条文本嵌入
"""
from __future__ import annotations

from typing import Optional

from openai import OpenAI

from core.config import get_settings


# 全局单例客户端，避免重复创建连接
_embedding_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """返回 OpenAI 客户端单例，用于调用嵌入接口。"""
    global _embedding_client
    if _embedding_client is None:
        settings = get_settings()
        _embedding_client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
        )
    return _embedding_client


def embed(texts: list[str]) -> list[list[float]]:
    """批量生成嵌入向量。

    Args:
        texts: 文本列表

    Returns:
        与输入顺序一致的向量列表，每个向量为 float 列表
    """
    if not texts:
        return []
    settings = get_settings()
    client = get_client()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    # 返回 data[i].embedding 列表
    return [data.embedding for data in response.data]


def embed_one(text: str) -> list[float]:
    """生成单条文本的嵌入向量。

    Args:
        text: 待嵌入的文本

    Returns:
        嵌入向量（float 列表）
    """
    return embed([text])[0]
