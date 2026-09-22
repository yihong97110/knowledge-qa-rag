"""Milvus 向量数据库工具模块。

提供 Milvus 连接管理、集合创建、数据插入和向量检索功能。
基于 pymilvus ORM 风格 API（connections / Collection / utility）。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from core.config import get_settings

logger = logging.getLogger(__name__)

# 默认连接别名
_DEFAULT_ALIAS = "default"


def connect() -> None:
    """连接 Milvus 服务（从 config 读取 url）。

    幂等设计：已连接时直接返回，不重复建立连接。
    """
    settings = get_settings()
    if connections.has_connection(_DEFAULT_ALIAS):
        return
    connections.connect(alias=_DEFAULT_ALIAS, uri=settings.milvus_url)
    logger.info("Milvus 连接成功: %s", settings.milvus_url)


def ensure_collection(collection_name: str, dim: int) -> Collection:
    """确保 Collection 存在，不存在则创建。

    字段结构：
        - chunk_id:    INT64（主键）
        - dense_vector: FLOAT_VECTOR(dim)
        - content:     VARCHAR(65535)
        - title:       VARCHAR(65535)
        - item_name:   VARCHAR(65535)
        - created_at:  VARCHAR(65535)

    度量类型: COSINE

    Args:
        collection_name: 集合名称。
        dim: 稠密向量维度。

    Returns:
        Collection 对象。
    """
    connect()

    # 集合已存在则直接返回（幂等设计）
    if utility.has_collection(collection_name):
        logger.info("集合已存在: %s", collection_name)
        return Collection(collection_name)

    # 定义字段 schema
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="item_name", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=65535),
    ]

    schema = CollectionSchema(fields=fields, description="知识库切片向量集合")
    collection = Collection(name=collection_name, schema=schema)

    # 为 dense_vector 字段创建索引
    collection.create_index(
        field_name="dense_vector",
        index_params={
            "metric_type": "COSINE",
            "index_type": "AUTOINDEX",
            "params": {},
        },
    )
    logger.info("集合创建成功: %s, dim=%d", collection_name, dim)
    return collection


def insert_chunks(collection_name: str, chunks: list[dict]) -> list:
    """批量插入切片数据。

    每个 dict 应包含: content, dense_vector, title, item_name, chunk_id。
    若不含 created_at，则自动填充当前时间。

    Args:
        collection_name: 目标集合名称。
        chunks: 切片数据列表。

    Returns:
        插入结果（MutationResult），包含 insert_count 和 primary_keys。
    """
    connect()

    if not chunks:
        logger.warning("chunks 为空，跳过插入")
        return []

    collection = Collection(collection_name)

    # 为缺少 created_at 的切片自动填充时间戳
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    for chunk in chunks:
        if "created_at" not in chunk:
            chunk["created_at"] = now

    result = collection.insert(chunks)
    collection.flush()
    logger.info("成功插入 %d 条数据到 %s", len(chunks), collection_name)
    return result


def search(
    collection_name: str,
    query_vector: list[float],
    top_k: int = 10,
    item_name: Optional[str] = None,
) -> list[dict]:
    """向量检索。

    Args:
        collection_name: 集合名称。
        query_vector: 查询向量。
        top_k: 返回结果数量，默认 10。
        item_name: 按商品名称过滤（可选）。

    Returns:
        结果列表，每项为 {content, title, score, item_name}。
    """
    connect()

    collection = Collection(collection_name)
    collection.load()

    # 构建 item_name 过滤表达式
    expr = None
    if item_name:
        # 转义双引号，防止表达式注入
        escaped = item_name.replace('"', '\\"')
        expr = f'item_name == "{escaped}"'

    # 执行向量检索（带 item_name 过滤）
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_vector],
        anns_field="dense_vector",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["content", "title", "item_name"],
    )

    # 解析检索结果
    hits: list[dict] = []
    for hit in results[0]:
        entity = hit.entity
        hits.append({
            "content": entity.get("content", ""),
            "title": entity.get("title", ""),
            "score": hit.score,
            "item_name": entity.get("item_name", ""),
        })

    # 过滤检索为空时回退到全库检索（避免 item_name 不匹配导致漏召回）
    if not hits and expr is not None:
        logger.warning("按 item_name=%s 过滤检索 0 条，回退全库检索", item_name)
        results = collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            expr=None,
            output_fields=["content", "title", "item_name"],
        )
        for hit in results[0]:
            entity = hit.entity
            hits.append({
                "content": entity.get("content", ""),
                "title": entity.get("title", ""),
                "score": hit.score,
                "item_name": entity.get("item_name", ""),
            })

    logger.info("检索完成: %s, 命中 %d 条", collection_name, len(hits))
    return hits


def fetch_all_chunks() -> list[dict]:
    """拉取知识库全量切片（供 BM25 稀疏检索构造索引用）。

    BM25 不依赖向量，需要把整库切片捞出来本地构造倒排索引。
    必须用全库语料：若只按当前 item 过滤，小文档（1-2 个切片）里
    任何词 DF=N_docs，BM25Okapi 的 IDF 为负/零，得分全部 <=0，
    BM25 路恒空。全库语料让 DF 有分布差异，恢复区分能力。

    Returns:
        切片列表，每条包含 content/title/item_name 字段。
    """
    settings = get_settings()
    connect()
    if not utility.has_collection(settings.chunks_collection):
        return []
    col = Collection(settings.chunks_collection)
    col.load()
    res = col.query(
        expr="",
        output_fields=["content", "title", "item_name"],
        limit=10000,  # 单次拉取上限，演示库规模足够
    )
    return [
        {
            "content": r.get("content", ""),
            "title": r.get("title", ""),
            "item_name": r.get("item_name", ""),
        }
        for r in res
    ]


def count_chunks() -> int | None:
    """当前语料切片总数（供 BM25 索引新鲜度轻量校验）。

    Returns:
        切片总数；Milvus 不可用等异常时返回 None（调用方自行兜底）。
    """
    settings = get_settings()
    try:
        connect()
        if not utility.has_collection(settings.chunks_collection):
            return 0
        col = Collection(settings.chunks_collection)
        col.load()
        return col.num_entities
    except Exception as e:
        logger.warning("统计语料条数失败: %s", e)
        return None


def delete_item(collection_name: str, item_name: str) -> int:
    """按文档名删除其全部切片（用于改名重导/清理脏数据）。

    实现：先查全量主键在内存中过滤（避免 Milvus 表达式对含换行/引号的
    文档名无法转义的问题），再按 chunk_id 主键批量删除。

    Args:
        collection_name: 集合名称。
        item_name: 要删除的文档名（精确匹配）。

    Returns:
        删除的条数。
    """
    connect()

    collection = Collection(collection_name)
    collection.load()  # 服务刚启动时集合可能处于 released 状态，query 前必须 load
    rows = collection.query(
        expr="", output_fields=["chunk_id", "item_name"]
    )
    ids = [
        r["chunk_id"] for r in rows
        if str(r.get("item_name", "")) == item_name and r.get("chunk_id") is not None
    ]
    if not ids:
        logger.warning("文档「%r」在 %s 中无切片", item_name, collection_name)
        return 0
    result = collection.delete(f"chunk_id in {ids}")
    collection.flush()
    deleted = getattr(result, "delete_count", 0) or len(ids)
    logger.info("从 %s 删除文档「%r」共 %d 条", collection_name, item_name, deleted)
    return deleted


def list_item_names() -> list[str]:
    """返回知识库中所有产品名（去重、排序），供问题路由参考。

    Returns:
        产品名列表，如 ["HELM-BERT", "数字万用表", "灰姑娘"]。
    """
    settings = get_settings()
    connect()
    if not utility.has_collection(settings.chunks_collection):
        return []
    col = Collection(settings.chunks_collection)
    col.load()
    res = col.query(expr="", output_fields=["item_name"], limit=10000)
    names = sorted({str(r.get("item_name", "")).strip() for r in res} - {""})
    logger.info("知识库产品名: %s", names)
    return names


def disconnect() -> None:
    """断开 Milvus 连接。"""
    if connections.has_connection(_DEFAULT_ALIAS):
        connections.disconnect(_DEFAULT_ALIAS)
        logger.info("Milvus 连接已断开")
