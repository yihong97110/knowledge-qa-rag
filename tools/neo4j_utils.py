"""Neo4j 知识图谱工具模块。

提供 Neo4j 驱动管理、实体关系创建、图查询和数据清除功能。
基于 neo4j Python driver，使用 session.run() 执行 Cypher。
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from neo4j import GraphDatabase

from core.config import get_settings

logger = logging.getLogger(__name__)

# Neo4j 驱动单例
_driver = None

# 关系类型合法字符校验（仅允许大写字母、数字、下划线，防止 Cypher 注入）
_REL_TYPE_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")

# ---------------- 向量索引（实体链接第三级：语义变体兜底） ----------------

# 向量索引状态单例：None=未检查 / True=可用 / False=不可用（Neo4j 版本过旧或创建失败）
_vector_index_ready: Optional[bool] = None

# 向量召回参数：只兜前两级 miss 的实体，top-k 与阈值都取保守值
_VECTOR_INDEX_NAME = "entity_name_vec"
_VECTOR_TOP_K = 3
_VECTOR_MIN_SCORE = 0.70


def get_driver():
    """返回 GraphDatabase.driver 单例。

    首次调用时根据 config 中的 neo4j_uri / neo4j_username / neo4j_password 创建驱动，
    后续调用复用同一实例。
    """
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        logger.info("Neo4j 驱动已创建: %s", settings.neo4j_uri)
    return _driver


def close() -> None:
    """关闭 Neo4j 驱动。"""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
        logger.info("Neo4j 驱动已关闭")


def _ensure_vector_index() -> bool:
    """确保 Entity.name_embedding 的原生向量索引存在（Neo4j 5.13+）。

    每进程只尝试一次；失败时置 False 并告警，查询侧自动跳过向量档，
    退回纯字符串匹配——向量档是增强能力，不能阻塞主链路。
    """
    global _vector_index_ready
    if _vector_index_ready is not None:
        return _vector_index_ready

    settings = get_settings()
    dims = int(settings.embedding_dim)
    cypher = (
        f"CREATE VECTOR INDEX {_VECTOR_INDEX_NAME} IF NOT EXISTS "
        "FOR (n:Entity) ON n.name_embedding "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dims}, "
        f"`vector.similarity_function`: 'cosine'}}}}"
    )
    driver = get_driver()
    try:
        with driver.session(database=settings.neo4j_database) as session:
            session.run(cypher)
        _vector_index_ready = True
        logger.info("Neo4j 向量索引就绪: %s (dims=%d)", _VECTOR_INDEX_NAME, dims)
    except Exception as exc:
        _vector_index_ready = False
        logger.warning("Neo4j 向量索引创建失败，实体链接退回纯字符串匹配: %s", exc)
    return _vector_index_ready


def _embed_names(names: list[str]) -> dict[str, list]:
    """批量向量化实体名，返回 {name: vector}。

    DashScope 单请求上限 10 条，按 10 分批；失败时返回已成功的部分
    （未拿到向量的实体写入 null，可由 backfill_entity_embeddings 补齐）。
    """
    if not names:
        return {}
    try:
        from tools.embedding_utils import embed
    except Exception as exc:  # 依赖缺失时静默降级，不阻塞导入主链路
        logger.warning("embedding 模块不可用，跳过实体名向量化: %s", exc)
        return {}

    result: dict[str, list] = {}
    try:
        for i in range(0, len(names), 10):
            batch = names[i:i + 10]
            vectors = embed(batch)
            result.update(zip(batch, vectors))
    except Exception as exc:
        logger.warning("实体名向量化失败（已成功 %d 个）: %s", len(result), exc)
    return result


def _query_vector_names(
    entity_name: str,
    item_name: Optional[str],
) -> list[str]:
    """向量索引召回与 entity_name 语义相似的节点名（不负责遍历）。

    背景：精确/子串两级都是字面匹配，用户口语（「路由器产品」）与
    受控实体名（「RX-27 路由器」）可能字面零重叠。用原生向量索引做
    余弦 top-k，item_name 限定时只收同子图候选，缩小误召回面。
    """
    if not _ensure_vector_index():
        return []

    settings = get_settings()
    try:
        from tools.embedding_utils import embed
        query_vec = embed([entity_name])[0]
    except Exception as exc:
        logger.warning("查询实体向量化失败，跳过向量召回: %s", exc)
        return []

    driver = get_driver()
    try:
        with driver.session(database=settings.neo4j_database) as session:
            result = session.run(
                f"""
                CALL db.index.vector.queryNodes('{_VECTOR_INDEX_NAME}', $k, $vec)
                YIELD node, score
                WHERE score >= $min_score
                  AND node.name IS NOT NULL
                  AND ($item_name IS NULL OR node.item_name = $item_name)
                RETURN node.name AS name, score
                ORDER BY score DESC
                """,
                k=_VECTOR_TOP_K,
                vec=query_vec,
                min_score=_VECTOR_MIN_SCORE,
                item_name=item_name,
            )
            return [record["name"] for record in result if record["name"]]
    except Exception as exc:
        logger.warning("向量召回查询失败，跳过: %s", exc)
        return []


def create_entities_and_relations(
    entities: list[dict],
    relations: list[dict],
    item_name: str,
) -> None:
    """用 MERGE 创建实体和关系。

    实体使用 (n:Entity {name: $name, item_name: $item_name}) 做唯一约束，
    重复执行不会创建重复数据。

    Args:
        entities: 实体列表，每个 dict 含 name, types, item_name。
        relations: 关系列表，每个 dict 含 source, target, type。
        item_name: 商品名称，用于关联实体。
    """
    driver = get_driver()
    settings = get_settings()

    # 确保 name_embedding 向量索引存在（写入与检索依赖同一索引）
    _ensure_vector_index()

    # 实体名批量向量化，写入节点属性供向量索引使用
    unique_names = list(dict.fromkeys(
        str(e.get("name", "")).strip()
        for e in entities
        if str(e.get("name", "")).strip()
    ))
    emb_map = _embed_names(unique_names)

    with driver.session(database=settings.neo4j_database) as session:
        # 创建实体（MERGE 保证幂等）
        for entity in entities:
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            types = entity.get("types", [])
            entity_item = entity.get("item_name", item_name)

            session.run(
                """
                MERGE (n:Entity {name: $name, item_name: $item_name})
                ON CREATE SET n.types = $types,
                              n.name_embedding = $name_embedding
                ON MATCH SET n.types = $types,
                             n.name_embedding = $name_embedding
                """,
                name=name,
                item_name=entity_item,
                types=types,
                name_embedding=emb_map.get(name),
            )

        # 创建关系（MERGE 保证幂等）
        for rel in relations:
            source = str(rel.get("source", "")).strip()
            target = str(rel.get("target", "")).strip()
            rel_type = str(rel.get("type", "RELATED_TO")).strip().upper()

            if not source or not target:
                continue

            # 校验关系类型，非法则降级为 RELATED_TO，防止 Cypher 注入
            if not _REL_TYPE_PATTERN.match(rel_type):
                rel_type = "RELATED_TO"

            # 关系类型无法参数化，需拼接进 Cypher（已通过白名单校验保证安全）
            session.run(
                f"""
                MATCH (h:Entity {{name: $source, item_name: $item_name}})
                MATCH (t:Entity {{name: $target, item_name: $item_name}})
                MERGE (h)-[:{rel_type}]->(t)
                """,
                source=source,
                target=target,
                item_name=item_name,
            )

    logger.info(
        "图数据写入完成: %d 个实体, %d 条关系 (item_name=%s)",
        len(entities), len(relations), item_name,
    )


def _query_exact(
    entity_name: str,
    item_name: Optional[str],
    max_depth: int,
) -> list[dict]:
    """按精确实体名查询子图三元组（query_by_entity 的内部实现）。"""
    driver = get_driver()
    settings = get_settings()
    depth = int(max_depth)

    with driver.session(database=settings.neo4j_database) as session:
        # Cypher 不支持参数化关系深度，用字符串拼接（depth 已转 int，安全）
        cypher = (
            "MATCH (n:Entity {name: $entity_name})"
            f"-[rels*1..{depth}]-(m) "
            "WHERE $item_name IS NULL OR n.item_name = $item_name "
            "UNWIND rels AS rel "
            "RETURN DISTINCT "
            "  startNode(rel).name AS source, "
            "  type(rel) AS relation, "
            "  endNode(rel).name AS target"
        )
        result = session.run(
            cypher,
            entity_name=entity_name,
            item_name=item_name,
        )

        triples: list[dict] = []
        for record in result:
            triples.append({
                "source": record["source"] or "",
                "relation": record["relation"],
                "target": record["target"] or "",
            })

    return triples


def _find_fuzzy_names(
    entity_name: str,
    item_name: str,
) -> list[str]:
    """在 item 子图内找与 entity_name 互相包含的节点名（模糊回退候选）。

    背景：锚点 item_name 与图谱 Device 节点名来自不同 LLM 调用，
    可能不一致（如 item_name「RX-27 路由器」vs 节点名「RX-27」），
    精确匹配落空时用它兜底。按名字长度倒序，更具体的名字优先。
    """
    driver = get_driver()
    settings = get_settings()

    with driver.session(database=settings.neo4j_database) as session:
        result = session.run(
            """
            MATCH (n:Entity {item_name: $item_name})
            WHERE n.name CONTAINS $entity_name
               OR ($entity_name CONTAINS n.name AND size(n.name) >= 2)
            RETURN n.name AS name
            ORDER BY size(n.name) DESC
            LIMIT 3
            """,
            entity_name=entity_name,
            item_name=item_name,
        )
        return [record["name"] for record in result if record["name"]]


def query_by_entity(
    entity_name: str,
    item_name: Optional[str] = None,
    max_depth: int = 2,
) -> list[dict]:
    """从实体出发查询关联节点和关系。

    使用可变长度路径匹配（1..max_depth 跳），返回沿途所有三元组。
    实体链接为三级瀑布：
    1. 精确匹配（属性等值）；
    2. 子图内双向子串包含回退（锚点名「RX-27 路由器」命中节点「RX-27」）；
    3. Neo4j 原生向量索引相似召回（语义变体兜底，如「路由器产品」→「RX-27 路由器」），
       前两级都落空才触发，且只在向量索引可用时生效。

    Args:
        entity_name: 起始实体名称。
        item_name: 商品名称过滤（可选，为 None 则不过滤）。
        max_depth: 最大查询深度，默认 2。

    Returns:
        三元组列表 [{source, relation, target}, ...]。
    """
    triples = _query_exact(entity_name, item_name, max_depth)
    if triples:
        logger.info(
            "图查询完成: entity=%s, item_name=%s, 命中 %d 条三元组",
            entity_name, item_name, len(triples),
        )
        return triples

    # 模糊回退：仅在限定了 item_name 时才有可靠的候选范围
    if item_name:
        for name in _find_fuzzy_names(entity_name, item_name):
            if name == entity_name:
                continue
            triples.extend(_query_exact(name, item_name, max_depth))

    # 第三级：向量相似召回，兜住字面零重叠的语义变体（口语名 vs 受控实体名）
    if not triples:
        for name in _query_vector_names(entity_name, item_name):
            if name == entity_name:
                continue
            triples.extend(_query_exact(name, item_name, max_depth))

    logger.info(
        "图查询完成: entity=%s, item_name=%s, 命中 %d 条三元组",
        entity_name, item_name, len(triples),
    )
    return triples


def backfill_entity_embeddings(batch_size: int = 10, limit: int = 500) -> int:
    """给缺失 name_embedding 的存量 Entity 节点补算向量。

    场景：向量索引是后加的升级，升级前已入库的节点没有向量属性，
    不会被向量索引收录；导入侧新写入的节点自带向量，无需回填。

    Args:
        batch_size: 每批向量化条数（DashScope 单请求上限 10）。
        limit: 单次回填上限，防止一次性调爆 embedding 额度。

    Returns:
        本次补齐的节点数。
    """
    _ensure_vector_index()
    settings = get_settings()
    driver = get_driver()

    with driver.session(database=settings.neo4j_database) as session:
        records = session.run(
            """
            MATCH (n:Entity)
            WHERE n.name_embedding IS NULL
              AND n.name IS NOT NULL AND trim(n.name) <> ''
            RETURN DISTINCT n.name AS name
            LIMIT $limit
            """,
            limit=int(limit),
        )
        names = [r["name"] for r in records if r["name"]]

    if not names:
        logger.info("向量回填：无缺失向量的实体")
        return 0

    vectors = _embed_names(names)
    written = 0
    with driver.session(database=settings.neo4j_database) as session:
        for name, vec in vectors.items():
            session.run(
                "MATCH (n:Entity {name: $name}) SET n.name_embedding = $vec",
                name=name,
                vec=vec,
            )
            written += 1

    logger.info("向量回填完成: %d / %d 个实体", written, len(names))
    return written


def clear_all(item_name: Optional[str] = None) -> None:
    """清除图谱数据。

    Args:
        item_name: 指定商品名则只清除该商品的图谱数据；为 None 则清除全部。
    """
    driver = get_driver()
    settings = get_settings()

    with driver.session(database=settings.neo4j_database) as session:
        if item_name:
            session.run(
                "MATCH (n) WHERE n.item_name = $item_name DETACH DELETE n",
                item_name=item_name,
            )
            logger.info("已清除 item_name=%s 的图谱数据", item_name)
        else:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("已清除全部图谱数据")
