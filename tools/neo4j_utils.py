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
                ON CREATE SET n.types = $types
                ON MATCH SET n.types = $types
                """,
                name=name,
                item_name=entity_item,
                types=types,
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
    精确名未命中且提供了 item_name 时，退化为子图内双向包含的模糊匹配
    （锚点名「RX-27 路由器」能回退命中节点名「RX-27」）。

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

    logger.info(
        "图查询完成: entity=%s, item_name=%s, 命中 %d 条三元组",
        entity_name, item_name, len(triples),
    )
    return triples


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
