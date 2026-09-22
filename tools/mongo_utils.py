"""MongoDB 会话历史管理：基于 pymongo，提供对话历史的增删查改。"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection

from core.config import get_settings

# 对话历史 collection 名称
HISTORY_COLLECTION = "chat_history"
# 文档指纹 collection 名称（content_hash -> 历史导入使用过的 item_name 列表）
FINGERPRINT_COLLECTION = "doc_fingerprints"

IMPORT_TASKS_COLLECTION = "import_tasks"

# MongoClient 单例
_client: Optional[MongoClient] = None
# 索引是否已建立（索引持久化于数据库，无需重复创建）
_index_ensured: bool = False


def get_client() -> MongoClient:
    """返回 MongoClient 单例，首次调用时根据配置创建连接。"""
    global _client
    if _client is None:
        settings = get_settings()
        _client = MongoClient(settings.mongo_url)
    return _client


def get_collection() -> Collection:
    """返回对话历史 collection，并确保查询所需索引已建立。"""
    global _index_ensured
    settings = get_settings()
    coll = get_client()[settings.mongo_db][HISTORY_COLLECTION]
    if not _index_ensured:
        # 按 session_id 过滤、按 ts 倒序取最近记录的复合索引
        coll.create_index([("session_id", 1), ("ts", -1)])
        _index_ensured = True
    return coll


def add_message(session_id: str, role: str, text: str, extra: dict = None) -> None:
    """添加一条对话记录。

    Args:
        session_id: 会话ID
        role: 消息角色（user / assistant）
        text: 消息文本
        extra: 附加字段，可包含 rewritten_query、item_names 等，展开存入文档
    """
    doc: dict = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "ts": datetime.now(timezone.utc),
    }
    if extra:
        doc.update(extra)
    get_collection().insert_one(doc)


def get_history(session_id: str, limit: int = 10) -> list[dict]:
    """获取指定会话最近 N 条历史，按时间正序返回。"""
    cursor = (
        get_collection()
        .find({"session_id": session_id}, {"_id": 0})
        .sort("ts", -1)
        .limit(limit)
    )
    # 倒序取出最近 N 条后反转为时间正序
    return list(reversed(list(cursor)))


def clear_history(session_id: str) -> None:
    """清除指定会话的全部历史记录。"""
    get_collection().delete_many({"session_id": session_id})


def close() -> None:
    """关闭 MongoDB 连接并重置单例。"""
    global _client
    if _client is not None:
        _client.close()
        _client = None


# ---------------------------------------------------------------------------
# 文档指纹：同一文档重导时按历史 item_name 清理旧数据（幂等的关键）
#
# 背景：产品名 item_name 由 LLM 从文档抽取，同一文档两次导入可能给出
# 不同名字（如「RX-27 路由器」vs「RX-27 双频路由器」），而 Milvus 切片、
# Neo4j 图谱、graph_data.json 的清理都按 item_name 匹配——改名即漏清。
# 以文档内容 hash 为身份记录历史名字，重导时全部清理。
# ---------------------------------------------------------------------------

def _fingerprint_collection() -> Collection:
    """返回文档指纹 collection（无历史数据时按需建立索引）。"""
    settings = get_settings()
    return get_client()[settings.mongo_db][FINGERPRINT_COLLECTION]


def find_item_names_by_hash(content_hash: str) -> list[str]:
    """按内容 hash 查询该文档历史导入使用过的全部 item_name。"""
    doc = _fingerprint_collection().find_one({"_id": content_hash}, {"item_names": 1})
    return list(doc.get("item_names", [])) if doc else []


def upsert_fingerprint(content_hash: str, item_name: str) -> None:
    """记录/更新文档指纹：把本次导入的 item_name 并入该 hash 的历史列表。"""
    _fingerprint_collection().update_one(
        {"_id": content_hash},
        {"$addToSet": {"item_names": item_name}},
        upsert=True,
    )


# ---------------------------------------------------------------------------
# 导入任务持久化：API 进程把任务记录放在内存 dict，重启即丢，
# 落库到 import_tasks 后，重启仍可查询历史任务状态。
# ---------------------------------------------------------------------------

def _import_tasks_collection() -> Collection:
    settings = get_settings()
    return get_client()[settings.mongo_db][IMPORT_TASKS_COLLECTION]


def save_task(task: dict) -> None:
    """整份覆盖写入一条任务记录（以 task_id 为主键，幂等）。"""
    doc = dict(task)
    doc["_id"] = doc["task_id"]
    _import_tasks_collection().replace_one(
        {"_id": doc["task_id"]}, doc, upsert=True
    )


def load_task(task_id: str) -> dict | None:
    """按 task_id 读取任务记录（排除 _id 字段）。"""
    doc = _import_tasks_collection().find_one({"_id": task_id})
    if doc:
        doc.pop("_id", None)
    return doc
