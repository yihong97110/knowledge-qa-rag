"""文件会话历史存储：MongoDB 不可用时的持久化降级方案。

数据落地为 JSONL（每行一条 JSON），重启/关机后历史仍在。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

# 与 graph_data.json 一致，落在项目根 data/ 目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_lock = threading.Lock()


def _file_path() -> Path:
    path = _PROJECT_ROOT / "data" / "chat_history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def add_message(session_id: str, role: str, text: str, extra: dict = None) -> None:
    """追加一条对话记录到 JSONL 文件。"""
    record = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "ts": datetime_now(),
    }
    if extra:
        record.update(extra)
    with _lock:
        with _file_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_history(session_id: str, limit: int = 10) -> list[dict]:
    """按 session_id 过滤，返回最近 limit 条（按时间正序）。"""
    path = _file_path()
    if not path.exists():
        return []
    records: list[dict] = []
    with _lock:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 跳过损坏行
                if rec.get("session_id") == session_id:
                    records.append(rec)
    return records[-limit:]


def clear_history(session_id: str) -> None:
    """删除指定会话的全部记录（重写文件）。"""
    path = _file_path()
    if not path.exists():
        return
    kept: list[str] = []
    with _lock:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError:
                    kept.append(stripped)
                    continue
                if rec.get("session_id") != session_id:
                    kept.append(stripped)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for line in kept:
                f.write(line + "\n")
        tmp.replace(path)


def datetime_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
