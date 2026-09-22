"""查询审计：每次查询把关键运行指标落盘成 JSONL。

借鉴 OpenClaw 的 run_summary.json 思想——每次运行留下可复核的现场：
各路检索计数、降级组件、耗时、答案长度。排查问题时看文件即可还原现场，
不依赖日志滚动翻找。

文件位置：data/query_runs/YYYY-MM-DD.jsonl（一行一条 JSON，按天分文件）。
写入失败只告警，绝不影响查询主流程。
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("query.audit")

_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = _ROOT / "data" / "query_runs"


def _chunk_ref(d: dict) -> dict:
    """切片引用（评估用轻量标识）。

    检索结果里没有全局唯一 chunk_id（RRF 以 content 去重），评估脚本
    与金标集统一用「内容 SHA256 前 16 位」对齐身份；item_name 用于
    分文档统计。
    """
    content = d.get("content", "")
    return {
        "item_name": d.get("item_name", "") or d.get("title", ""),
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
    }


def _config_snapshot() -> dict:
    """检索相关配置指纹（回归对比的锚点：不同 config 的评估结果不可比）。"""
    try:
        from core.config import get_settings
        s = get_settings()
    except Exception:
        return {}
    keys = (
        "rrf_k", "rrf_max_results", "embedding_search_limit",
        "embedding_model", "rerank_model", "llm_model", "chunks_collection",
    )
    return {k: getattr(s, k) for k in keys if hasattr(s, k)}


def audit_query_run(state: dict, elapsed_sec: float | None = None) -> dict:
    """把一次查询的运行指标追加写入审计文件。

    Args:
        state: 查询流程最终状态（QueryGraphState 或其子集）。
        elapsed_sec: 检索+生成总耗时（秒），可为 None。

    Returns:
        写入的审计记录（便于调用方继续使用，如日志打印）。
    """
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_id": state.get("session_id", ""),
        "query": (state.get("rewritten_query") or state.get("original_query", ""))[:200],
        "item_names": state.get("item_names", []),
        "counts": {
            "embedding": len(state.get("embedding_chunks", [])),
            "hyde": len(state.get("hyde_chunks", [])),
            "kg": len(state.get("kg_chunks", [])),
            "bm25": len(state.get("bm25_docs", [])),
            "rrf": len(state.get("rrf_chunks", [])),
            "reranked": len(state.get("reranked_docs", [])),
        },
        "kg_entities": state.get("kg_entities", []),
        "degraded": list(state.get("degraded", [])),
        "answer_chars": len(state.get("answer", "")),
        "elapsed_sec": round(elapsed_sec, 2) if elapsed_sec is not None else None,
        # ---- 评估闭环扩展字段 ----
        # 答案全文快照：LLM-as-judge（忠实度/幻觉率）的评估原料
        "answer": state.get("answer", ""),
        # 首字延迟（秒），流式生成时由 AnswerOutputNode 记录
        "ttft_sec": state.get("ttft_sec"),
        # 分路明细（各路前 10 条）：分路 recall 与检索/排序归因的数据源
        "per_route": {
            "embedding": [_chunk_ref(c) for c in state.get("embedding_chunks", [])[:10]],
            "hyde": [_chunk_ref(c) for c in state.get("hyde_chunks", [])[:10]],
            "kg": [_chunk_ref(c) for c in state.get("kg_chunks", [])[:10]],
            "bm25": [_chunk_ref(c) for c in state.get("bm25_docs", [])[:10]],
        },
        # 最终 top-k（精排+断崖截断后）：recall@k 的直接数据源
        "final_topk": [_chunk_ref(c) for c in state.get("reranked_docs", [])],
        # 最终送入 LLM 的上下文原文：faithfulness 判定依据
        "contexts": [c.get("content", "") for c in state.get("reranked_docs", [])],
        # 检索配置指纹：两次评估结果可比的前提
        "config": _config_snapshot(),
    }
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path = RUNS_DIR / f"{time.strftime('%Y-%m-%d')}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:  # 审计绝不能拖垮查询
        logger.warning("审计写入失败（不影响查询）: %s", e)
    return rec
