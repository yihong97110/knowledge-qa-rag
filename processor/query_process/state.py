"""查询流程状态类型定义

定义查询流程中在节点间传递的完整状态结构，
并提供默认状态工厂函数。
"""
from __future__ import annotations

import copy
from typing import TypedDict


class QueryGraphState(TypedDict, total=False):
    """查询流程图状态。

    使用 total=False，所有字段均可选，节点只需更新自己关心的字段。

    Attributes:
        session_id: 会话 ID，用于追踪多轮对话。
        original_query: 用户原始查询。
        item_names: 从查询中识别出的产品名称列表。
        rewritten_query: 经过商品名确认后改写的查询。
        embedding_chunks: 向量检索结果。
        hyde_chunks: HyDE 检索结果。
        kg_chunks: 知识图谱检索结果。
        kg_entities: KG 路实际查询的实体名列表（LLM 提取 + item_name 锚点），
            随审计落盘，便于定位「KG 零召回」时是实体没对上还是图谱没数据。
        bm25_docs: BM25 稀疏检索结果。
        rrf_chunks: RRF 融合后的切片列表。
        reranked_docs: Reranker 精排+断崖检测后的文档列表。
        prompt: 拼接好的提示词。
        answer: 最终生成的答案。
        history: 历史对话列表。
        is_stream: 是否启用流式输出。
        ttft_sec: 首字延迟（秒）。流式生成时由 AnswerOutputNode 记录
            （从开始调用 LLM 到收到首个 token），非流式为 None；随审计
            落盘，用于端到端性能评估（P50/P95）。
        degraded: 降级组件列表（如 ["kg", "rerank"]），检索/生成路径上
            某环节失败回退时追加，随审计落盘并透传到 SSE final 事件。
    """

    session_id: str
    original_query: str
    item_names: list[str]
    rewritten_query: str
    embedding_chunks: list[dict]
    hyde_chunks: list[dict]
    kg_chunks: list[dict]
    kg_entities: list[str]
    bm25_docs: list[dict]
    rrf_chunks: list[dict]
    reranked_docs: list[dict]
    prompt: str
    answer: str
    history: list[dict]
    is_stream: bool
    ttft_sec: float | None
    degraded: list[str]


# ==================== 默认状态模板 ====================

DEFAULT_STATE: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "item_names": [],
    "rewritten_query": "",
    "embedding_chunks": [],
    "hyde_chunks": [],
    "kg_chunks": [],
    "kg_entities": [],
    "bm25_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "history": [],
    "is_stream": False,
    "ttft_sec": None,
    "degraded": [],
}


def create_default_state(**overrides) -> QueryGraphState:
    """创建默认状态，支持字段覆盖。

    Args:
        **overrides: 要覆盖的字段键值对。

    Returns:
        新的状态实例，包含默认值和覆盖值。

    Examples:
        >>> state = create_default_state(
        ...     session_id="session_001",
        ...     original_query="万用表如何测量电压？"
        ... )
    """
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state
