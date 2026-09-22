"""查询流程主图

使用 LangGraph 构建知识库查询工作流。
如果 langgraph 未安装，提供 run_query_fallback 顺序执行。

流程结构::

    item_name_confirm
          │
          ├── (有答案) ──────────────────────> answer_output
          │                                        │
          └── (无答案) ──> multi_search ──┬──>    │
                               │          │       │
                     ┌─────────┼──────────┼───┐   │
                     │         │          │   │   │
                     v         v          v   v   │
               embedding  hyde    kg      bm25    │  （四路并行召回）
                     │         │          │   │   │
                     └─────────┴──────────┴───┘   │
                                  │               │
                                  v               │
                                join              │
                                  │               │
                                  v               │
                                 rrf              │  （RRF 融合排序）
                                  │               │
                                  v               │
                               rerank             │  （Cross-Encoder 精排+断崖截断）
                                  │               │
                                  v               │
                          answer_output <─────────┘
                                  │
                                  v
                                END
"""
from __future__ import annotations

import logging
import time

# 尝试导入 langgraph，未安装时提供 fallback
try:
    from langgraph.graph import StateGraph, START, END
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    StateGraph = None  # type: ignore
    START = None  # type: ignore
    END = None  # type: ignore

from processor.query_process.state import (
    QueryGraphState,
    create_default_state,
)
from tools.query_audit import audit_query_run
from processor.query_process.nodes import (
    ItemNameConfirmNode,
    SearchEmbeddingNode,
    SearchHydeNode,
    QueryKgNode,
    BM25SearchNode,
    RrfNode,
    RerankerNode,
    AnswerOutputNode,
)

logger = logging.getLogger("query.main_graph")


# ================================================================== #
#                       条件路由                                      #
# ================================================================== #

def route_after_item_confirm(state: QueryGraphState) -> str:
    """商品名确认后的路由逻辑。

    根据是否已有答案决定是否跳过搜索直接输出。

    Args:
        state: 查询图状态。

    Returns:
        "answer_output" 表示已有答案直接输出；
        "multi_search" 表示继续检索流程。
    """
    if state.get("answer"):
        return "answer_output"
    return "multi_search"


# ================================================================== #
#                       图构建                                        #
# ================================================================== #

def create_query_graph():
    """创建查询流程图。

    Returns:
        编译后的 StateGraph 实例。

    Raises:
        ImportError: langgraph 未安装时抛出。
    """
    if not _LANGGRAPH_AVAILABLE:
        raise ImportError(
            "langgraph 未安装，请运行 pip install langgraph，"
            "或使用 run_query_fallback() 顺序执行。"
        )

    # 1. 创建状态图
    workflow = StateGraph(QueryGraphState)

    # 2. 实例化节点
    nodes = {
        "item_name_confirm": ItemNameConfirmNode(),
        "multi_search": lambda state: {},   # 多路搜索分发（虚节点）
        "search_embedding": SearchEmbeddingNode(),
        "search_embedding_hyde": SearchHydeNode(),
        "query_kg": QueryKgNode(),
        "bm25_search": BM25SearchNode(),       # 替换 web_search：BM25 稀疏检索
        "join": lambda state: {},           # 多路搜索汇合（虚节点）
        "rrf": RrfNode(),
        "rerank": RerankerNode(),             # 新增：Cross-Encoder 精排 + 断崖检测
        "answer_output": AnswerOutputNode(),
    }

    # 3. 添加节点
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # 4. 设置入口点
    workflow.add_edge(START, "item_name_confirm")

    # 5. 条件边：商品名确认后根据是否有答案路由
    workflow.add_conditional_edges(
        "item_name_confirm",
        route_after_item_confirm,
        {
            "answer_output": "answer_output",   # 有答案，直接输出
            "multi_search": "multi_search",     # 无答案，继续检索
        },
    )

    # 6. 多路搜索分发（并行执行）
    workflow.add_edge("multi_search", "search_embedding")
    workflow.add_edge("multi_search", "search_embedding_hyde")
    workflow.add_edge("multi_search", "query_kg")
    workflow.add_edge("multi_search", "bm25_search")

    # 7. 多路搜索汇合
    workflow.add_edge("search_embedding", "join")
    workflow.add_edge("search_embedding_hyde", "join")
    workflow.add_edge("query_kg", "join")
    workflow.add_edge("bm25_search", "join")

    # 8. 顺序边：RRF 融合 → Reranker 精排+断崖截断 → 答案生成
    workflow.add_edge("join", "rrf")
    workflow.add_edge("rrf", "rerank")
    workflow.add_edge("rerank", "answer_output")
    workflow.add_edge("answer_output", END)

    # 9. 编译并返回
    return workflow.compile()


# 全局图实例（懒加载）
_query_app = None


def _get_query_app():
    """获取全局图实例（懒加载）。"""
    global _query_app
    if _query_app is None and _LANGGRAPH_AVAILABLE:
        _query_app = create_query_graph()
    return _query_app


# ================================================================== #
#                       便捷函数                                      #
# ================================================================== #

def run_query(query: str, session_id: str = "", is_stream: bool = False) -> dict:
    """便捷函数：运行查询流程。

    如果 langgraph 已安装，使用图流程；否则使用 fallback 顺序执行。

    Args:
        query: 用户查询文本。
        session_id: 会话 ID，默认为 "default"。
        is_stream: 是否启用流式输出。

    Returns:
        最终状态字典。
    """
    if _LANGGRAPH_AVAILABLE:
        # 使用 LangGraph 图流程
        app = _get_query_app()
        initial_state = create_default_state(
            session_id=session_id or "default",
            original_query=query,
            is_stream=is_stream,
        )
        t0 = time.monotonic()
        final_state = app.invoke(initial_state)
        audit_query_run(final_state, elapsed_sec=time.monotonic() - t0)
        return final_state
    else:
        # langgraph 未安装，使用 fallback
        logger.warning("langgraph 未安装，使用 fallback 顺序执行")
        return run_query_fallback(query, session_id, is_stream)


def run_query_fallback(query: str, session_id: str = "", is_stream: bool = False) -> dict:
    """Fallback：不依赖 langgraph，顺序执行查询流程。

    Args:
        query: 用户查询文本。
        session_id: 会话 ID。
        is_stream: 是否启用流式输出。

    Returns:
        最终状态字典。
    """
    # 1. 创建初始状态
    state = create_default_state(
        session_id=session_id or "default",
        original_query=query,
        is_stream=is_stream,
    )

    # 2. 商品名确认
    state.update(ItemNameConfirmNode()(state))

    # 3. 条件路由：有答案则跳过检索
    if not state.get("answer"):
        # 4. 多路检索（顺序执行，模拟并行）
        state.update(SearchEmbeddingNode()(state))
        state.update(SearchHydeNode()(state))
        state.update(QueryKgNode()(state))
        state.update(BM25SearchNode()(state))

        # 5. RRF 融合 → Reranker 精排 + 断崖截断
        state.update(RrfNode()(state))
        state.update(RerankerNode()(state))

    # 6. 答案输出
    state.update(AnswerOutputNode()(state))

    # 7. 审计落盘（fallback 路径与图路径保持同等可观测性）
    audit_query_run(state)

    return state


# ================================================================== #
#                       命令行入口                                    #
# ================================================================== #

if __name__ == "__main__":
    import sys

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 60)
    print("知识库查询流程测试")
    print(f"langgraph 可用: {_LANGGRAPH_AVAILABLE}")
    print("=" * 60)

    # 支持命令行参数
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        test_query = "万用表如何测量电压？"

    print(f"查询: {test_query}")
    print("-" * 60)

    try:
        result = run_query(
            query=test_query,
            session_id="test_001",
            is_stream=False,
        )

        print("-" * 60)
        print("流程完成!")
        print("-" * 60)

        # 输出结果摘要
        print(f"识别商品: {result.get('item_names', [])}")
        print(f"检索切片数: {len(result.get('embedding_chunks', []))}")
        print(f"HyDE 切片数: {len(result.get('hyde_chunks', []))}")
        print(f"KG 切片数: {len(result.get('kg_chunks', []))}")
        print(f"BM25 结果数: {len(result.get('bm25_docs', []))}")
        print(f"RRF 融合数: {len(result.get('rrf_chunks', []))}")
        print("-" * 60)

        # 输出答案（截取前 500 字符）
        answer = result.get("answer", "N/A")
        print("答案:")
        print(answer[:500] + "..." if len(answer) > 500 else answer)

    except Exception as e:
        print(f"流程执行失败: {e}")
        import traceback
        traceback.print_exc()
