"""导入流程主图。

使用 LangGraph StateGraph 构建文档导入工作流；若 langgraph 未安装，
则降级为顺序执行函数 run_import_fallback。

流程结构：
    entry -> pdf_to_md -> document_split -> item_name_recognition
          -> bge_embedding -> import_milvus -> knowledge_graph -> END

注：pdf_to_md 负责把 PDF / Markdown 读取为 md_content，
    document_split 依赖该字段，因此必须位于 document_split 之前。
"""
from __future__ import annotations

import logging

from core.config import get_settings
from processor.import_process.state import ImportGraphState, create_default_state
from processor.import_process.nodes import (
    EntryNode,
    PdfToMdNode,
    DocumentSplitNode,
    ItemNameRecognitionNode,
    BgeEmbeddingNode,
    ImportMilvusNode,
    KnowledgeGraphNode,
)

logger = logging.getLogger(__name__)

# 尝试导入 langgraph，未安装时降级为顺序执行
try:
    from langgraph.graph import StateGraph, START, END

    _HAS_LANGGRAPH = True
except ImportError:  # langgraph 未安装
    _HAS_LANGGRAPH = False
    logger.warning("langgraph 未安装，将使用顺序执行模式 run_import_fallback")


def create_import_graph():
    """创建并编译导入流程图。

    Returns:
        编译后的 LangGraph 可执行图。

    Raises:
        RuntimeError: langgraph 未安装时调用。
    """
    if not _HAS_LANGGRAPH:
        raise RuntimeError("langgraph 未安装，无法构建图，请改用 run_import_fallback")

    # 1. 定义工作流
    workflow = StateGraph(ImportGraphState)

    # 2. 实例化并注册节点
    nodes = {
        "entry": EntryNode(),
        "pdf_to_md": PdfToMdNode(),
        "document_split": DocumentSplitNode(),
        "item_name_recognition": ItemNameRecognitionNode(),
        "bge_embedding": BgeEmbeddingNode(),
        "import_milvus": ImportMilvusNode(),
        "knowledge_graph": KnowledgeGraphNode(),
    }
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # 3. 设置入口节点（LangGraph 1.x 标准写法，0.2.x 兼容）
    workflow.add_edge(START, "entry")

    # 4. 顺序边
    workflow.add_edge("entry", "pdf_to_md")
    workflow.add_edge("pdf_to_md", "document_split")
    workflow.add_edge("document_split", "item_name_recognition")
    workflow.add_edge("item_name_recognition", "bge_embedding")
    workflow.add_edge("bge_embedding", "import_milvus")
    workflow.add_edge("import_milvus", "knowledge_graph")
    workflow.add_edge("knowledge_graph", END)

    # 5. 编译并返回
    return workflow.compile()


def run_import_fallback(file_path: str) -> dict:
    """langgraph 未安装时的顺序执行版本。

    按节点顺序依次处理 state，等价于线性图。
    """
    state = create_default_state(file_path=file_path)
    nodes = [
        EntryNode(),
        PdfToMdNode(),
        DocumentSplitNode(),
        ItemNameRecognitionNode(),
        BgeEmbeddingNode(),
        ImportMilvusNode(),
        KnowledgeGraphNode(),
    ]
    for node in nodes:
        state = node(state)
    return state


def run_import(file_path: str) -> dict:
    """便捷函数：运行导入流程。

    优先使用 LangGraph 图执行；若 langgraph 未安装则降级为顺序执行。

    Args:
        file_path: 待导入文件路径（PDF 或 Markdown）。

    Returns:
        最终状态字典。
    """
    # 触发配置加载，便于尽早暴露配置问题
    get_settings()

    if not _HAS_LANGGRAPH:
        logger.info("使用 fallback 顺序执行导入流程: %s", file_path)
        return run_import_fallback(file_path)

    app = create_import_graph()
    initial_state = create_default_state(file_path=file_path)
    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) < 2:
        print("用法: python -m processor.import_process.main_graph <文件路径>")
        sys.exit(1)

    target = sys.argv[1]
    result = run_import(target)
    print("=" * 50)
    print("导入流程完成")
    print(f"文件标题: {result.get('file_title', 'N/A')}")
    print(f"识别商品: {result.get('item_name', 'N/A')}")
    print(f"切片数量: {len(result.get('chunks', []))}")
