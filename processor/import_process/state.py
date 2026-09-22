"""导入流程状态类型定义。

定义图状态结构 ImportGraphState 和默认状态工厂函数。
所有字段在节点之间传递，使用 TypedDict 提供类型提示。
"""
from __future__ import annotations

import copy
from typing import TypedDict


class ImportGraphState(TypedDict, total=False):
    """导入流程图状态。

    在整个导入流程的各节点之间传递的数据。
    使用 total=False 表示所有字段都是可选的，节点只需更新自己关心的字段。
    """

    # 任务标识，用于任务追踪
    task_id: str
    # 上传文件路径（PDF 或 Markdown）
    file_path: str
    # 文件标题（不含扩展名）
    file_title: str
    # 是否为 PDF 文件
    is_pdf: bool
    # Markdown / 纯文本内容
    md_content: str
    # 文档切片列表，每个 chunk 含 content、title、chunk_id
    chunks: list[dict]
    # 识别出的产品 / 设备名称
    item_name: str
    # 历史识别出的商品名称列表
    item_names: list[str]
    # 同一文档历史上导入使用过的产品名（LLM 抽取不稳定可能改名，
    # 重导时按这些旧名清理 Milvus 切片与图谱数据，保证幂等）
    previous_item_names: list[str]


# 默认状态模板
GRAPH_DEFAULT_STATE: ImportGraphState = {
    "task_id": "",
    "file_path": "",
    "file_title": "",
    "is_pdf": False,
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "item_names": [],
    "previous_item_names": [],
}


def create_default_state(**overrides) -> ImportGraphState:
    """创建默认状态，支持覆盖字段。

    内部对全局模板做深拷贝，避免修改污染全局模板。

    Args:
        **overrides: 需要覆盖的字段。

    Returns:
        新的状态实例。

    Examples:
        >>> state = create_default_state(task_id="task_001", file_path="doc.pdf")
    """
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state
