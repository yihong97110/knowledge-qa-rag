"""FastAPI 后端 - 对接真实 LangGraph 导入/查询流程。

启动后访问 http://127.0.0.1:8000 即可看到前端页面。

接口说明：
- POST /upload        上传文件，异步执行导入流程（解析→切分→向量化→入库→知识图谱）
- GET  /status/{id}   查询导入任务进度
- POST /query         发起查询，返回 session_id
- GET  /stream/{id}   SSE 流式推送查询结果（商品名识别→多路检索→RRF→答案生成）
- GET  /history/{id}  获取会话历史
- DELETE /history/{id} 清除会话历史
- GET  /health        健康检查
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

app = FastAPI(title="企业技术文档智能问答系统", version="1.0.0")

STATIC_DIR = _PROJECT_ROOT / "static"
UPLOAD_DIR = _PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 静态文件
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "import.html")


# ---------------------------------------------------------------------------
# 内存存储（任务进度 & 待处理查询）
# ---------------------------------------------------------------------------
_tasks: dict[str, dict] = {}
_pending_queries: dict[str, dict] = {}  # session_id -> {query, created_at}


def _update_task(task_id: str, **fields) -> None:
    """更新任务状态：内存 + MongoDB 双写（内存为主，落库兜底）。
    任务记录原先只在内存，重启后 /status 一律 404；落库后历史可查。"""
    task = _tasks.setdefault(
        task_id, {"task_id": task_id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    )
    task.update(fields)
    try:
        from tools.mongo_utils import save_task
        save_task(task)
    except Exception as e:
        logger.warning("任务状态落库失败 task=%s: %s", task_id, e)

# MongoDB 可用性标记（启动时检测）
_mongo_available: bool = False


def _check_mongo() -> bool:
    """检测 MongoDB 是否可连接（2 秒超时，不阻塞启动）。"""
    global _mongo_available
    try:
        from pymongo import MongoClient
        from core.config import get_settings
        settings = get_settings()
        client = MongoClient(settings.mongo_url, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        client.close()
        _mongo_available = True
        logger.info("MongoDB 连接成功")
    except Exception as e:
        _mongo_available = False
        logger.warning("MongoDB 不可用，历史记录将使用内存存储: %s", e)
    return _mongo_available


# ---------------------------------------------------------------------------
# 会话历史（MongoDB 优先 → JSONL 文件持久化 → 内存）
# ---------------------------------------------------------------------------
_sessions: dict[str, list[dict]] = {}  # 内存兜底存储


def _add_message(session_id: str, role: str, text: str, extra: dict = None) -> None:
    """添加一条对话记录。"""
    if _mongo_available:
        try:
            from tools.mongo_utils import add_message
            add_message(session_id, role, text, extra)
            return
        except Exception as e:
            logger.warning("MongoDB 写入失败，降级到文件: %s", e)
    try:
        # 文件持久化：重启/关机后历史仍在
        from tools.file_history import add_message as file_add
        file_add(session_id, role, text, extra)
    except Exception as e:
        logger.warning("文件历史写入失败，降级到内存: %s", e)
        if session_id not in _sessions:
            _sessions[session_id] = []
        _sessions[session_id].append({"role": role, "text": text, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})


def _get_history(session_id: str, limit: int = 10) -> list[dict]:
    """获取会话历史。"""
    if _mongo_available:
        try:
            from tools.mongo_utils import get_history
            return get_history(session_id, limit)
        except Exception as e:
            logger.warning("MongoDB 读取失败，降级到文件: %s", e)
    try:
        from tools.file_history import get_history as file_get
        return file_get(session_id, limit)
    except Exception as e:
        logger.warning("文件历史读取失败，降级到内存: %s", e)
        return _sessions.get(session_id, [])[-limit:]


def _clear_history(session_id: str) -> None:
    """清除会话历史。"""
    if _mongo_available:
        try:
            from tools.mongo_utils import clear_history
            clear_history(session_id)
            return
        except Exception:
            pass
    try:
        from tools.file_history import clear_history as file_clear
        file_clear(session_id)
    except Exception:
        pass
    _sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------
class QueryReq(BaseModel):
    query: str = ""
    session_id: str = ""
    is_stream: bool = True


# ---------------------------------------------------------------------------
# SSE 辅助函数
# ---------------------------------------------------------------------------
def _sse(event_type: str, data: dict) -> str:
    """格式化 SSE 事件。"""
    return f"data: {json.dumps({'type': event_type, 'data': data}, ensure_ascii=False)}\n\n"


# ===========================================================================
# 导入服务接口
# ===========================================================================

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    """上传文件，异步执行导入流程。"""
    task_ids = []
    for file in files:
        # 保存文件到 uploads 目录（先读内容算 hash）
        safe_name = file.filename or "unknown"
        content = await file.read()
        # 文件名用内容 hash 前缀：同一文档重传时覆盖旧文件，而非堆积副本
        content_hash = hashlib.sha256(content).hexdigest()
        file_path = UPLOAD_DIR / f"{content_hash[:8]}_{safe_name}"
        file_path.write_bytes(content)

        # 文档指纹：同一文档重导时，LLM 抽取的 item_name 可能改变，
        # 按 hash 找到历史使用过的 item_name，供导入管线清理旧数据
        try:
            from tools.mongo_utils import find_item_names_by_hash
            previous_item_names = find_item_names_by_hash(content_hash)
        except Exception as e:
            logger.warning("查询文档指纹失败（跳过旧数据清理）: %s", e)
            previous_item_names = []

        task_id = f"task_{uuid.uuid4().hex[:12]}"
        _update_task(
            task_id,
            status="pending",
            progress=0,
            message="等待处理...",
            file_name=safe_name,
            file_path=str(file_path),
        )
        task_ids.append(task_id)
        # 异步执行导入流程
        asyncio.create_task(
            _process_import(task_id, str(file_path), content_hash, previous_item_names)
        )

    return {"message": "Files uploaded successfully", "task_ids": task_ids}


async def _process_import(
    task_id: str,
    file_path: str,
    content_hash: str = "",
    previous_item_names: list[str] = None,
) -> None:
    """异步执行导入流程，逐步更新进度。

    依次调用 7 个节点：
    entry -> pdf_to_md -> document_split -> item_name_recognition
          -> bge_embedding -> import_milvus -> knowledge_graph
    """
    from processor.import_process.state import create_default_state as create_import_state
    from processor.import_process.nodes import (
        EntryNode,
        PdfToMdNode,
        DocumentSplitNode,
        ItemNameRecognitionNode,
        BgeEmbeddingNode,
        ImportMilvusNode,
        KnowledgeGraphNode,
    )

    _update_task(task_id, status="processing")

    steps = [
        (10, "正在解析文档...", EntryNode()),
        (25, "正在读取文档内容...", PdfToMdNode()),
        (40, "正在切分文档...", DocumentSplitNode()),
        (55, "正在识别产品名称...", ItemNameRecognitionNode()),
        (70, "正在生成向量嵌入...", BgeEmbeddingNode()),
        (85, "正在写入向量数据库...", ImportMilvusNode()),
        (95, "正在构建知识图谱...", KnowledgeGraphNode()),
    ]

    state = create_import_state(
        file_path=file_path,
        task_id=task_id,
        previous_item_names=list(previous_item_names or []),
    )

    for progress, message, node in steps:
        _update_task(task_id, progress=progress, message=message)
        try:
            # 在线程中执行同步节点，避免阻塞事件循环
            result = await asyncio.to_thread(node, state)
            if isinstance(result, dict):
                state.update(result)
        except Exception as e:
            logger.error("导入失败 task=%s: %s", task_id, e, exc_info=True)
            _update_task(task_id, status="failed", message=f"导入失败: {e}")
            return

    chunk_count = len(state.get("chunks", []))
    item_name = state.get("item_name", "未知")

    # 导入成功后记录文档指纹：下次重导按历史 item_name 清旧数据
    if content_hash and item_name:
        try:
            from tools.mongo_utils import upsert_fingerprint
            upsert_fingerprint(content_hash, item_name)
        except Exception as e:
            logger.warning("写入文档指纹失败: %s", e)

    _update_task(
        task_id,
        progress=100,
        message=f"导入完成，共 {chunk_count} 个切片，产品名：{item_name}",
        status="completed",
    )


@app.get("/status/{task_id}")
def task_status(task_id: str):
    """查询任务处理状态。"""
    task = _tasks.get(task_id)
    if not task:
        # 内存未命中（如服务重启后）：回查 MongoDB 持久化记录并回填内存
        try:
            from tools.mongo_utils import load_task
            task = load_task(task_id)
        except Exception as e:
            logger.warning("回查任务记录失败 task=%s: %s", task_id, e)
            task = None
        if task:
            _tasks[task_id] = task
    if not task:
        raise HTTPException(404, f"任务不存在: {task_id}")
    return task


def _check_neo4j() -> bool:
    """实时探活 Neo4j（bolt 连通性验证，异常即视为不可用）。"""
    try:
        from tools.neo4j_utils import get_driver
        get_driver().verify_connectivity()
        return True
    except Exception as e:
        logger.warning("Neo4j 不可用: %s", e)
        return False


def _check_milvus() -> bool:
    """实时探活 Milvus（先建立幂等连接，再读集合元数据，不触发 load）。"""
    try:
        from pymilvus import utility
        from tools.milvus_utils import connect
        from core.config import get_settings
        connect()
        return bool(utility.has_collection(get_settings().chunks_collection))
    except Exception as e:
        logger.warning("Milvus 不可用: %s", e)
        return False


@app.get("/health")
def health():
    """健康检查：Mongo / Neo4j / Milvus 三存储并行实时探活。

    设计要点：健康端点必须反映"现在"而不是"启动那一刻"——
    存储恢复后无需重启服务，/health 即应转绿。
    """
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        f_mongo = ex.submit(_check_mongo)
        f_neo4j = ex.submit(_check_neo4j)
        f_milvus = ex.submit(_check_milvus)
        # 各探活限时兜底，防止单个存储挂起拖死整个 /health
        try:
            mongo_ok = f_mongo.result(timeout=4)
        except Exception as e:
            logger.warning("Mongo 探活超时/异常: %s", e)
            mongo_ok = False
        try:
            neo4j_ok = f_neo4j.result(timeout=4)
        except Exception as e:
            logger.warning("Neo4j 探活超时/异常: %s", e)
            neo4j_ok = False
        try:
            milvus_ok = f_milvus.result(timeout=6)
        except Exception as e:
            logger.warning("Milvus 探活超时/异常: %s", e)
            milvus_ok = False

    return {
        "status": "ok",
        "mongo": mongo_ok,
        "neo4j": neo4j_ok,
        "milvus": milvus_ok,
    }


# ===========================================================================
# 查询服务接口
# ===========================================================================

@app.post("/query")
async def query(req: QueryReq):
    """发起查询，返回 session_id。"""
    session_id = req.session_id or f"sess_{uuid.uuid4().hex[:8]}"

    # 存储待处理查询，供 SSE 端点读取
    _pending_queries[session_id] = {
        "query": req.query,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    return {
        "message": "流式处理中",
        "session_id": session_id,
        "answer": "",
    }


@app.get("/stream/{session_id}")
async def stream(session_id: str):
    """SSE 流式推送查询结果。

    流程：
    1. 商品名识别（LLM 提取产品名 / 闲聊直接回复）
    2. 多路检索（向量检索 + HyDE + 知识图谱 + BM25 稀疏检索）
    3. RRF 融合 -> Cross-Encoder 精排 + 断崖检测
    4. LLM 流式生成答案
    """
    pending = _pending_queries.pop(session_id, None)
    if not pending:
        raise HTTPException(404, f"会话不存在或已过期: {session_id}")

    query_text = pending["query"]

    async def event_stream():
        try:
            from tools.query_audit import audit_query_run
            t0 = time.monotonic()

            # 获取历史对话
            history = _get_history(session_id, limit=6)

            # ---- Phase 1: 商品名识别 + 多路检索 ----
            yield _sse("tool_start", {"name": "商品名识别"})

            state = await asyncio.to_thread(
                _run_pre_answer, query_text, session_id, history
            )

            # 闲聊直接回复
            if state.get("answer"):
                yield _sse("tool_end", {"name": "直接回复"})
                answer = state["answer"]
                for char in answer:
                    yield _sse("delta", {"delta": char})
                    await asyncio.sleep(0.01)
                yield _sse("final", {
                    "answer": answer,
                    "status": "completed",
                    "degraded": state.get("degraded", []),
                })
                state["answer"] = answer
                audit_query_run(state, elapsed_sec=time.monotonic() - t0)
                # 保存历史
                _add_message(session_id, "user", query_text)
                _add_message(session_id, "assistant", answer)
                return

            # ---- Phase 2: 检索完成，生成答案 ----
            yield _sse("tool_end", {"name": "检索完成，正在生成答案..."})

            messages = _build_answer_messages(state)
            full_answer = ""

            # 流式生成答案
            def _next_token(gen):
                try:
                    return next(gen)
                except StopIteration:
                    return None

            from tools.llm_utils import chat_stream
            gen = chat_stream(messages)
            t0_gen = time.monotonic()
            t_first: float | None = None
            while True:
                token = await asyncio.to_thread(_next_token, gen)
                if token is None:
                    break
                if t_first is None:
                    t_first = time.monotonic()
                full_answer += token
                yield _sse("delta", {"delta": token})

            state["ttft_sec"] = round(t_first - t0_gen, 3) if t_first is not None else None

            # degraded 标志随 final 事件透传给前端：让"这次答案是降级状态下
            # 生成的"显式可见，而不是静默吞掉
            yield _sse("final", {
                "answer": full_answer,
                "status": "completed",
                "degraded": state.get("degraded", []),
            })

            # 审计落盘：检索计数 / 降级组件 / 耗时一次留痕
            state["answer"] = full_answer
            audit_query_run(state, elapsed_sec=time.monotonic() - t0)

            # 保存历史
            _add_message(session_id, "user", query_text)
            _add_message(session_id, "assistant", full_answer)

        except Exception as e:
            logger.error("查询流式失败 session=%s: %s", session_id, e, exc_info=True)
            yield _sse("error", {"message": f"处理失败: {e}"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/history/{session_id}")
def history(session_id: str, limit: int = 10):
    """获取会话历史。"""
    msgs = _get_history(session_id, limit)
    return {"session_id": session_id, "items": msgs}


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    """清除会话历史。"""
    _clear_history(session_id)
    return {"deleted": session_id}


# ===========================================================================
# 查询流程辅助函数
# ===========================================================================

def _run_pre_answer(query: str, session_id: str, history: list[dict]) -> dict:
    """执行答案生成前的所有节点（同步，在线程中调用）。

    1. ItemNameConfirmNode  — 商品名识别 / 闲聊直接回复
    2. SearchEmbeddingNode  — 向量检索 top_k=10
    3. SearchHydeNode       — HyDE 假设文档检索 top_k=5
    4. QueryKgNode          — 知识图谱实体查询（max_depth=2 多跳）
    5. BM25SearchNode       — BM25 稀疏检索（关键词/型号字面匹配）
    6. RrfNode              — RRF 多路融合
    7. RerankerNode         — Cross-Encoder 精排 + 断崖检测动态截断

    Returns:
        包含 reranked_docs / rrf_chunks / answer 等字段的状态字典。
    """
    from processor.query_process.state import create_default_state
    from processor.query_process.nodes import (
        ItemNameConfirmNode,
        SearchEmbeddingNode,
        SearchHydeNode,
        QueryKgNode,
        BM25SearchNode,
        RrfNode,
        RerankerNode,
    )

    state = create_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=True,
    )
    state["history"] = history

    # 1. 商品名确认
    state.update(ItemNameConfirmNode()(state))

    # 如果闲聊直接回复，跳过检索
    if state.get("answer"):
        return state

    # 2-5. 多路检索（顺序执行，模拟并行）
    state.update(SearchEmbeddingNode()(state))
    state.update(SearchHydeNode()(state))
    state.update(QueryKgNode()(state))
    state.update(BM25SearchNode()(state))

    # 6. RRF 融合 → 7. Reranker 精排 + 断崖截断
    state.update(RrfNode()(state))
    state.update(RerankerNode()(state))

    return state


def _build_answer_messages(state: dict) -> list[dict]:
    """根据查询状态构建 LLM 答案生成的 messages。

    逻辑与 AnswerOutputNode.process 一致：
    系统提示词 + reranked_docs 上下文 + 历史对话 + 当前问题。
    优先用 Reranker 精排+断崖截断后的结果，回退到 rrf_chunks。
    """
    query = state.get("rewritten_query") or state.get("original_query", "")
    context_docs = state.get("reranked_docs") or state.get("rrf_chunks", [])
    history = state.get("history", [])

    # 构建上下文
    context_parts = []
    for i, chunk in enumerate(context_docs, 1):
        content = chunk.get("content", "")
        if content:
            context_parts.append(f"[{i}] {content}")
    context = "\n\n".join(context_parts) if context_parts else "（无检索结果）"

    system_prompt = (
        "你是一个专业的知识库问答助手。请根据以下知识库内容回答用户问题。\n"
        "如果知识库内容不足以回答问题，请如实说明。\n"
        "回答要准确、简洁、有条理。"
        f"\n\n知识库内容：\n{context}"
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    # 历史对话（最近 6 条）
    for msg in history[-6:]:
        role = msg.get("role", "user")
        text = msg.get("text", msg.get("content", ""))
        if text:
            messages.append({"role": role, "content": text})

    # 当前问题
    messages.append({"role": "user", "content": query})

    return messages


# ===========================================================================
# 知识图谱可视化接口
# ===========================================================================

@app.get("/graph")
def graph_data(item_name: Optional[str] = None):
    """返回知识图谱节点和边数据（供前端可视化）。

    数据源优先级：本地 graph_data.json（导入时持久化）。
    可选按 item_name 过滤。
    """
    graph_file = _PROJECT_ROOT / "data" / "graph_data.json"
    if not graph_file.exists():
        return {"nodes": [], "edges": [], "message": "暂无图谱数据，请先导入文档"}

    try:
        data = json.loads(graph_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"图谱数据读取失败: {e}")

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # 按 item_name 过滤
    if item_name:
        nodes = [n for n in nodes if n.get("item_name") == item_name]
        edges = [e for e in edges if e.get("item_name") == item_name]

    return {"nodes": nodes, "edges": edges}


@app.get("/graph/items")
def graph_items():
    """返回所有已导入的产品名列表（供前端筛选下拉框）。"""
    graph_file = _PROJECT_ROOT / "data" / "graph_data.json"
    if not graph_file.exists():
        return {"items": []}
    try:
        data = json.loads(graph_file.read_text(encoding="utf-8"))
    except Exception:
        return {"items": []}
    items = sorted({n.get("item_name", "") for n in data.get("nodes", []) if n.get("item_name")})
    return {"items": items}


# ===========================================================================
# 启动事件
# ===========================================================================

@app.on_event("startup")
async def startup_event():
    """启动时检测 MongoDB 连接。"""
    _check_mongo()


# ===========================================================================
# 启动
# ===========================================================================

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)
