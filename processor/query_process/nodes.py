"""查询流程节点定义

包含节点基类 BaseNode 和 8 个查询流程节点：
- ItemNameConfirmNode   商品名确认（入口节点）
- SearchEmbeddingNode   向量检索
- SearchHydeNode        HyDE 检索
- QueryKgNode           知识图谱检索
- BM25SearchNode        BM25 稀疏检索（关键词/型号字面匹配）
- RrfNode               RRF 多路融合
- RerankerNode          Cross-Encoder 精排 + 断崖检测
- AnswerOutputNode      答案生成输出
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from tools.llm_utils import chat, chat_json, chat_stream
from tools.embedding_utils import embed_one
from tools.milvus_utils import search as milvus_search
from tools.neo4j_utils import query_by_entity
from core.config import get_settings

from processor.query_process.state import QueryGraphState


# ================================================================== #
#                        节点基类（同导入流程）                        #
# ================================================================== #

class BaseNode(ABC):
    """查询流程节点基类。

    所有节点继承此基类，实现 process 方法即可。
    基类提供统一的日志输出和异常处理。
    """

    name: str = "base_node"

    def __init__(self, name: Optional[str] = None):
        """初始化节点。

        Args:
            name: 节点名称，默认使用类属性 name。
        """
        if name is not None:
            self.name = name
        self.logger = logging.getLogger(f"query.{self.name}")

    def __call__(self, state: QueryGraphState) -> dict:
        """节点执行入口，LangGraph 调用此方法。

        提供统一的日志输出和异常处理。
        节点失败时返回空字典（不中断流程），保证管道韧性。
        """
        self.logger.info("--- %s 开始 ---", self.name)
        try:
            result = self.process(state)
            self.logger.info("--- %s 完成 ---", self.name)
            return result
        except Exception as e:
            self.logger.error("%s 执行失败（跳过）: %s", self.name, e)
            # 崩溃兜底：把失败节点标记为降级（追加而非覆盖，保留已有标记）
            degraded = list(state.get("degraded", []))
            if self.name not in degraded:
                degraded.append(self.name)
            return {"degraded": degraded}

    @abstractmethod
    def process(self, state: QueryGraphState) -> dict:
        """节点核心处理逻辑，子类必须实现。

        Args:
            state: 当前图状态。

        Returns:
            需要更新的状态字段字典（部分更新）。
        """
        ...

    def log_step(self, step_name: str, message: str = ""):
        """记录步骤日志。"""
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


# ================================================================== #
#                     1. 商品名确认节点                               #
# ================================================================== #

class ItemNameConfirmNode(BaseNode):
    """商品名确认节点。

    Semantic Router 式两段路由：
    1. 向量召回——query 拼最近 2 条历史做确定性检索，按文档聚合取 top-3 候选；
    2. LLM 封闭确认——只在候选里选（或判闲聊/全库检索），不再面对全量文档名
       开放选择。
    召回异常时回退旧行为（注入全量文档列表让 LLM 选）。
    """

    # 文档级候选数（top-3）
    RECALL_TOP_K = 3
    # chunk 级检索条数（聚合前）
    RECALL_SEARCH_K = 15
    # 召回下限：COSINE 分数低于此值的文档不进入候选（经验值；实测跨领域
    # 文档相似度 ~0.08，同领域 ~0.6+，闲聊短句通常 <0.3）
    RECALL_MIN_SCORE = 0.30

    def __init__(self, name: str = "item_name_confirm"):
        super().__init__(name=name)

    def _recall_candidates(self, query: str, history: list):
        """向量召回候选文档：按 item_name 聚合 chunk 最高分，取 top-3。

        召回输入 = 最近 2 条历史 + 当前问题（解决「刚才说的蜂鸣档」这类
        指代问题的召回，不依赖 LLM 改写）。返回 (候选列表, 是否失败)：
        失败返回 ([], True) 供上层回退全量列表模式；正常无候选返回 ([], False)。
        """
        settings = get_settings()
        try:
            ctx_texts = [
                str(m.get("content", "")) for m in history[-2:] if m.get("content")
            ]
            recall_text = " ".join([*ctx_texts, query]).strip()
            vec = embed_one(recall_text)
            hits = milvus_search(
                settings.chunks_collection, vec, top_k=self.RECALL_SEARCH_K
            )
            best: dict[str, float] = {}
            for h in hits:
                name = str(h.get("item_name", "")).strip()
                score = float(h.get("score", 0.0))
                if name and score > best.get(name, -1.0):
                    best[name] = score
            ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
            candidates = [
                n for n, s in ranked if s >= self.RECALL_MIN_SCORE
            ][: self.RECALL_TOP_K]
            self.log_step(
                "recall",
                f"召回候选: {candidates}（分数: "
                f"{ {n: round(s, 3) for n, s in ranked[: self.RECALL_TOP_K]} }）",
            )
            return candidates, False
        except Exception as e:
            self.logger.warning("向量召回候选失败，回退全量列表模式: %s", e)
            return [], True

    def process(self, state: QueryGraphState) -> dict:
        query = state.get("original_query", "")
        history = state.get("history", [])

        # 拼接历史对话上下文
        history_text = ""
        if history:
            history_text = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('text', msg.get('content', ''))}"
                for msg in history[-6:]  # 取最近 6 条
            )

        # 候选策略三分支：封闭候选（常规）→ 无候选提示 → 召回失败回退全量列表
        candidates, recall_failed = self._recall_candidates(query, history)
        degraded = list(state.get("degraded", []))
        if recall_failed and "item_recall" not in degraded:
            degraded.append("item_recall")
        if candidates:
            items_line = (
                "候选文档（按向量相关性排序，item_names 只能从候选中选择，可多个）：\n"
                + "\n".join(f"{i}. {c}" for i, c in enumerate(candidates, 1))
                + "\n若候选都与问题无关但问题属于知识性问题，item_names 返回 []"
                "（系统会走全库检索）。\n\n"
            )
        elif not recall_failed:
            items_line = (
                "知识库中未召回到与问题明显相关的文档。\n"
                "若问题属于知识性问题，item_names 返回 []（系统会走全库检索）；"
                "若属于类型A则按类型A输出。\n\n"
            )
        else:
            # 召回失败兜底：注入全量文档列表（旧行为）
            try:
                from tools.milvus_utils import list_item_names
                known_items = list_item_names()
            except Exception as e:
                self.logger.warning("获取知识库文档列表失败: %s", e)
                known_items = []
            items_line = (
                f"知识库现有文档：{'、'.join(known_items)}。\n"
                "item_names 必须从上述列表中选择（可多个）；问题与列表中所有文档都无关时才属于类型A。\n\n"
                if known_items
                else ""
            )

        # 构建提示词，要求 LLM 返回 JSON
        # 路由规则：只有纯闲聊/问候才直接回答，一切知识库问题都必须走检索
        # 注意：示例值给真实回复（含 few-shot），避免小模型照抄元描述「你的简短回答」
        user_prompt = (
            "你是知识库问答助手的问题路由助手。判断用户问题类型，输出JSON（只输出JSON，不要其他文字）。\n\n"
            f"{items_line}"
            "类型A - 纯闲聊/问候/与知识库内容无关（如'你好'、'你是谁'、'讲个笑话'）：\n"
            '输出：{"item_names": [], "rewritten_query": "", "direct_answer": "<你回复用户的原话>"}\n'
            '示例：用户问"你好" → {"item_names": [], "rewritten_query": "", "direct_answer": "你好！请问有什么知识库相关的问题需要帮忙？"}\n\n'
            "类型B - 需要知识库回答的问题（设备操作、参数、故障，或文档中的故事剧情、\n"
            "论文内容、人物事件、概念解释等一切知识性问题）：\n"
            '输出：{"item_names": ["<相关文档名>"], "rewritten_query": "<补全上下文后的完整问题>", "direct_answer": ""}\n'
            '示例1：用户问"怎么测电压" → {"item_names": ["数字万用表"], "rewritten_query": "数字万用表如何测量电压？", "direct_answer": ""}\n'
            '示例2：用户问"主角最后怎么样了"（历史对话提到灰姑娘）→ {"item_names": ["灰姑娘"], "rewritten_query": "灰姑娘的主角最后怎么样了？", "direct_answer": ""}\n\n'
            "注意：direct_answer 必须是面向用户的真实回复内容；知识库问题必须走检索，"
            "direct_answer 留空字符串，即使你知道答案。\n\n"
            f"{'历史对话：' + chr(10) + history_text + chr(10) if history_text else ''}"
            f"用户问题：{query}"
        )

        messages = [
            {"role": "system", "content": "你是知识库问答系统的路由助手，负责区分闲聊与知识库问题。知识库问题必须检索知识库。"},
            {"role": "user", "content": user_prompt},
        ]

        # 调用 LLM：chat_json 内置 temperature=0 + response_format 约束解码
        settings = get_settings()
        result = chat_json(messages, model=settings.item_model)
        item_names = result.get("item_names", [])
        rewritten_query = result.get("rewritten_query", "")
        direct_answer = result.get("direct_answer", "")

        # 根据 LLM 返回决定后续流程
        if direct_answer:
            # LLM 直接回答了（闲聊等），设 answer 跳过检索
            self.log_step("direct_answer", f"直接回答: {direct_answer[:50]}...")
            return {"answer": direct_answer, "degraded": degraded}

        if item_names:
            # 成功提取产品名
            self.log_step("extracted", f"产品名: {item_names}")
            return {
                "item_names": item_names,
                "rewritten_query": rewritten_query or query,
                "degraded": degraded,
            }

        # 既没提取到产品名也没直接回答，继续检索（不过滤）
        self.log_step("no_item", "未识别到产品名，继续检索")
        return {"rewritten_query": query, "degraded": degraded}


# ================================================================== #
#                     2. 向量检索节点                                 #
# ================================================================== #

class SearchEmbeddingNode(BaseNode):
    """向量检索节点。

    用 embed_one 编码 query，然后 milvus search 检索 top_k=10，
    支持按 item_name 过滤。
    """

    SEARCH_TOP_K = 10

    def __init__(self, name: str = "search_embedding"):
        super().__init__(name=name)

    def process(self, state: QueryGraphState) -> dict:
        query = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state.get("item_names", [])

        self.log_step("step_1", f"查询向量化: {query}")

        # 编码查询
        query_vector = embed_one(query)

        # 按商品名过滤（取第一个商品名）
        item_name = item_names[0] if item_names else None

        # Milvus 向量检索
        settings = get_settings()
        self.log_step("step_2", f"向量检索 top_k={self.SEARCH_TOP_K}")
        hits = milvus_search(
            collection_name=settings.chunks_collection,
            query_vector=query_vector,
            top_k=self.SEARCH_TOP_K,
            item_name=item_name,
        )

        # 统一格式，添加 source 字段
        chunks = [
            {
                "content": hit.get("content", ""),
                "title": hit.get("title", ""),
                "item_name": hit.get("item_name", ""),
                "score": hit.get("score", 0.0),
                "source": "embedding",
            }
            for hit in hits
        ]

        self.log_step("step_3", f"检索完成，返回 {len(chunks)} 条结果")
        return {"embedding_chunks": chunks}


# ================================================================== #
#                     3. HyDE 检索节点                                #
# ================================================================== #

class SearchHydeNode(BaseNode):
    """HyDE 检索节点。

    用 LLM 生成假设答案，再用假设答案做 embed_one + milvus search，
    检索 top_k=5。
    """

    SEARCH_TOP_K = 5

    def __init__(self, name: str = "search_embedding_hyde"):
        super().__init__(name=name)

    def process(self, state: QueryGraphState) -> dict:
        query = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state.get("item_names", [])

        # Step 1: LLM 生成假设答案
        self.log_step("step_1", "生成假设答案")
        messages = [
            {"role": "system", "content": "你是一个技术文档专家，请根据问题写一段可能的答案，用于检索增强。"},
            {"role": "user", "content": f"问题：{query}\n\n请写一段可能的答案（100-200字）："},
        ]
        hyde_text = chat(messages)

        # Step 2: 用假设答案做向量检索
        self.log_step("step_2", "用假设答案向量化并检索")
        query_vector = embed_one(hyde_text)
        item_name = item_names[0] if item_names else None

        settings = get_settings()
        hits = milvus_search(
            collection_name=settings.chunks_collection,
            query_vector=query_vector,
            top_k=self.SEARCH_TOP_K,
            item_name=item_name,
        )

        # 统一格式
        chunks = [
            {
                "content": hit.get("content", ""),
                "title": hit.get("title", ""),
                "item_name": hit.get("item_name", ""),
                "score": hit.get("score", 0.0),
                "source": "hyde",
            }
            for hit in hits
        ]

        self.log_step("step_3", f"HyDE 检索完成，返回 {len(chunks)} 条结果")
        return {"hyde_chunks": chunks}


# ================================================================== #
#                     4. 知识图谱检索节点                             #
# ================================================================== #

class QueryKgNode(BaseNode):
    """知识图谱检索节点。

    用 chat_json 让 LLM 从 query 提取实体名，
    然后用 query_by_entity 查询关联关系，返回关联的 chunk 信息。
    """

    def __init__(self, name: str = "query_kg"):
        super().__init__(name=name)

    def process(self, state: QueryGraphState) -> dict:
        query = state.get("rewritten_query") or state.get("original_query", "")
        item_names = state.get("item_names", [])

        # Step 1: LLM 提取实体名
        self.log_step("step_1", "LLM 提取实体名")
        messages = [
            {"role": "system", "content": "你是一个实体识别助手，从用户问题中提取关键实体名称。"},
            {"role": "user", "content": (
                f"从以下问题中提取关键实体名称（产品名、部件名、功能名等）。\n"
                f"问题：{query}\n\n"
                '请返回JSON格式：{"entities": ["实体1", "实体2"]}'
            )},
        ]
        settings = get_settings()
        result = chat_json(messages, model=settings.kg_model)
        entities = result.get("entities", [])

        if not entities:
            self.log_step("step_2", "未提取到实体，跳过 KG 检索")
            return {"kg_chunks": [], "kg_entities": []}

        # item_name 锚点：产品名是该文档子图中必然存在的 Device 节点，
        # LLM 提取的实体可能只是别名/短名（如「空气炸锅」≠「AF-301 空气炸锅」，
        # 实体查询是精确匹配会落空），并入锚点保证 KG 路至少有设备级召回。
        item_name = item_names[0] if item_names else None
        if item_name and item_name not in entities:
            entities.append(item_name)

        self.log_step("step_2", f"提取到实体: {entities}")

        # Step 2: 逐个实体查询知识图谱
        kg_chunks: list[dict] = []
        # 读取已有降级标记（追加而非覆盖，LangGraph 状态是后值覆盖语义）
        degraded = list(state.get("degraded", []))

        for entity in entities:
            try:
                # 显式传 max_depth=2 支持多跳推理（如 设备->部件->操作步骤）
                triples = query_by_entity(entity, item_name=item_name, max_depth=2)
                for triple in triples:
                    content = f"{triple.get('source', '')} --[{triple.get('relation', '')}]--> {triple.get('target', '')}"
                    kg_chunks.append({
                        "content": content,
                        "title": entity,
                        "score": 1.0,
                        "source": "kg",
                    })
            except Exception as e:
                self.logger.warning("查询实体 %s 失败: %s", entity, e)
                # Neo4j 不可用等：显式标记 KG 路降级
                if "kg" not in degraded:
                    degraded.append("kg")

        self.log_step("step_3", f"KG 检索完成，返回 {len(kg_chunks)} 条结果")
        return {"kg_chunks": kg_chunks, "degraded": degraded, "kg_entities": entities}


# ================================================================== #
#                     5. BM25 稀疏检索节点                            #
# ================================================================== #

class BM25SearchNode(BaseNode):
    """BM25 稀疏检索节点。

    用 rank_bm25 库（纯 Python，无需 GPU）对知识库全量切片做关键词召回。
    BM25 是经典 IDF+TF 的概率检索模型，对型号、编号、专有名词等字面匹配强。
    实现：索引由导入期构建并 pickle 落盘（tools/bm25_utils），本节点查询期
    只查不建；pickle 缺失或语料失配时自动回源重建（自愈）。
    """

    def __init__(self, name: str = "bm25_search"):
        super().__init__(name=name)

    def process(self, state: QueryGraphState) -> dict:
        query = state.get("rewritten_query") or state.get("original_query", "")
        settings = get_settings()
        top_k = settings.embedding_search_limit  # 复用检索 top_k 配置

        try:
            from tools.bm25_utils import search as bm25_search
        except ImportError:
            self.log_step("skip", "rank_bm25 未安装，跳过 BM25 检索")
            return {"bm25_docs": []}

        degraded = list(state.get("degraded", []))
        try:
            # 索引由导入期构建并落盘（tools/bm25_utils），查询期只查不建；
            # pickle 缺失或与 Milvus 条数失配时由 bm25_utils 自愈重建
            results = bm25_search(query, top_k)
        except Exception as e:
            self.logger.warning("BM25 检索失败，标记降级: %s", e)
            if "bm25" not in degraded:
                degraded.append("bm25")
            return {"bm25_docs": [], "degraded": degraded}

        self.log_step("step_1", f"BM25 检索完成，返回 {len(results)} 条结果")
        return {"bm25_docs": results}


# ================================================================== #
#                     6. RRF 融合节点                                 #
# ================================================================== #

class RrfNode(BaseNode):
    """RRF 多路融合节点。

    对四路结果（embedding_chunks, hyde_chunks, kg_chunks, web_docs）
    做 RRF 融合。公式：score(d) = Σ 1/(k+rank_r(d))。
    按 score 降序取 top-10。
    """

    def __init__(self, name: str = "rrf"):
        super().__init__(name=name)

    def process(self, state: QueryGraphState) -> dict:
        settings = get_settings()
        k = settings.rrf_k  # RRF 平滑常数，默认 60
        max_results = settings.rrf_max_results  # 最大返回数，默认 10

        # 四路检索结果：向量检索 + HyDE + 知识图谱 + BM25稀疏检索
        channels: list[list[dict]] = [
            state.get("embedding_chunks", []),
            state.get("hyde_chunks", []),
            state.get("kg_chunks", []),
            state.get("bm25_docs", []),
        ]

        self.log_step("step_1", f"融合 {len(channels)} 路结果, k={k}")

        # 以 content 为 key 做去重和 RRF 打分
        scores: dict[str, dict] = {}
        for channel in channels:
            for rank, doc in enumerate(channel):
                content = doc.get("content", "")
                if not content:
                    continue
                # RRF 公式：1/(k + rank)，rank 从 0 开始所以 +1
                rrf_score = 1.0 / (k + rank + 1)
                if content not in scores:
                    scores[content] = {
                        "content": content,
                        "title": doc.get("title", ""),
                        "item_name": doc.get("item_name", ""),
                        "score": 0.0,
                        "source": doc.get("source", ""),
                    }
                scores[content]["score"] += rrf_score

        # 按 score 降序排序，取 top-N
        sorted_docs = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        rrf_chunks = sorted_docs[:max_results]

        self.log_step("step_2", f"RRF 融合完成，返回 {len(rrf_chunks)} 条结果")
        return {"rrf_chunks": rrf_chunks}


# ================================================================== #
#                     7. Reranker 精排节点（含断崖检测）              #
# ================================================================== #

class RerankerNode(BaseNode):
    """重排序节点：两阶段检索架构的精排层。

    工作原理：向量检索是双塔 Bi-Encoder（Q和D独立编码、互不可见），
    速度快但精度有限；Reranker 是 Cross-Encoder（Q和D拼接进模型，
    attention 层全交互），精度高但每对(Q,D)都要跑一次前向，慢。
    所以工程上分两阶段：向量召回粗筛 Top-30 → Reranker 精排 Top-5。

    本节点用 DashScope gte-rerank-v2 API（cross-encoder 重排模型，
    等价于 BGE-Reranker 但无需 GPU）。对 RRF 融合后的 Top-N 重新打分，
    再用断崖检测动态截断：相邻分数比 < 阈值视为分数断崖，截掉后续噪声。
    """

    # 断崖检测阈值：相邻分数比 < 此值视为分数断崖（典型 0.5-0.7）
    CLIFF_RATIO_THRESHOLD = 0.6
    # 最少保留条数：即使断崖很早，也保底喂给 LLM 这么多条
    MIN_KEEP = 2
    # 最多喂给 LLM 的条数：精排后上限
    MAX_KEEP = 6

    def __init__(self, name: str = "rerank"):
        super().__init__(name=name)

    @staticmethod
    def _cliff_cutoff(scored_docs: list[dict]) -> list[dict]:
        """断崖检测动态截断。

        算法：遍历相邻分数比，第一次出现 score[i+1]/score[i] < threshold
        时截断到 i+1。保底至少 MIN_KEEP 条，至多 MAX_KEEP 条。

        Args:
            scored_docs: 已按 score 降序的列表。

        Returns:
            截断后的列表。
        """
        if not scored_docs:
            return []
        keep = RerankerNode.MAX_KEEP
        for i in range(min(len(scored_docs), RerankerNode.MAX_KEEP) - 1):
            cur = scored_docs[i].get("score", 0)
            nxt = scored_docs[i + 1].get("score", 0)
            if cur <= 0:
                continue
            if nxt / cur < RerankerNode.CLIFF_RATIO_THRESHOLD:
                keep = i + 1
                break
        keep = max(keep, min(RerankerNode.MIN_KEEP, len(scored_docs)))
        return scored_docs[:keep]

    def process(self, state: QueryGraphState) -> dict:
        rrf_chunks = state.get("rrf_chunks", [])
        if not rrf_chunks:
            self.log_step("skip", "RRF 结果为空，跳过重排")
            return {"reranked_docs": []}

        query = state.get("rewritten_query") or state.get("original_query", "")
        settings = get_settings()
        rerank_model = getattr(settings, "rerank_model", "gte-rerank-v2")

        self.log_step("step_1", f"对 {len(rrf_chunks)} 条结果做 cross-encoder 精排")

        try:
            # DashScope rerank 接口走原生 HTTP（不走 OpenAI 兼容层）
            # POST https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
            import urllib.request

            url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
            payload = {
                "model": rerank_model,
                "input": {
                    "query": query,
                    "documents": [c.get("content", "") for c in rrf_chunks],
                },
                "parameters": {
                    "return_documents": False,  # 只返回分数和索引，不返回原文
                    "top_n": len(rrf_chunks),
                },
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            # DashScope rerank 返回结构: {output: {results: [{index, relevance_score}, ...]}}
            results = data.get("output", {}).get("results", [])
            scored = []
            for r in results:
                idx = r.get("index", 0)
                if 0 <= idx < len(rrf_chunks):
                    chunk = dict(rrf_chunks[idx])
                    chunk["rerank_score"] = float(r.get("relevance_score", 0))
                    scored.append(chunk)
            scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        except Exception as e:
            self.logger.warning("Rerank API 调用失败，回退 RRF 排序: %s", e)
            degraded = list(state.get("degraded", []))
            if "rerank" not in degraded:
                degraded.append("rerank")
            return {"reranked_docs": rrf_chunks[: self.MAX_KEEP], "degraded": degraded}

        if not scored:
            self.log_step("skip", "Rerank 返回空，回退 RRF 结果")
            degraded = list(state.get("degraded", []))
            if "rerank" not in degraded:
                degraded.append("rerank")
            return {"reranked_docs": rrf_chunks[: self.MAX_KEEP], "degraded": degraded}

        # 断崖检测动态截断
        before = len(scored)
        kept = self._cliff_cutoff(scored)
        self.log_step(
            "step_2",
            f"断崖检测：{before} -> {len(kept)} 条（阈值 {self.CLIFF_RATIO_THRESHOLD}）",
        )
        return {"reranked_docs": kept}


# ================================================================== #
#                     8. 答案生成输出节点                             #
# ================================================================== #

class AnswerOutputNode(BaseNode):
    """答案生成输出节点。

    拼接 prompt：系统提示词 + reranked_docs（Reranker 精排+断崖截断后）
    的内容作为上下文 + 历史对话 + 用户问题。调 chat_stream 生成答案。
    若 rerank 阶段被跳过或失败，回退到 rrf_chunks。
    """

    # 系统提示词
    SYSTEM_PROMPT = (
        "你是一个专业的知识库问答助手。请根据以下知识库内容回答用户问题。\n"
        "如果知识库内容不足以回答问题，请如实说明。\n"
        "回答要准确、简洁、有条理。"
    )

    def __init__(self, name: str = "answer_output"):
        super().__init__(name=name)

    def process(self, state: QueryGraphState) -> dict:
        # 如果已有答案（如闲聊直接回复），直接返回
        if state.get("answer"):
            self.log_step("skip", "已有答案，跳过生成")
            return {}

        query = state.get("rewritten_query") or state.get("original_query", "")
        # 优先用 Reranker 精排+断崖截断后的结果，无则回退 RRF 结果
        context_docs = state.get("reranked_docs") or state.get("rrf_chunks", [])
        history = state.get("history", [])

        # 构建上下文
        context_parts = []
        for i, chunk in enumerate(context_docs, 1):
            content = chunk.get("content", "")
            if content:
                context_parts.append(f"[{i}] {content}")
        context = "\n\n".join(context_parts) if context_parts else "（无检索结果）"

        # 构建系统提示词
        system_prompt = f"{self.SYSTEM_PROMPT}\n\n知识库内容：\n{context}"

        # 构建 messages
        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        # 添加历史对话
        for msg in history[-6:]:  # 取最近 6 条
            role = msg.get("role", "user")
            text = msg.get("text", msg.get("content", ""))
            if text:
                messages.append({"role": role, "content": text})

        # 添加当前问题
        messages.append({"role": "user", "content": query})

        self.log_step("step_1", f"生成答案，上下文 {len(context_docs)} 条")

        # 生成答案
        is_stream = state.get("is_stream", False)
        if is_stream:
            # 流式输出：收集所有 token，并记录首字延迟 TTFT（秒）——
            # 从开始调用 LLM 到收到首个 token 的耗时，随审计落盘供
            # 端到端性能评估（P50/P95）使用
            t0_gen = time.perf_counter()
            t_first: float | None = None
            parts: list[str] = []
            for token in chat_stream(messages):
                if t_first is None:
                    t_first = time.perf_counter()
                parts.append(token)
            answer = "".join(parts)
            ttft_sec = (
                round(time.perf_counter() - t0_gen, 3)
                if t_first is not None
                else None
            )
        else:
            # 非流式：直接调用 chat
            answer = chat(messages)
            ttft_sec = None

        self.log_step("step_2", f"答案生成完成，长度 {len(answer)} 字")
        return {"answer": answer, "prompt": system_prompt, "ttft_sec": ttft_sec}


# ================================================================== #
#                        辅助函数                                     #
# ================================================================== #
