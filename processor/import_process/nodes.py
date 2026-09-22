"""导入流程节点定义。

每个节点继承自 BaseNode，实现 process 方法处理图状态。
节点执行顺序由 main_graph.py 编排。

运行时 cwd 为 knowledge/，因此使用绝对导入：
    from core.config import get_settings
    from tools.llm_utils import chat, chat_json
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from pathlib import Path

from core.config import get_settings
from tools.llm_utils import chat, chat_json
from tools.embedding_utils import embed
from tools.milvus_utils import delete_item, ensure_collection, insert_chunks
from tools.neo4j_utils import create_entities_and_relations, clear_all

logger = logging.getLogger(__name__)

# graph_data.json 是跨导入任务的共享文件（读-改-写非原子），
# 并发导入两份文档时必须串行化，否则后写者会覆盖掉先写者的增量
_GRAPH_JSON_LOCK = threading.Lock()


class BaseNode:
    """节点基类。

    子类通过实现 process 方法完成具体逻辑，
    实例可直接作为 LangGraph 节点被调用。
    可选容错模式：失败时记录日志并跳过，不中断整个导入流程。
    """

    name: str = "base"
    # 容错节点名单：这些节点失败时不中断流程（如 Neo4j 未启动时跳过图谱构建）
    FAULT_TOLERANT = False

    def __init__(self, fault_tolerant: bool = None):
        if fault_tolerant is not None:
            self.FAULT_TOLERANT = fault_tolerant

    def __call__(self, state):
        if not self.FAULT_TOLERANT:
            return self.process(state)
        try:
            return self.process(state)
        except Exception as e:
            logger.error("节点 %s 执行失败（容错跳过）: %s", self.name, e)
            return state

    def process(self, state):
        return state


class EntryNode(BaseNode):
    """入口节点：检测文件后缀，设置 is_pdf 和 file_title。"""

    name = "entry"

    def process(self, state):
        file_path = state.get("file_path", "")
        suffix = Path(file_path).suffix.lower()
        state["is_pdf"] = (suffix == ".pdf")
        state["file_title"] = Path(file_path).stem
        logger.info("entry: file=%s, is_pdf=%s", file_path, state["is_pdf"])
        return state


class PdfToMdNode(BaseNode):
    """文档读取节点。

    - PDF：尝试用 pdfplumber / PyPDF2 / pymupdf 读取为纯文本；
    - Markdown / 其他文本：直接读取文件内容到 md_content。
    若 PDF 解析库均未安装，则跳过返回空文本。
    """

    name = "pdf_to_md"

    def process(self, state):
        file_path = state.get("file_path", "")
        if state.get("is_pdf"):
            state["md_content"] = self._pdf_to_text(file_path)
        else:
            # Markdown 或其他文本文件，直接读取
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    state["md_content"] = f.read()
            except Exception as e:  # 读取失败时置空，不中断流程
                logger.warning("pdf_to_md: 读取文件失败 %s: %s", file_path, e)
                state["md_content"] = ""
        logger.info("pdf_to_md: md_content 长度=%d", len(state.get("md_content", "")))
        return state

    @staticmethod
    def _pdf_to_text(file_path: str) -> str:
        """尝试依次使用 pdfplumber、PyPDF2、pymupdf 提取 PDF 文本。"""
        # 1. 优先 pdfplumber
        try:
            import pdfplumber

            texts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    texts.append(page.extract_text() or "")
            return "\n\n".join(texts)
        except ImportError:
            pass

        # 2. 其次 PyPDF2
        try:
            import PyPDF2  # noqa: N814

            texts = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    texts.append(page.extract_text() or "")
            return "\n\n".join(texts)
        except ImportError:
            pass

        # 3. 最后 pymupdf (fitz)
        try:
            import fitz  # type: ignore

            doc = fitz.open(file_path)
            texts = []
            for page in doc:
                texts.append(page.get_text() or "")
            doc.close()
            return "\n\n".join(texts)
        except ImportError:
            pass

        # 4. 均未安装，跳过返回空文本
        logger.warning("pdf_to_md: 未安装任何 PDF 解析库（pdfplumber/PyPDF2/pymupdf），跳过")
        return ""


class DocumentSplitNode(BaseNode):
    """文档切分节点。

    - 按 ``##`` 标题分段（识别代码围栏，避免误切）；
    - 超长段落按 max_content_length 滑窗切分（带重叠）；
    - 过短段落合并到相邻切片。
    输出 chunks 列表，每个 chunk 含 content、title、chunk_id。
    """

    name = "document_split"

    # 代码围栏行（```开头）用于切换是否在代码块内
    _FENCE_RE = re.compile(r"^\s*```")
    # 二级标题行：## 标题
    _H2_RE = re.compile(r"^\s*##\s+.+")
    # 滑窗重叠长度
    _OVERLAP = 200

    def process(self, state):
        settings = get_settings()
        max_len = settings.max_content_length
        min_len = settings.min_content_length

        md_content = state.get("md_content", "")
        sections = self._split_by_heading(md_content)

        chunks: list[dict] = []
        for title, content in sections:
            content = content.strip()
            if not content:
                continue
            if len(content) > max_len:
                # 超长段落滑窗切分
                for piece in self._sliding_window(content, max_len, self._OVERLAP):
                    chunks.append({"content": piece, "title": title, "chunk_id": 0})
            else:
                chunks.append({"content": content, "title": title, "chunk_id": 0})

        # 短段落合并
        chunks = self._merge_short_chunks(chunks, min_len, max_len)

        # 重新编号 chunk_id（Milvus 主键为 INT64）
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = i

        state["chunks"] = chunks
        logger.info("document_split: 切片数量=%d", len(chunks))
        return state

    @classmethod
    def _split_by_heading(cls, md_content: str) -> list[tuple[str, str]]:
        """按 ## 标题切分为 (title, content) 列表。

        代码围栏内的 ``##`` 不视为标题。标题之前的正文归入空标题段。
        """
        sections: list[tuple[str, str]] = []
        current_title = ""
        current_lines: list[str] = []
        in_code_fence = False

        for line in md_content.splitlines():
            # 代码围栏开关
            if cls._FENCE_RE.match(line):
                in_code_fence = not in_code_fence
                current_lines.append(line)
                continue
            # 非代码块内的 ## 标题：开启新段
            if not in_code_fence and cls._H2_RE.match(line):
                sections.append((current_title, "\n".join(current_lines)))
                current_title = line.lstrip()[2:].strip()  # 去掉 "## " 前缀
                current_lines = []
            else:
                current_lines.append(line)

        # 末尾段
        sections.append((current_title, "\n".join(current_lines)))
        return sections

    @staticmethod
    def _sliding_window(text: str, max_len: int, overlap: int) -> list[str]:
        """滑窗切分，超限时回退到句子边界，替代直接腰斩。

        切分长度超限时不直接截断，而是在窗口内从后往前找最后一个
        句子边界（「。」→「！」→「？」→换行），把边界回退到句子末尾，
        避免把一句话劈成两半破坏检索语义。

        两级保护：
        - 边界回退后切片长度不足 max_len 的 50% 时放弃回退（说明该段
          是无句号的密集文本如代码/表格，硬找边界会产出碎片切片）；
        - 找不到任何边界符号时按原边界截断。
        """
        if overlap >= max_len:
            overlap = max_len // 5
        pieces: list[str] = []
        start = 0
        total = len(text)
        while start < total:
            end = min(start + max_len, total)
            piece = text[start:end]
            if end < total:
                # 超限段：回退寻找最后一个句子边界
                boundary = max(
                    piece.rfind("。"),
                    piece.rfind("！"),
                    piece.rfind("？"),
                    piece.rfind("\n"),
                )
                if boundary >= int(max_len * 0.5):
                    end = start + boundary + 1  # 边界含在当前切片内
                    piece = text[start:end]
            pieces.append(piece)
            if end >= total:
                break  # 已到文末，避免尾部重叠区被反复切碎
            # 步长 = 实际切片长度 - 重叠，保证相邻切片仍有 overlap 重合
            start += max(len(piece) - overlap, 1)
        return pieces

    @staticmethod
    def _merge_short_chunks(
        chunks: list[dict], min_len: int, max_len: int
    ) -> list[dict]:
        """将过短的切片合并到上一个切片（合并后不超过 max_len）。"""
        if not chunks:
            return chunks
        merged: list[dict] = [dict(chunks[0])]
        for chunk in chunks[1:]:
            prev = merged[-1]
            cur_content = chunk.get("content", "")
            prev_content = prev.get("content", "")
            if (
                len(cur_content) < min_len
                and len(prev_content) + len(cur_content) + 1 <= max_len
            ):
                prev["content"] = prev_content + "\n" + cur_content
            else:
                merged.append(dict(chunk))
        return merged


class ItemNameRecognitionNode(BaseNode):
    """商品名识别节点。

    取前 3 个 chunk 的内容作为输入，调用 LLM 提取产品 / 设备名称。
    使用 chat_json 约束输出 + 代码层单行清洗：LLM 偶尔无视 prompt 吐多行
    （实测简历文档抽出「jiuan-lite\nGraphRAG\nLangGraph\n...」技能清单），
    JSON 约束 + 截断清洗双保险，异常时降级文件标题。
    """

    name = "item_name_recognition"
    # 取前 N 个 chunk 作为识别输入
    _preview_k = 3
    # 名称最大长度（超长视为异常输出，降级文件标题）
    _max_name_len = 60

    def process(self, state):
        chunks = state.get("chunks", [])
        if not chunks:
            return state

        preview = "\n\n".join(
            c.get("content", "") for c in chunks[: self._preview_k]
        )
        file_title = str(state.get("file_title", "") or "").strip()
        title_line = f"（文件标题：{file_title}，可作为参考）" if file_title else ""
        messages = [
            {
                "role": "user",
                "content": (
                    "请为以下文档提取一个简短的文档名称（如产品名、设备名、"
                    "论文标题、人物名等），单个名称，不超过 20 个字。\n"
                    f"{title_line}\n"
                    '只输出JSON：{"item_name": "名称"}\n\n'
                    f"{preview}"
                ),
            }
        ]
        try:
            result = chat_json(messages, model=get_settings().item_model)
            raw = str(result.get("item_name", "")).strip()
            # 单行清洗：取首个非空行、截断；空/超长视为异常，降级文件标题
            name = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
            if not name or len(name) > self._max_name_len:
                raise ValueError(f"异常名称输出: {raw!r}")
        except Exception as e:
            # LLM 调用失败或输出异常时降级使用文件标题
            logger.warning("item_name_recognition: 降级使用文件标题: %s", e)
            name = file_title

        state["item_name"] = name
        # 同步记录到 item_names 列表
        item_names = state.get("item_names", [])
        if name and name not in item_names:
            item_names.append(name)
        state["item_names"] = item_names
        logger.info("item_name_recognition: item_name=%s", name)
        return state


class BgeEmbeddingNode(BaseNode):
    """向量化节点。

    用 embed() 批量生成 chunks 的向量，给每个 chunk 加 dense_vector 字段。
    批量大小 5。
    """

    name = "bge_embedding"
    _batch_size = 5

    def process(self, state):
        chunks = state.get("chunks", [])
        if not chunks:
            return state

        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]
            texts = [c.get("content", "") for c in batch]
            try:
                vectors = embed(texts)
            except Exception as e:
                logger.error("bge_embedding: 批次 %d 向量化失败: %s", i, e)
                continue
            for j, vec in enumerate(vectors):
                batch[j]["dense_vector"] = vec

        logger.info("bge_embedding: 完成向量化，切片数=%d", len(chunks))
        return state


class ImportMilvusNode(BaseNode):
    """向量入库节点。

    确保 Milvus 集合存在，并把带向量的 chunks 写入。
    """

    name = "import_milvus"

    @staticmethod
    def _make_chunk_id(item_name: str, index: int, content: str) -> int:
        """生成全局唯一且确定性的 chunk_id（INT64 主键）。

        以 item_name+序号+内容 的 MD5 前 8 字节作主键：
        - 不同文档切片主键互不冲突（修复按 0 重新编号导致跨文档主键覆盖的问题）；
        - 同一文档重复导入生成相同主键，实现幂等覆盖。
        """
        digest = hashlib.md5(f"{item_name}::{index}::{content}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=True)

    def process(self, state):
        chunks = state.get("chunks", [])
        if not chunks:
            return state

        settings = get_settings()
        item_name = state.get("item_name", "")

        # 给每个 chunk 补充 item_name，并重写全局唯一主键
        for i, chunk in enumerate(chunks):
            chunk["item_name"] = item_name
            chunk["chunk_id"] = self._make_chunk_id(item_name, i, chunk.get("content", ""))

        # 同名/改名重导：先清旧向量（Milvus insert 同主键是追加不是覆盖）。
        # LLM 抽取的 item_name 不稳定，同一文档可能改名（如「RX-27 路由器」
        # →「RX-27 双频路由器」），历史名也要一并清理，否则旧切片残留。
        cleanup_names = list(dict.fromkeys(
            [item_name, *state.get("previous_item_names", [])]
        ))
        for name in cleanup_names:
            if not name:
                continue
            deleted = delete_item(settings.chunks_collection, name)
            if deleted:
                logger.info("import_milvus: 清理旧向量 %d 条 (%s)", deleted, name)

        # 确保集合存在并插入
        ensure_collection(settings.chunks_collection, dim=settings.embedding_dim)
        insert_chunks(settings.chunks_collection, chunks)
        logger.info("import_milvus: 插入 %d 条到 %s", len(chunks), settings.chunks_collection)

        # 导入成功后立即重建 BM25 持久索引（导入期构建、查询期只查）。
        # rebuild_index 内部吞掉所有异常只告警：索引失败不阻断导入，
        # 查询侧 get_index 会按语料条数失配自愈重建。
        from tools.bm25_utils import rebuild_index
        idx = rebuild_index()
        if idx:
            logger.info("import_milvus: BM25 索引已重建（%d 条切片）", idx["n_chunks"])
        return state


class KnowledgeGraphNode(BaseNode):
    """知识图谱构建节点。

    先对整篇文档做一次类型分类（manual/paper/narrative/resume/other），
    再按类型套用固定 Schema 从每个 chunk 抽取实体与关系，合并后做实体对齐
    与关系方向校验（写反的边自动翻转/丢弃），写入 Neo4j。

    固定 Schema（实体类型 → 关系类型）：
      manual    设备手册：Device/Part/Operation/Step/Warning/Condition/Tool
                → HAS_OPERATION/HAS_PART/HAS_STEP/USES_TOOL/HAS_WARNING/
                  NEXT_STEP/AFFECTS/REQUIRES
      paper     学术论文：Paper/Method/Dataset/Metric/Task/Finding
                → PROPOSES/USES_DATASET/EVALUATES_ON/OUTPERFORMS/
                  COMPARED_WITH/ADDRESSES
      narrative 故事剧本：Character/Location/Event/Item/Theme
                → APPEARS_IN/LOCATED_AT/PARTICIPATES_IN/CAUSES/FOLLOWS/OWNS
      resume    个人简历：Person/School/Company/Project/Skill/Course
                → EDUCATED_AT/WORKS_AT/LEADS/USES_TECH/LEARNED
      other     无法归类：回退为让 LLM 自定义 3~8 类
    """

    name = "knowledge_graph"
    # 容错节点：Neo4j 未启动时跳过图谱构建，不阻断导入流程
    FAULT_TOLERANT = True

    # 固定 Schema 表：doc_type -> (实体类型列表, 关系类型列表)
    SCHEMAS = {
        "manual": (
            ["Device", "Part", "Operation", "Step", "Warning", "Condition", "Tool"],
            ["HAS_OPERATION", "HAS_PART", "HAS_STEP", "USES_TOOL",
             "HAS_WARNING", "NEXT_STEP", "AFFECTS", "REQUIRES"],
        ),
        "paper": (
            ["Paper", "Method", "Dataset", "Metric", "Task", "Finding"],
            ["PROPOSES", "USES_DATASET", "EVALUATES_ON", "OUTPERFORMS",
             "COMPARED_WITH", "ADDRESSES"],
        ),
        "narrative": (
            ["Character", "Location", "Event", "Item", "Theme"],
            ["APPEARS_IN", "LOCATED_AT", "PARTICIPATES_IN", "CAUSES",
             "FOLLOWS", "OWNS"],
        ),
        "resume": (
            ["Person", "School", "Company", "Project", "Skill", "Course"],
            ["EDUCATED_AT", "WORKS_AT", "LEADS", "USES_TECH", "LEARNED"],
        ),
    }

    # 关系方向约束：doc_type -> {关系类型: (允许的头实体类型, 允许的尾实体类型)}
    # 方向写反是 LLM 抽取的高频错误（如 School-WORKS_AT->Person）。校验策略：
    # 头尾都匹配保留；仅反方向匹配则自动翻转；两者都不匹配丢弃；
    # 端点实体缺类型时保守保留（宁可不裁决，不误删）。
    RELATION_DIRECTIONS = {
        "manual": {
            "HAS_OPERATION": ({"Device"}, {"Operation"}),
            "HAS_PART": ({"Device", "Operation", "Step"}, {"Part", "Tool"}),
            "HAS_STEP": (
                {"Operation", "Step"},
                {"Step", "Condition", "Part", "Operation"},
            ),
            "USES_TOOL": ({"Operation", "Step"}, {"Tool", "Part"}),
            "HAS_WARNING": (
                {"Device", "Operation", "Step", "Condition"},
                {"Warning"},
            ),
            "NEXT_STEP": ({"Step", "Operation"}, {"Step", "Operation"}),
            "AFFECTS": ({"Warning", "Operation", "Step"}, {"Device", "Part", "Condition"}),
            "REQUIRES": (
                {"Device", "Operation", "Step"},
                {"Tool", "Condition", "Part", "Operation"},
            ),
        },
        "paper": {
            "PROPOSES": ({"Paper"}, {"Method", "Finding", "Task"}),
            "USES_DATASET": ({"Paper", "Method"}, {"Dataset"}),
            "EVALUATES_ON": ({"Paper", "Method"}, {"Dataset", "Task", "Metric"}),
            "OUTPERFORMS": ({"Method", "Paper"}, {"Method", "Paper"}),
            "COMPARED_WITH": ({"Method", "Paper"}, {"Method", "Paper"}),
            "ADDRESSES": ({"Paper", "Method"}, {"Task"}),
        },
        "narrative": {
            "APPEARS_IN": ({"Character", "Item"}, {"Location", "Event", "Item"}),
            "LOCATED_AT": ({"Character", "Item", "Event"}, {"Location"}),
            "PARTICIPATES_IN": ({"Character"}, {"Event"}),
            "CAUSES": ({"Event", "Character", "Item"}, {"Event", "Theme"}),
            "FOLLOWS": ({"Event"}, {"Event"}),
            "OWNS": ({"Character"}, {"Item"}),
        },
        "resume": {
            "EDUCATED_AT": ({"Person"}, {"School"}),
            "WORKS_AT": ({"Person"}, {"School", "Company"}),
            "LEADS": ({"Person"}, {"Project"}),
            "USES_TECH": ({"Person", "Project"}, {"Skill"}),
            "LEARNED": ({"Person"}, {"Course", "Skill"}),
        },
    }

    # 实体对齐两段式：
    # 1) 余弦相似度 >= ALIGN_THRESHOLD 的高置信对直接合并；
    # 2) 相似度处于 [CANDIDATE_LOW, ALIGN_THRESHOLD) 的模糊对 + 子串包含对
    #    交给 LLM 裁决（DERA 式），确认后才合并。
    # 依据实测：text-embedding-v4 中文短名相似度整体偏低且区分度差
    # （「王子/年轻的王子」仅 0.61，而「王子/国王」反达 0.75），
    # 单靠阈值会错并，故模糊带必须 LLM 确认。两个阈值均为经验值（非论文出处）。
    ALIGN_THRESHOLD = 0.85
    CANDIDATE_LOW = 0.55
    # GraphRAG 式 Gleaning 补采轮数（原论文默认 1 轮，补采返回空即提前停止）
    GLEANING_ROUNDS = 1
    # 向量化批量大小（与 BgeEmbeddingNode 保持一致，DashScope 单请求上限 10）
    _EMBED_BATCH = 5

    def process(self, state):
        chunks = state.get("chunks", [])
        item_name = state.get("item_name", "")

        all_entities: list[dict] = []
        all_relations: list[dict] = []

        # 整篇文档只分类一次：用首个非空切片开头采样，按类型套固定 Schema
        doc_type = "other"
        for chunk in chunks:
            if chunk.get("content"):
                doc_type = self._classify_doc(chunk["content"][:800])
                break
        entity_types, relation_types = self.SCHEMAS.get(doc_type, ([], []))
        logger.info("knowledge_graph: 文档「%s」分类为 %s", item_name, doc_type)

        for chunk in chunks:
            content = chunk.get("content", "")
            if not content:
                continue
            try:
                result = self._extract(content, entity_types, relation_types)
            except Exception as e:
                logger.warning("knowledge_graph: 切片 %s 抽取失败: %s", chunk.get("chunk_id"), e)
                continue
            all_entities.extend(result.get("entities", []))
            all_relations.extend(result.get("relations", []))

        # 固定 Schema 硬校验：prompt 是软约束，LLM 偶尔越出（实测补采轮输出过
        # 列表外的 THEME_OF），此处按白名单过滤，保证图谱类型严格合规
        if entity_types:
            allowed_e = set(entity_types)
            allowed_r = set(relation_types)
            dropped_r = len(all_relations)
            all_entities = [
                e for e in all_entities
                if any(t in allowed_e for t in e.get("types", []))
            ]
            all_relations = [
                r for r in all_relations
                if str(r.get("type", "")).strip().upper() in allowed_r
            ]
            dropped_r -= len(all_relations)
            if dropped_r:
                logger.info("knowledge_graph: Schema 硬校验丢弃越界关系 %d 条", dropped_r)

        # 实体对齐：跨 chunk 合并同义别名（embedding 相似度聚类），再统一去重。
        # 失败时跳过合并，退回精确名去重，不阻断导入。
        if all_entities:
            try:
                all_entities, all_relations = self._align_entities(all_entities, all_relations)
            except Exception as e:
                logger.warning("knowledge_graph: 实体对齐失败，跳过合并: %s", e)

            # 关系方向校验/修复：写反的边自动翻转，语义不通的丢弃
            try:
                all_relations = self._validate_directions(
                    doc_type, all_entities, all_relations
                )
            except Exception as e:
                logger.warning("knowledge_graph: 关系方向校验失败，跳过: %s", e)

        # 锚点保障：查询侧 KG 检索以 item_name 为锚点（精确名 → 模糊包含回退），
        # 但 LLM 抽取不保证生成产品实体（实测小文档只抽出「电源适配器/网线/
        # 回形针」等配件实体，没有产品节点），锚点无处可挂导致 kg=0。
        self._ensure_anchor(all_entities, all_relations, item_name)

        # 持久化到本地 JSON（供可视化，不依赖 Neo4j）
        if all_entities or all_relations:
            self._save_graph_json(
                all_entities, all_relations, item_name,
                previous_item_names=state.get("previous_item_names", []),
            )

        # 写入 Neo4j（create_entities_and_relations 内部用 MERGE，幂等）。
        # 先清历史名 + 本次名的旧图数据：与 graph_data.json 的「替换」语义
        # 对齐，避免重导后旧关系残留、改名后旧子图悬空。
        if all_entities or all_relations:
            cleanup_names = list(dict.fromkeys(
                [item_name, *state.get("previous_item_names", [])]
            ))
            for name in cleanup_names:
                if not name:
                    continue
                try:
                    clear_all(name)
                except Exception as e:
                    logger.warning("knowledge_graph: 清理旧图数据失败 (%s): %s", name, e)
            create_entities_and_relations(all_entities, all_relations, item_name)
        logger.info(
            "knowledge_graph: 写入完成，实体=%d，关系=%d",
            len(all_entities), len(all_relations),
        )
        return state

    @staticmethod
    def _ensure_anchor(entities: list[dict], relations: list[dict],
                       item_name: str) -> None:
        """保证子图含名为 item_name 的锚点实体（就地修改两个列表）。

        查询侧以 item_name 为锚点检索子图，但 LLM 抽取不稳定——
        实测小文档只抽出配件/环境实体，产品节点缺失，锚点落空 kg=0。
        缺失时补建锚点实体，并把它连到度数最高的若干实体：孤立节点
        查不出任何三元组，有边才能作为检索入口。边类型按目标实体
        类型选择（manual 白名单内），其他文档类型用通用 ASSOCIATED_WITH。
        """
        if not item_name:
            return
        names = {str(e.get("name", "")).strip() for e in entities}
        if item_name in names:
            return

        # 度数统计：被越多关系引用的实体越核心，优先挂边
        degree: dict[str, int] = {}
        for r in relations:
            for key in ("source", "target"):
                n = str(r.get(key, "")).strip()
                if n:
                    degree[n] = degree.get(n, 0) + 1
        types_by_name = {
            str(e.get("name", "")).strip(): set(e.get("types", []))
            for e in entities
        }
        top = [n for n, _ in sorted(degree.items(), key=lambda kv: -kv[1])[:8]]
        # 完全没有关系时退化为按实体列表顺序（至少保证锚点有边）
        top = top or [str(e.get("name", "")).strip() for e in entities[:8]]

        entities.append({"name": item_name, "types": ["Device"]})
        for name in top:
            if not name or name == item_name:
                continue
            rtype = KnowledgeGraphNode._anchor_rel_type(
                types_by_name.get(name, set()))
            relations.append(
                {"source": item_name, "target": name, "type": rtype})
        logger.info(
            "knowledge_graph: 抽取结果缺少产品实体，补建锚点「%s」并挂边 %d 条",
            item_name, min(len(top), 8),
        )

    @staticmethod
    def _anchor_rel_type(types: set[str]) -> str:
        """按目标实体类型挑一条语义合适的锚点边类型。"""
        if types & {"Part", "Tool"}:
            return "HAS_PART"
        if "Operation" in types:
            return "HAS_OPERATION"
        if "Warning" in types:
            return "HAS_WARNING"
        return "ASSOCIATED_WITH"

    def _align_entities(self, entities: list[dict], relations: list[dict]):
        """实体对齐：合并指向同一对象的不同实体名写法。

        动机：各 chunk 独立抽取，同一对象常以不同写法出现（如「王子」/
        「年轻的王子」），精确名去重无法合并。两段式做法：
        1. 实体名批量向量化，生成合并候选：
           - 相似度 >= ALIGN_THRESHOLD 的高置信对直接合并；
           - 相似度处于模糊带 + 子串包含对，交 LLM 裁决（DERA 式）后合并；
        2. 并查集合并，规范名 = 出现频次最高（平票取更长者）；
        3. 实体 types 取并集，关系端点重写为规范名，去自环、去重。

        返回 (对齐后实体列表, 对齐后关系列表)。
        """
        import numpy as np

        from tools.embedding_utils import embed

        # 统计频次（同一名字在不同 chunk 可能被抽出多次）
        freq: dict[str, int] = {}
        for e in entities:
            name = str(e.get("name", "")).strip()
            if name:
                freq[name] = freq.get(name, 0) + 1
        names = list(freq)
        if len(names) < 2:
            return entities, relations

        # 分批向量化（DashScope 单请求有批量上限，与导入侧批量大小一致）
        vectors: list[list[float]] = []
        for i in range(0, len(names), self._EMBED_BATCH):
            vectors.extend(embed(names[i : i + self._EMBED_BATCH]))

        # 归一化后一次算全两两余弦相似度
        mat = np.asarray(vectors, dtype=np.float32)
        mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        sims = mat @ mat.T

        # 两段式候选生成
        high_pairs: list[tuple[int, int]] = []      # 高置信，直接合并
        candidates: list[tuple[int, int]] = []      # 模糊带/子串，需 LLM 裁决
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                s = float(sims[i, j])
                if s >= self.ALIGN_THRESHOLD:
                    high_pairs.append((i, j))
                elif s >= self.CANDIDATE_LOW:
                    candidates.append((i, j))
                elif (
                    len(names[i]) >= 2 and len(names[j]) >= 2
                    and (names[i] in names[j] or names[j] in names[i])
                ):
                    # 子串包含（如「年轻的王子」含「王子」）也值得裁决
                    candidates.append((i, j))

        pairs_to_merge = list(high_pairs)
        if candidates:
            confirmed = self._judge_pairs(names, candidates)
            logger.info(
                "knowledge_graph: LLM 裁决候选对 %d 个，确认同体 %d 个",
                len(candidates), len(confirmed),
            )
            pairs_to_merge.extend(confirmed)

        # 并查集合并
        parent = list(range(len(names)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in pairs_to_merge:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        # 组内选规范名：频次最高，平票取更长者
        groups: dict[int, list[int]] = {}
        for i in range(len(names)):
            groups.setdefault(find(i), []).append(i)
        alias_map: dict[str, str] = {}
        for members in groups.values():
            canonical = max(members, key=lambda k: (freq[names[k]], len(names[k])))
            for k in members:
                alias_map[names[k]] = names[canonical]

        merged_aliases = {a: c for a, c in alias_map.items() if a != c}
        if not merged_aliases:
            return entities, relations
        for alias, canon in sorted(merged_aliases.items()):
            logger.info("knowledge_graph: 实体对齐 %r -> %r", alias, canon)

        # 重写实体：规范名合并，types 取并集
        merged: dict[str, dict] = {}
        for e in entities:
            name = str(e.get("name", "")).strip()
            canon = alias_map.get(name, name)
            if canon in merged:
                for t in e.get("types", []):
                    if t not in merged[canon]["types"]:
                        merged[canon]["types"].append(t)
            else:
                merged[canon] = {"name": canon, "types": list(e.get("types", []))}
        new_entities = list(merged.values())

        # 重写关系端点；丢弃合并产生的自环，端点重写后去重
        new_relations: list[dict] = []
        seen_rel = set()
        for r in relations:
            src = alias_map.get(
                str(r.get("source", "")).strip(), str(r.get("source", "")).strip()
            )
            tgt = alias_map.get(
                str(r.get("target", "")).strip(), str(r.get("target", "")).strip()
            )
            if not src or not tgt or src == tgt:
                continue
            rtype = str(r.get("type", "RELATED_TO")).strip().upper()
            key = (src, tgt, rtype)
            if key in seen_rel:
                continue
            seen_rel.add(key)
            new_relations.append({"source": src, "target": tgt, "type": rtype})

        logger.info(
            "knowledge_graph: 实体对齐完成 %d 个别名合并，实体 %d->%d，关系 %d->%d",
            len(merged_aliases), len(entities), len(new_entities),
            len(relations), len(new_relations),
        )
        return new_entities, new_relations

    def _judge_pairs(
        self, names: list[str], candidates: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """把模糊候选对交给 LLM 批量裁决是否同一实体（DERA 式思路）。

        一次调用判断全部候选对；解析失败/异常时返回空列表——
        宁可少合并，不做错误合并。
        """
        lines = [
            f'{idx}. "{names[i]}" / "{names[j]}"'
            for idx, (i, j) in enumerate(candidates, 1)
        ]
        prompt = (
            "以下实体名对来自同一篇文档的知识图谱抽取结果。"
            "请判断每一对是否指同一个对象（同一人/物/地点/概念的不同写法，"
            "如「年轻的王子」与「王子」）。\n"
            "注意：仅字面相似但实际是不同对象的必须判 false"
            "（例如「妈妈」与「新妈妈」通常不是同一人，「国王」与「王子」不是同一人）。\n"
            '只输出JSON：{"judgments": [{"id": 1, "same": true}, ...]}\n\n'
            "候选对：\n" + "\n".join(lines)
        )
        try:
            result = chat_json(
                [{"role": "user", "content": prompt}],
                model=get_settings().kg_model,
            )
        except Exception as e:
            logger.warning("knowledge_graph: 实体对齐裁决失败，放弃模糊对: %s", e)
            return []

        id2pair = {idx: pair for idx, pair in enumerate(candidates, 1)}
        confirmed: list[tuple[int, int]] = []
        for item in result.get("judgments", []):
            try:
                pid = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            if item.get("same") is True and pid in id2pair:
                confirmed.append(id2pair[pid])
        return confirmed

    def _validate_directions(
        self, doc_type: str, entities: list[dict], relations: list[dict]
    ) -> list[dict]:
        """按固定 Schema 的关系方向约束校验/修复边。

        方向写反是 LLM 抽取的高频错误（如 School-WORKS_AT->Person）。
        对每条边：头尾类型都匹配 → 保留；仅反方向匹配 → 自动翻转；
        都不匹配 → 丢弃。端点实体缺类型或关系无方向约束时保守保留。
        other 类型无 Schema，直接原样返回。
        """
        directions = self.RELATION_DIRECTIONS.get(doc_type)
        if not directions:
            return relations

        # 实体名 -> 类型集合（对齐合并后 types 是并集）
        type_map: dict[str, set] = {}
        for e in entities:
            types = e.get("types") or []
            if types:
                type_map.setdefault(str(e.get("name", "")), set(types))

        kept: list[dict] = []
        flipped = dropped = 0
        for r in relations:
            rtype = str(r.get("type", "")).strip().upper()
            src = str(r.get("source", ""))
            tgt = str(r.get("target", ""))
            constraint = directions.get(rtype)
            if constraint is None:
                kept.append({**r, "type": rtype})
                continue
            head_types, tail_types = constraint
            stype = type_map.get(src, set())
            ttype = type_map.get(tgt, set())
            if not stype or not ttype:
                # 端点实体无类型信息（可能被对齐合并/硬过滤波及），保守保留
                kept.append({**r, "type": rtype})
                continue
            if stype & head_types and ttype & tail_types:
                kept.append({**r, "type": rtype})
            elif stype & tail_types and ttype & head_types:
                kept.append({"source": tgt, "target": src, "type": rtype})
                flipped += 1
            else:
                dropped += 1

        if flipped or dropped:
            logger.info(
                "knowledge_graph: 方向校验（%s）翻转 %d 条、丢弃 %d 条",
                doc_type, flipped, dropped,
            )
        return kept

    @staticmethod
    def _save_graph_json(
        entities: list[dict],
        relations: list[dict],
        item_name: str,
        previous_item_names: list[str] = None,
    ):
        """将图谱数据追加到 data/graph_data.json（增量合并，按实体名去重）。

        持模块级锁串行化：并发导入时读-改-写共享文件，无锁会互相覆盖。
        """
        with _GRAPH_JSON_LOCK:
            KnowledgeGraphNode._write_graph_json_locked(
                entities, relations, item_name, previous_item_names
            )

    @staticmethod
    def _write_graph_json_locked(
        entities: list[dict],
        relations: list[dict],
        item_name: str,
        previous_item_names: list[str] = None,
    ):
        """写入 graph_data.json（调用方必须已持有 _GRAPH_JSON_LOCK）。"""
        import json as _json
        from pathlib import Path as _Path

        data_dir = _Path(__file__).resolve().parent.parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        graph_file = data_dir / "graph_data.json"

        # 读取已有数据
        existing = {"nodes": [], "edges": []}
        if graph_file.exists():
            try:
                existing = _json.loads(graph_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        # 同文档替换：清掉该 item_name 及历史名的旧节点/旧边，再写入本次
        # 抽取结果（保证重复导入幂等，图谱反映最新一次抽取，而非新旧叠加；
        # 历史名来自文档指纹——LLM 抽取名不稳定，同一文档可能改名）
        stale_names = {item_name, *(previous_item_names or [])}
        existing["nodes"] = [
            n for n in existing["nodes"] if n.get("item_name") not in stale_names
        ]
        existing["edges"] = [
            e for e in existing["edges"] if e.get("item_name") not in stale_names
        ]

        # 合并节点（按 name+item_name 去重）
        seen_nodes = {(n.get("name"), n.get("item_name")) for n in existing["nodes"]}
        for e in entities:
            name = str(e.get("name", "")).strip()
            if not name:
                continue
            key = (name, item_name)
            if key not in seen_nodes:
                existing["nodes"].append({
                    "name": name,
                    "types": e.get("types", []),
                    "item_name": item_name,
                })
                seen_nodes.add(key)

        # 有效节点名集合（全量，跨产品），用于边端点校验
        valid_names = {str(n.get("name", "")).strip() for n in existing["nodes"]}
        valid_names.discard("")

        # 合并边（按 source+target+type 去重，且两端必须存在对应节点）
        seen_edges = {(e.get("source"), e.get("target"), e.get("type")) for e in existing["edges"]}
        for r in relations:
            src = str(r.get("source", "")).strip()
            tgt = str(r.get("target", "")).strip()
            rtype = str(r.get("type", "RELATED_TO")).strip().upper()
            if not src or not tgt:
                continue
            key = (src, tgt, rtype)
            if key in seen_edges:
                continue
            if src not in valid_names or tgt not in valid_names:
                logger.debug("graph_data: 丢弃孤儿边 %s -[%s]-> %s", src, rtype, tgt)
                continue
            existing["edges"].append({
                "source": src,
                "target": tgt,
                "type": rtype,
                "item_name": item_name,
            })
            seen_edges.add(key)

        # 自愈：清理存量孤儿边（历史数据可能引用了不存在的节点）
        before = len(existing["edges"])
        existing["edges"] = [
            e for e in existing["edges"]
            if e.get("source") in valid_names and e.get("target") in valid_names
        ]
        if len(existing["edges"]) < before:
            logger.info("graph_data: 清理孤儿边 %d 条", before - len(existing["edges"]))

        graph_file.write_text(
            _json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("graph_data.json 已更新: %d 节点, %d 边", len(existing["nodes"]), len(existing["edges"]))

    def _classify_doc(self, sample: str) -> str:
        """按文档开头采样判断文档类型，返回 manual/paper/narrative/other。

        temperature=0 + JSON 约束保证分类稳定；失败时回退 other
        （抽取端自动降级为 LLM 自定义类型）。
        """
        prompt = (
            "判断以下文本属于哪种文档类型，只输出JSON。\n"
            "- manual：设备手册/产品说明书/操作指南\n"
            "- paper：学术论文/技术报告\n"
            "- narrative：小说/剧本/故事/叙事文\n"
            "- resume：个人简历/求职简历\n"
            "- other：以上都不是\n"
            '输出格式：{"doc_type": "manual|paper|narrative|resume|other"}\n\n'
            f"文本开头：\n{sample}"
        )
        try:
            result = chat_json(
                [{"role": "user", "content": prompt}],
                model=get_settings().kg_model,
            )
            doc_type = str(result.get("doc_type", "other")).strip().lower()
        except Exception as e:
            logger.warning("knowledge_graph: 文档分类失败，回退 other: %s", e)
            return "other"
        return doc_type if doc_type in self.SCHEMAS else "other"

    def _extract(self, content: str, entity_types: list[str], relation_types: list[str]) -> dict:
        """调用 LLM 从单个 chunk 抽取实体与关系（含 Gleaning 补采），返回 dict。

        doc_type 有固定 Schema 时强约束实体/关系类型只能从列表中选；
        other（分类失败或未覆盖的类型）回退为 LLM 自定义 3~8 类。
        首轮抽取后按 GLEANING_ROUNDS 做 GraphRAG 式补采，合并去重。
        """
        if entity_types:
            type_instruction = (
                "实体类型只能从以下列表中选择：\n"
                f"{', '.join(entity_types)}\n"
                f"关系类型只能从以下列表中选择：\n{', '.join(relation_types)}\n"
            )
        else:
            type_instruction = (
                "请根据内容自行定义 3~8 个合适的实体类型（如 Character/Location/"
                "Event/Concept/Method/Dataset 等）和 3~8 个关系类型"
                "（如 APPEARS_IN/LOCATED_AT/CAUSES/PRODUCES/EVALUATES_ON 等）。\n"
            )
        def _build_prompt(extra: str = "") -> str:
            return (
                "你是一个知识图谱信息抽取助手。请从以下文本中抽取实体和关系。\n\n"
                f"{type_instruction}\n"
                "要求：抽取 5~30 个最重要的实体，实体名使用文本中的原文词汇；"
                "每条关系必须有文本依据。\n"
                "请输出JSON，格式如下：\n"
                "{\n"
                '  "entities": [{"name": "实体名称", "types": ["实体类型"]}],\n'
                '  "relations": [{"source": "源实体名称", "target": "目标实体名称", "type": "关系类型"}]\n'
                "}\n"
                "只输出JSON，不要包含解释性文字。"
                f"{extra}文本内容：\n{content}"
            )

        # 第一轮抽取
        first = chat_json(
            [{"role": "user", "content": _build_prompt()}],
            model=get_settings().kg_model,
        )
        entities = list(first.get("entities", []))
        relations = list(first.get("relations", []))

        # GraphRAG 式 Gleaning 补采：带已抽取清单再问一轮「漏了什么」，
        # 补采返回空（无遗漏）即提前停止；不做多轮以免成本翻倍。
        for _ in range(self.GLEANING_ROUNDS):
            known = [str(e.get("name", "")).strip() for e in entities if e.get("name")]
            known_line = "、".join(known) if known else "（无）"
            extra = (
                f"\n注意：以下实体已经抽取过，不要再输出：{known_line}。\n"
                "请补充抽取此前遗漏的重要实体和关系（沿用相同的类型约束）；"
                "若确无遗漏，entities 和 relations 返回空数组。"
            )
            glean = chat_json(
                [{"role": "user", "content": _build_prompt(extra)}],
                model=get_settings().kg_model,
            )
            g_entities = glean.get("entities", []) or []
            g_relations = glean.get("relations", []) or []
            if not g_entities and not g_relations:
                break

            seen_names = {str(e.get("name", "")).strip() for e in entities}
            for e in g_entities:
                name = str(e.get("name", "")).strip()
                if name and name not in seen_names:
                    entities.append(e)
                    seen_names.add(name)

            seen_rel = {
                (str(r.get("source", "")).strip(),
                 str(r.get("target", "")).strip(),
                 str(r.get("type", "")).strip().upper())
                for r in relations
            }
            for r in g_relations:
                src = str(r.get("source", "")).strip()
                tgt = str(r.get("target", "")).strip()
                rtype = str(r.get("type", "")).strip().upper()
                if src and tgt and (src, tgt, rtype) not in seen_rel:
                    relations.append({"source": src, "target": tgt, "type": rtype})
                    seen_rel.add((src, tgt, rtype))

        logger.info(
            "knowledge_graph: 切片抽取完成（含补采），实体=%d，关系=%d",
            len(entities), len(relations),
        )
        return {"entities": entities, "relations": relations}
