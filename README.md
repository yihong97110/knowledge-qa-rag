# 掌柜智库 · 企业技术文档智能问答系统（RAG + GraphRAG）

基于 RAG + GraphRAG 混合架构的企业文档问答系统：四路并行检索 → RRF 名次融合 → 重排序 → SSE 流式生成，并内置完整评估闭环（recall@k / MRR / 分路归因 / LLM-as-judge 幻觉率）。

## 核心特性

- **四路并行检索**：向量检索（Milvus）/ HyDE 假设文档 / BM25（导入期构建、pickle 持久化倒排索引，查询期仅查不建）/ 知识图谱（Neo4j，两跳关系遍历）
- **RRF 名次融合 + Reranker 精排**：多路命中加权，分数不可比也能公平融合；重排后阈值截断
- **流式输出**：FastAPI SSE 逐 token 推送，全程记录 TTFT（首字延迟）
- **导入流水线**：PDF→Markdown → 句子边界滑窗切分（超限回退句号边界，不腰斩）→ 商品名识别 → 向量化入库 → 知识图谱构建（固定 Schema 白名单 + Gleaning 补采 + 实体对齐 + 关系方向校验 + 锚点保障）
- **评估闭环**：金标集离线回放（recall@k / MRR / 分路归因，区分"检索失败"与"排序失败"）+ LLM-as-judge 忠实度评估（幻觉率估计）

## 技术栈

FastAPI · Milvus Lite · Neo4j · MongoDB · BM25 (rank_bm25) · DashScope (qwen) · BGE / text-embedding-v4 · gte-rerank-v2

## 目录结构

```
├── api/app.py            # FastAPI 入口（SSE 流式问答 + 文档导入任务）
├── core/config.py        # 全局配置（环境变量驱动）
├── processor/
│   ├── import_process/   # 导入流水线：切分 / 向量化 / 图谱构建
│   └── query_process/    # 查询流水线：改写 / 四路检索 / RRF / 精排
├── tools/                # 存储与工具层（milvus / neo4j / mongo / bm25 / llm / 审计 / 评估）
├── static/               # 前端页面（chat 问答 / import 导入 / graph 图谱可视化）
└── requirements.txt
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env   # 填入你的 API Key 与数据库密码

# 3. 启动依赖服务：MongoDB(27017)、Neo4j(7687)

# 4. 启动应用（在仓库根目录下执行）
python -m uvicorn api.app:app --host 127.0.0.1 --port 8001
```

然后浏览器打开：

- `http://127.0.0.1:8001/` —— 文档导入页
- `http://127.0.0.1:8001/static/chat.html` —— 问答页（流式）
- `http://127.0.0.1:8001/static/graph.html` —— 知识图谱可视化

> Milvus Lite 与 BM25 索引为本地文件，首次导入文档后自动生成。

## 评估工具

```bash
# 从 BM25 索引导出全库切片清单，供人工标注金标集
python tools/eval_recall.py export

# 金标集 + 查询审计日志 → recall@k / MRR / 分路归因
python tools/eval_recall.py eval

# LLM-as-judge：对最近一次运行记录做忠实度评估，输出幻觉率
python tools/eval_faithfulness.py --limit 10
```

## 说明

- `.env` 不入库（已在 .gitignore 排除），请使用 `.env.example` 作为模板
- `data/`（审计日志、金标集、BM25 索引）与 `uploads/`（上传的原始文档）为运行时产物，不入库
