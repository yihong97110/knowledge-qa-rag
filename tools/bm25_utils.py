"""BM25 稀疏检索的索引持久化与共享分词。

为什么需要本模块（对照旧实现）：
- 旧实现（query BM25SearchNode）每次查询都执行 fetch_all_chunks()
  （Milvus 全表扫描）+ 全库分词 + BM25Okapi 构建，语料增长后成为
  查询延迟的主要开销之一。
- 现改为「导入期构建、查询期只查」：
  * 导入侧：insert_chunks 成功后调用 rebuild_index()，构建索引并
    原子落盘 pickle（data/bm25_index.pkl）；
  * 查询侧：search() 走进程内单例 → pickle → 回源重建的三级获取，
    仅当 pickle 缺失或与 Milvus 语料条数失配时才回源（自愈）。

pickle 安全性说明：索引文件仅由本服务在本机写入，不对外分发，
pickle.load 只用于受信任的本地文件。
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
import time
from pathlib import Path

logger = logging.getLogger("bm25.index")

_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = _ROOT / "data" / "bm25_index.pkl"

_CJK = re.compile(r"[\u4e00-\u9fff]")

# 进程内单例：查询热路径零磁盘读、零网络扫
_INDEX: dict | None = None


# ================================================================== #
#                       共享分词（查询/语料/导入侧统一）              #
# ================================================================== #

def _split_token(t: str) -> list[str]:
    if len(t) <= 2 or not _CJK.search(t):
        return [t]  # 短 token 或纯英文/数字，原样保留
    return [t] + [t[i:i + 2] for i in range(len(t) - 1)]


def tokenize(text: str) -> list[str]:
    """中英文混合分词，查询与语料必须共用同一规则保证词面空间一致。

    规则：\\w+ 切基础 token（lowercase）；含中文的长 token 追加「相邻
    两字」二元切分——\\w+ 会把无标点中文整句切成一个超长 token，导致
    BM25 字面匹配永远打不中（旧实现踩过的坑，见 query nodes.py 注释）。
    """
    tokens: list[str] = []
    for t in re.findall(r"\w+", text.lower()):
        if len(t) > 1:
            tokens.extend(_split_token(t))
    return tokens


# ================================================================== #
#                       索引构建 / 落盘 / 加载                        #
# ================================================================== #

def _fingerprint(chunks: list[dict]) -> str:
    """语料指纹：逐条内容哈希后排序再总哈希（与切片顺序无关）。"""
    per = sorted(
        hashlib.md5(
            f"{c.get('content', '')}\x1f{c.get('title', '')}\x1f{c.get('item_name', '')}"
            .encode("utf-8")
        ).hexdigest()
        for c in chunks
    )
    return hashlib.md5("\x1e".join(per).encode("utf-8")).hexdigest()


def build_index(chunks: list[dict]) -> dict:
    """构建 BM25 索引对象（含语料副本，保证索引与切片按下标对齐）。

    语料必须用全库而非当前 item：小文档（1-2 个切片）里任何词 DF=N_docs，
    BM25Okapi 的 IDF 为负/零，得分全部 <=0 被过滤，BM25 路恒空。
    全库语料让 DF 有分布差异，恢复区分能力；跨 item 的词面命中交给
    后续 RRF 融合与 Reranker 按相关性排序。
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(c.get("content", "")) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return {
        "bm25": bm25,
        "chunks": chunks,          # 与 BM25Okapi 内部 doc 顺序按下标一一对应
        "n_chunks": len(chunks),
        "fingerprint": _fingerprint(chunks),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def save_index(index: dict) -> None:
    """原子落盘：先写临时文件再 os.replace，避免读到半写文件。"""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, INDEX_PATH)


def load_index() -> dict | None:
    """读取落盘索引；文件缺失/损坏/结构异常一律返回 None（回源重建）。"""
    if not INDEX_PATH.exists():
        return None
    try:
        with INDEX_PATH.open("rb") as f:
            idx = pickle.load(f)
        if not isinstance(idx, dict) or "bm25" not in idx or "chunks" not in idx:
            return None
        return idx
    except Exception as e:
        logger.warning("BM25 索引读取失败，将回源重建: %s", e)
        return None


def _corpus_count() -> int | None:
    """当前 Milvus 语料条数（轻量新鲜度检查）；Milvus 不可用返回 None。"""
    try:
        from tools.milvus_utils import count_chunks
        return count_chunks()
    except Exception:
        return None


# ================================================================== #
#                       查询侧入口                                    #
# ================================================================== #

def get_index() -> dict | None:
    """获取可用 BM25 索引：进程内单例 → pickle → 回源重建（自愈）。

    新鲜度策略：仅在首次加载（进程启动后第一次查询）时用 Milvus
    条数做轻量校验；条数一致或 Milvus 不可查时信任 pickle。之后
    信任进程内单例——导入 API 与查询同进程，导入节点写完向量会
    立即 rebuild_index() 刷新单例；若是外部进程改了库（如 CLI 导入），
    重启服务进程即重新校验。
    """
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    idx = load_index()
    if idx is not None:
        n_live = _corpus_count()
        if n_live is None or n_live == idx.get("n_chunks"):
            _INDEX = idx
            logger.info(
                "BM25 索引已加载（pickle）: %d 条切片, 构建于 %s",
                idx["n_chunks"], idx.get("built_at", "?"),
            )
            return _INDEX
        logger.info(
            "BM25 pickle 落后于 Milvus 语料（pickle=%s, live=%s），回源重建",
            idx.get("n_chunks"), n_live,
        )
    return rebuild_index()


def rebuild_index() -> dict | None:
    """回源全量重建索引并落盘（导入侧写入成功后 / pickle 缺失时调用）。

    所有失败路径只告警并返回 None，绝不抛出——导入主流程与查询
    主流程都不应被索引问题阻断（查询侧拿不到索引时 BM25 路为空，
    由 degraded 标记暴露）。
    """
    global _INDEX
    try:
        from tools.milvus_utils import fetch_all_chunks
        chunks = fetch_all_chunks()
    except Exception as e:
        logger.warning("BM25 索引重建失败（Milvus 不可用？）: %s", e)
        return None
    if not chunks:
        logger.info("知识库为空，跳过 BM25 索引重建")
        return None

    idx = build_index(chunks)
    try:
        save_index(idx)
    except Exception as e:
        logger.warning("BM25 索引落盘失败（不影响本次使用）: %s", e)
    _INDEX = idx
    logger.info(
        "BM25 索引已重建: %d 条切片, 指纹=%s", idx["n_chunks"], idx["fingerprint"][:8]
    )
    return idx


def search(query: str, top_k: int) -> list[dict]:
    """BM25 检索（查询期只查不建）。

    Args:
        query: 查询文本（已改写或原始）。
        top_k: 返回条数上限。

    Returns:
        [{content, title, item_name, score, source:"bm25"}]，按分数
        降序，过滤零分；索引不可用时返回空列表。
    """
    idx = get_index()
    if idx is None:
        return []

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    bm25 = idx["bm25"]
    chunks = idx["chunks"]
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results: list[dict] = []
    for i in ranked_indices[:top_k]:
        if scores[i] <= 0:
            break
        chunk = chunks[i]
        results.append({
            "content": chunk.get("content", ""),
            "title": chunk.get("title", ""),
            "item_name": chunk.get("item_name", ""),
            "score": float(scores[i]),
            "source": "bm25",
        })
    return results
