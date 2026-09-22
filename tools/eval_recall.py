# -*- coding: utf-8 -*-
"""评估最小闭环：recall@k 离线回放（跳过 LLM，零成本高频回归主力）。

数据流：
    1. 标注金标集：data/eval/golden_set.jsonl
       每行一条 JSON：{"query_id", "query", "relevant_hashes": [...], "note"}
       relevant_hashes 是金标切片的 content_hash（SHA256 前 16 位），
       与审计记录 final_topk/per_route 里的 content_hash 同源对齐。
       标注方法：先跑 `python tools/eval_recall.py export` 拿到
       全库切片清单（hash + item_name + 摘要），对照语料人工挑选
       能回答该 query 的切片 hash 写入金标集。20~50 条起步即可。

    2. 积累运行记录：对金标 query 各发起一次线上查询（/query + /stream），
       审计自动落盘到 data/query_runs/YYYY-MM-DD.jsonl。

    3. 离线评估：python tools/eval_recall.py eval
       从运行记录读 final_topk（recall@k / MRR）与 per_route
       （分路 recall，归因到检索路），末尾输出汇总。

归因约定（失败三分法）：
    - recall@k 低              → 检索失败（改分块/查询扩展/embedding）
    - 分路 recall 有明显短板    → 对应路的问题（如 BM25 分词、KG 锚点）
    - final_topk 命中但被截断   → 排序失败（改 RRF k/reranker/CLIFF_RATIO）

注意：审计记录的 query 字段是「改写后 query（无改写则原始）」，本脚本用
规范化后的双向包含做匹配；若 LLM 改写幅度过大导致匹配不上，该 query 会
被列入 [未匹配] 清单，属已知限制。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = _ROOT / "data" / "eval" / "golden_set.jsonl"

# 导入审计模块复用 RUNS_DIR，保证与写入侧一致
sys.path.insert(0, str(_ROOT))
from tools.query_audit import RUNS_DIR  # noqa: E402


# ---------------------------------------------------------------- #
#                     子命令 1：导出候选标注清单                       #
# ---------------------------------------------------------------- #
def cmd_export() -> None:
    """从 BM25 pickle 导出全库切片清单，供人工标注金标集。"""
    import pickle

    from tools.query_audit import _chunk_ref

    index_path = _ROOT / "data" / "bm25_index.pkl"
    if not index_path.exists():
        print(f"[错误] 索引文件不存在: {index_path}")
        print("先执行导入流程或调用 tools.bm25_utils.rebuild_index() 构建索引。")
        return

    with index_path.open("rb") as f:
        idx = pickle.load(f)
    chunks = idx.get("chunks", [])

    print(f"全库共 {len(chunks)} 条切片。标注格式（写入 {GOLDEN_PATH}）：")
    print('{"query_id": "g001", "query": "你的问题", '
          '"relevant_hashes": ["hash1", "hash2"], "note": "标注理由"}')
    print("=" * 72)
    for i, c in enumerate(chunks):
        # pickle chunks 未存 hash，现场按同一规则计算（SHA256 前 16 位）
        ref = _chunk_ref(c)
        summary = c.get("content", "")[:40].replace("\n", " ")
        print(f"{i:3d} | {ref['item_name'][:12]:12s} | "
              f"{ref['content_hash']} | {summary}")
    print("=" * 72)
    print("提示：金标标注原则——挑「包含答案依据」的切片，不是「相似」的切片；")


def _load_golden() -> list[dict]:
    if not GOLDEN_PATH.exists():
        print(f"[错误] 金标集不存在: {GOLDEN_PATH}")
        print("先执行 export 子命令获取切片清单，人工标注后重试。")
        return []
    items = []
    for ln, line in enumerate(GOLDEN_PATH.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[警告] 金标集第 {ln} 行 JSON 解析失败，跳过: {e}")
            continue
        if rec.get("query") and rec.get("relevant_hashes"):
            items.append(rec)
        else:
            print(f"[警告] 金标集第 {ln} 行缺 query 或 relevant_hashes，跳过")
    return items


def _load_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for path in sorted(RUNS_DIR.glob("*.jsonl")):
        for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[警告] {path.name}:{ln} JSON 解析失败，跳过")
    return runs


def _norm(s: str) -> str:
    return "".join(s.split()).lower()


def _match_run(query: str, runs: list[dict]) -> list[dict]:
    """金标 query 与运行记录宽松匹配（规范化后双向包含）。"""
    nq = _norm(query)
    hits = []
    for r in runs:
        nr = _norm(r.get("query", ""))
        if not nr:
            continue
        if nq in nr or nr in nq:
            hits.append(r)
    return hits


# ---------------------------------------------------------------- #
#                     子命令 2：离线评估                              #
# ---------------------------------------------------------------- #
def cmd_eval(k_values: list[int]) -> None:
    golden = _load_golden()
    if not golden:
        return
    runs = _load_runs()
    print(f"金标 {len(golden)} 条 | 运行记录 {len(runs)} 条\n" + "=" * 72)

    # 汇总累加器
    agg = {f"recall@{k}": [] for k in k_values}
    agg["mrr"] = []
    route_agg: dict[str, list[float]] = {}
    unmatched, no_run = [], []

    for g in golden:
        qid = g.get("query_id", "?")
        query = g["query"]
        relevant = set(g["relevant_hashes"])

        matched = _match_run(query, runs)
        if not matched:
            no_run.append((qid, query))
            continue

        # 同一 query 多次运行：取最新一次（按 ts 排序）
        run = max(matched, key=lambda r: r.get("ts", ""))

        final = [d["content_hash"] for d in run.get("final_topk", []) if d.get("content_hash")]
        if not final:
            print(f"[{qid}] {query[:30]}  运行记录无 final_topk（可能检索全空）")
            continue

        # ---- 逐 query 明细 ----
        detail = f"[{qid}] {query[:34]} | top{len(final)} "
        hit_rank = next((i + 1 for i, h in enumerate(final) if h in relevant), None)
        for k in k_values:
            rc = len(set(final[:k]) & relevant) / len(relevant)
            agg[f"recall@{k}"].append(rc)
            detail += f"R@{k}={rc:.2f} "
        if hit_rank:
            agg["mrr"].append(1.0 / hit_rank)
            detail += f"MRR=1/{hit_rank}"
        else:
            agg["mrr"].append(0.0)
            detail += "MRR=0(未命中)"
        print(detail)

        # ---- 分路 recall（前 10 条）----
        per_route = run.get("per_route", {})
        for route, refs in per_route.items():
            hashes = [d["content_hash"] for d in refs if d.get("content_hash")]
            if not hashes and route == "kg":
                # kg 路常空，跳过统计避免拉低均值误导
                continue
            rc = len(set(hashes) & relevant) / len(relevant)
            route_agg.setdefault(route, []).append(rc)

    print("=" * 72)
    n_eval = max(len(agg["mrr"]), 1)
    if n_eval == 1 and not agg["mrr"]:
        print("汇总：无可用评估数据")
    else:
        print(f"汇总（n={len(agg['mrr'])} 条已评估）：")
        for k in k_values:
            vals = agg[f"recall@{k}"]
            avg = sum(vals) / len(vals) if vals else 0.0
            print(f"  recall@{k:<2} = {avg:.3f}")
        print(f"  MRR       = {sum(agg['mrr']) / n_eval:.3f}")
        for route, vals in sorted(route_agg.items()):
            avg = sum(vals) / len(vals)
            bar = "#" * int(avg * 20)
            print(f"  分路 recall@10 [{route:9s}] = {avg:.3f} {bar}")

    if no_run:
        print(f"\n[未匹配] {len(no_run)} 条金标 query 无运行记录，先发起线上查询积累数据：")
        for qid, q in no_run:
            print(f"  {qid}: {q[:40]}")

    # 归因提示
    if agg["mrr"] and sum(agg["recall@10"]) / n_eval < 0.6:
        print("\n[归因提示] recall@10 均值偏低 → 检索失败为主，优先改分块/查询扩展。")
    elif agg["mrr"] and sum(agg["mrr"]) / n_eval < 0.5:
        print("\n[归因提示] 召回尚可但 MRR 低 → 排序失败为主，优先改 RRF k/reranker/截断。")


def main() -> None:
    parser = argparse.ArgumentParser(description="recall@k 离线评估（跳过 LLM）")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("export", help="导出全库切片清单供标注金标集")
    p_eval = sub.add_parser("eval", help="从金标集+运行记录计算 recall@k/MRR")
    p_eval.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10],
                        help="recall 截断点，默认 1 3 5 10")
    args = parser.parse_args()

    if args.cmd == "export":
        cmd_export()
    elif args.cmd == "eval":
        cmd_eval(args.k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
