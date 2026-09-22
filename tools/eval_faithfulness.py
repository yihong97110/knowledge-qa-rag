# -*- coding: utf-8 -*-
"""LLM-as-judge 忠实度评估：估计幻觉率。

忠实度（faithfulness）严格定义：答案中的每个论断是否被「实际召回并
送入 LLM 的 context」支持。它不需要金标答案，只需要两样东西：
    - answer：答案全文（审计记录已快照）
    - contexts：最终送入 LLM 的上下文原文（审计记录已快照）
因此可以直接离线评估任何一条历史查询。

幻觉率口径：
    hallucination_rate = verdict != "supported" 的回答数 / 总评估回答数
    （即「至少含一条无依据论断」的回答占比，偏保守的严格口径）

judge 工程纪律：
    - temperature=0 + seed 固定（tools.llm_utils.chat_json 内置）；
    - 强制 JSON 结构化输出 {verdict, hallucinated_claims, reason}；
    - judge 模型应与生成模型不同以避免自我偏好：生产环境用
      --judge-model 指定更强/不同家族的模型；演示库仅一把 key，
      默认复用配置模型，解读结果时注意该局限；
    - 评估明细落盘 data/eval/faithfulness_*.jsonl，抽 10~20% 人工
      复核校准 judge 的可靠性。

用法：
    python tools/eval_faithfulness.py            # 评估最新 10 条运行记录
    python tools/eval_faithfulness.py --limit 30 # 评估最新 30 条
    python tools/eval_faithfulness.py --date 2026-09-17  # 指定日期文件
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from tools.query_audit import RUNS_DIR  # noqa: E402

EVAL_DIR = _ROOT / "data" / "eval"

# 上下文截断保护：judge prompt 不超 token 上限
_CTX_PER_DOC = 500   # 每条 context 截断长度
_CTX_MAX_DOCS = 5    # 最多送入条数（faithfulness 看前列即可）


def _build_judge_messages(query: str, answer: str, contexts: list[str]) -> list[dict]:
    ctx_texts = [c[:_CTX_PER_DOC] for c in contexts[:_CTX_MAX_DOCS] if c]
    ctx_block = "\n---\n".join(f"[文档{i+1}] {t}" for i, t in enumerate(ctx_texts))
    system = (
        "你是严格的RAG答案忠实度评审。判定答案中的每个论断是否被给定文档支持。"
        "判定规则：论断与文档内容一致或可由文档直接推出=有依据；"
        "文档中没有依据、或与文档矛盾=幻觉论断。通用常识性寒暄（如开头问候语）不算论断。"
        "只输出JSON：{\"verdict\": \"supported|partial|unsupported\", "
        "\"hallucinated_claims\": [\"无依据论断原文\", ...], \"reason\": \"简短理由\"}。"
        "verdict 定义：supported=全部论断有依据；partial=部分有依据；"
        "unsupported=核心论断基本无依据。"
    )
    user = f"用户问题：{query}\n\n参考文档：\n{ctx_block or '（无检索结果）'}\n\n待评答案：\n{answer}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _load_runs(date: str | None, limit: int) -> list[dict]:
    paths = [RUNS_DIR / f"{date}.jsonl"] if date else sorted(RUNS_DIR.glob("*.jsonl"))
    runs: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    # 只评有 answer 的记录，取最新 limit 条
    runs = [r for r in runs if r.get("answer")]
    return runs[-limit:]


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge 忠实度评估")
    parser.add_argument("--limit", type=int, default=10, help="评估最新 N 条记录")
    parser.add_argument("--date", default=None, help="只评估指定日期 YYYY-MM-DD 的记录")
    parser.add_argument("--judge-model", default=None,
                        help="judge 模型名（默认用配置的 llm_model；生产建议指定不同模型）")
    args = parser.parse_args()

    runs = _load_runs(args.date, args.limit)
    if not runs:
        print("无可评估记录（审计中无 answer 字段或文件不存在）")
        return

    from tools.llm_utils import chat_json

    out_path = EVAL_DIR / f"faithfulness_{time.strftime('%Y-%m-%d')}.jsonl"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_lines: list[str] = []
    n_hallu = 0
    n_claims = 0

    print(f"评估 {len(runs)} 条记录（judge={args.judge_model or '配置模型'}）")
    print("=" * 72)
    for rec in runs:
        msgs = _build_judge_messages(rec.get("query", ""), rec["answer"],
                                     rec.get("contexts", []))
        try:
            verdict = chat_json(msgs, model=args.judge_model)
        except Exception as e:
            print(f"[{rec.get('session_id')}] judge 调用失败，跳过: {e}")
            continue
        v = verdict.get("verdict", "unknown")
        claims = verdict.get("hallucinated_claims", [])
        n_claims += len(claims)
        if v in ("partial", "unsupported"):
            n_hallu += 1
        mark = "✅" if v == "supported" else "❌"
        print(f"{mark} [{rec.get('ts')}] {rec.get('query', '')[:30]:32s} "
              f"verdict={v:11s} 幻觉论断={len(claims)}")
        for c in claims[:3]:
            print(f"      - {c[:60]}")
        out_lines.append(json.dumps({
            "ts": rec.get("ts"),
            "session_id": rec.get("session_id"),
            "query": rec.get("query"),
            "verdict": v,
            "hallucinated_claims": claims,
            "reason": verdict.get("reason", ""),
            "config": rec.get("config", {}),
        }, ensure_ascii=False))

    # 汇总落盘（明细供人工复核校准 judge）
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    n = len(out_lines)
    print("=" * 72)
    if n:
        print(f"幻觉率 = {n_hallu}/{n} = {n_hallu / n:.1%}"
              f"（口径：含幻觉论断的回答占比）")
        print(f"幻觉论断总数 = {n_claims}")
    print(f"明细已落盘: {out_path}")
    print("提示：抽 10~20% 明细人工复核，校准 judge 判定可靠性。")


if __name__ == "__main__":
    main()
