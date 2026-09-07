# eval_rag/run_eval.py
"""对 RAG 评测集运行检索评测，计算 Recall@K / MRR / NDCG@K。

用法：
  python -m eval_rag.run_eval --k 5

输出：
  Per-query 指标 + 汇总平均值
  (可选) --keep 保留临时 store 不删除
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_team.sql_rag import SQLExperienceStore


def load_queries(queries_path: str) -> list[dict]:
    with open(queries_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_relevant(retrieved, query_meta: dict) -> bool:
    """判定 retrieved 经验是否与 query 相关。

    retrieved 可以是 SQLExperience 对象或 dict。
    标准：同一 db_id + 至少一张表重叠。
    """
    if hasattr(retrieved, 'db_id'):
        r_db, r_tables = retrieved.db_id, set(retrieved.tables_used)
    else:
        r_db, r_tables = retrieved.get("db_id", ""), set(retrieved.get("tables_used", []))
    if r_db != query_meta["db_id"]:
        return False
    return len(r_tables & set(query_meta["tables_used"])) > 0


def compute_dcg(scores: list[float], k: int) -> float:
    """Discounted Cumulative Gain @ K。"""
    dcg = 0.0
    for i, rel in enumerate(scores[:k]):
        dcg += rel / np.log2(i + 2)  # i+2 because log2(1+1) = log2(2) = 1
    return dcg


def compute_ndcg(retrieved_list: list[dict], query_meta: dict, k: int,
                 relevance_fn) -> float:
    """NDCG@K：归一化折损累积增益。

    relevance_fn(retrieved_item, query_meta) → float (relevance score)
    """
    # 获取每个检索结果的 relevance
    rels = [relevance_fn(r, query_meta) for r in retrieved_list[:k]]

    dcg = compute_dcg(rels, k)

    # Ideal DCG：假设所有相关结果排在前面
    ideal_rels = sorted(rels, reverse=True)
    idcg = compute_dcg(ideal_rels, k)

    return dcg / idcg if idcg > 0 else 0.0


def compute_recall_at_k(retrieved_list: list[dict], query_meta: dict, k: int,
                        relevance_fn, total_relevant: int) -> float:
    """Recall@K：前 K 个结果中命中了多少相关文档。"""
    if total_relevant == 0:
        return 0.0
    hits = sum(1 for r in retrieved_list[:k] if relevance_fn(r, query_meta))
    return hits / total_relevant


def compute_mrr(retrieved_list: list[dict], query_meta: dict,
                relevance_fn) -> float:
    """MRR：第一个相关结果的倒数排名。"""
    for i, r in enumerate(retrieved_list):
        if relevance_fn(r, query_meta):
            return 1.0 / (i + 1)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--store", default=None,
                        help="Path to eval_store.json (default: eval_rag/eval_store.json)")
    parser.add_argument("--queries", default=None,
                        help="Path to eval_queries.json (default: eval_rag/eval_queries.json)")
    parser.add_argument("--k", type=int, default=5, help="Top-K for Recall/NDCG (default: 5)")
    parser.add_argument("--exclude-self", action="store_true", default=True,
                        help="Exclude self-match from results (default: True)")
    parser.add_argument("--no-exclude-self", dest="exclude_self", action="store_false",
                        help="Keep self-match in results")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the eval store after running (default: delete it)")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    store_path = args.store or os.path.join(base_dir, "eval_store.json")
    queries_path = args.queries or os.path.join(base_dir, "eval_queries.json")

    if not os.path.exists(store_path):
        print(f"Store not found: {store_path}")
        print("Run build_eval_set.py first.")
        sys.exit(1)
    if not os.path.exists(queries_path):
        print(f"Queries not found: {queries_path}")
        sys.exit(1)

    # 加载
    store = SQLExperienceStore()
    store.load(store_path)
    queries = load_queries(queries_path)

    print(f"Loaded {store.count} experiences, {len(queries)} queries")

    # 统计每个 query 有多少相关文档（ground truth）
    query_total_relevant = []
    for qm in queries:
        q_tables = set(qm["tables_used"])
        q_db = qm["db_id"]
        count = 0
        for exp in store._experiences:
            if exp.db_id == q_db and set(exp.tables_used) & q_tables:
                count += 1
        query_total_relevant.append(count)

    # 逐题评测
    results = []
    for i, qm in enumerate(queries):
        question = qm["question"]

        # 检索 top-k（多取一个，因为可能被 self-exclude 掉一个）
        fetch_k = args.k + 1 if args.exclude_self else args.k
        retrieved_objs = store.retrieve(question, top_k=fetch_k)

        # Exclude self: 过滤掉与 query 完全相同题目的经验
        if args.exclude_self:
            retrieved_objs = [r for r in retrieved_objs if r.question != question]
        retrieved_objs = retrieved_objs[:args.k]

        # 调整 total_relevant：排除自身后
        total_rel = query_total_relevant[i]
        if args.exclude_self:
            # 自己是否算在 total_relevant 里？算的话减去 1
            total_rel = max(0, total_rel - 1)

        recall_1 = compute_recall_at_k(retrieved_objs, qm, 1, is_relevant, total_rel)
        recall_k = compute_recall_at_k(retrieved_objs, qm, args.k, is_relevant, total_rel)
        mrr = compute_mrr(retrieved_objs, qm, is_relevant)
        ndcg = compute_ndcg(retrieved_objs, qm, args.k, is_relevant)

        # 实际命中数
        hits = sum(1 for r in retrieved_objs if is_relevant(r, qm))

        results.append({
            "question_id": qm["question_id"],
            "db_id": qm["db_id"],
            "total_relevant": total_rel,
            "hits@K": hits,
            f"Recall@1": round(recall_1, 4),
            f"Recall@{args.k}": round(recall_k, 4),
            "MRR": round(mrr, 4),
            f"NDCG@{args.k}": round(ndcg, 4),
        })

        # 打印每条结果
        print(f"\n[{i+1}/{len(queries)}] {qm['question_id']} | {qm['db_id']}")
        print(f"  Q: {question[:80]}")
        print(f"  Relevant in store: {total_rel}")
        print(f"  Retrieved (top-{args.k}):")
        for j, r in enumerate(retrieved_objs):
            rel_mark = "HIT" if is_relevant(r, qm) else "MISS"
            q_preview = r.question[:50] + "..." if len(r.question) > 50 else r.question
            print(f"    [{j+1}] {rel_mark} db={r.db_id} tables={r.tables_used} q=\"{q_preview}\"")
        print(f"  Recall@1={recall_1:.3f}  Recall@{args.k}={recall_k:.3f}  "
              f"MRR={mrr:.3f}  NDCG@{args.k}={ndcg:.3f}")

    # 汇总
    avg_recall_1 = np.mean([r["Recall@1"] for r in results])
    avg_recall_k = np.mean([r[f"Recall@{args.k}"] for r in results])
    avg_mrr = np.mean([r["MRR"] for r in results])
    avg_ndcg = np.mean([r[f"NDCG@{args.k}"] for r in results])

    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(queries)} queries, top-{args.k})")
    print(f"  Recall@1:  {avg_recall_1:.4f}")
    print(f"  Recall@{args.k}: {avg_recall_k:.4f}")
    print(f"  MRR:       {avg_mrr:.4f}")
    print(f"  NDCG@{args.k}:  {avg_ndcg:.4f}")

    # 按 db_id 分组统计
    by_db = defaultdict(list)
    for r in results:
        by_db[r["db_id"]].append(r)
    print(f"\n  Per-database MRR:")
    for db_id in sorted(by_db):
        db_mrr = np.mean([r["MRR"] for r in by_db[db_id]])
        print(f"    {db_id}: MRR={db_mrr:.4f} ({len(by_db[db_id])} queries)")

    # 清理
    if not args.keep:
        os.remove(store_path)
        print(f"\nDeleted temp store: {store_path}")
    else:
        print(f"\nKept store: {store_path}")


if __name__ == "__main__":
    main()
