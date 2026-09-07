# scripts/backfill_rag_db_ids.py
"""从 BIRD 数据集和 benchmark 结果中匹配 question→db_id，补全 RAG 经验库的 db_id。

用法:
  cd nl2sql_agent
  python scripts/backfill_rag_db_ids.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_team.sql_rag import SQLExperienceStore


def build_question_map():
    """扫描 BIRD 数据集和 benchmark 输出，构建 question→db_id 映射。"""
    base = os.path.dirname(os.path.dirname(__file__))
    qmap = {}

    # 1. BIRD mini_dev 数据集
    bird_dev = os.path.join(
        base, "..", "_archive", "benchmarks", "bird", "minidev",
        "MINIDEV", "mini_dev_sqlite.json"
    )
    if os.path.exists(bird_dev):
        with open(bird_dev, encoding="utf-8") as f:
            for entry in json.load(f):
                q = (entry.get("question") or "").strip()
                if q and entry.get("db_id"):
                    qmap[q] = entry["db_id"]

    # 2. 所有 benchmark_output 下的 bird_results.json
    bo_dir = os.path.join(base, "benchmark_output")
    if os.path.isdir(bo_dir):
        for root, dirs, files in os.walk(bo_dir):
            if "bird_results.json" in files:
                fpath = os.path.join(root, "bird_results.json")
                try:
                    with open(fpath, encoding="utf-8") as f:
                        data = json.load(f)
                    for r in data.get("results", []):
                        q = (r.get("question") or "").strip()
                        if q and r.get("db_id"):
                            if q not in qmap:
                                qmap[q] = r["db_id"]
                except Exception:
                    pass

    # 3. bird_benchmark_output
    bird_bo = os.path.join(base, "bird_benchmark_output", "bird_results.json")
    if os.path.exists(bird_bo):
        try:
            with open(bird_bo, encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("results", []):
                q = (r.get("question") or "").strip()
                if q and r.get("db_id"):
                    if q not in qmap:
                        qmap[q] = r["db_id"]
        except Exception:
            pass

    return qmap


def main():
    rag_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "agent_team", "data", "sql_experiences.json",
    )
    if not os.path.exists(rag_path):
        print(f"ERROR: RAG file not found: {rag_path}")
        sys.exit(1)

    qmap = build_question_map()
    print(f"Question→db_id 映射: {len(qmap)} 条")

    store = SQLExperienceStore()
    store.load(rag_path)
    print(f"RAG 加载: {store.count} 条经验")

    updated = 0
    empty_before = sum(1 for e in store._experiences if not e.db_id)
    for exp in store._experiences:
        if exp.db_id:
            continue
        db_id = qmap.get(exp.question.strip())
        if db_id:
            exp.db_id = db_id
            updated += 1

    store._rebuild_embeddings()  # db_id 变了不影响向量
    store.save(rag_path)

    empty_after = sum(1 for e in store._experiences if not e.db_id)
    db_ids = set(e.db_id for e in store._experiences if e.db_id)
    print(f"补全前空 db_id: {empty_before}")
    print(f"补全后空 db_id: {empty_after} (更新 {updated} 条)")
    print(f"不同数据库数: {len(db_ids)}")
    print(f"db_ids: {sorted(db_ids)}")


if __name__ == "__main__":
    main()
