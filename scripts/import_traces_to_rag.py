# scripts/import_traces_to_rag.py
"""从 traces/ 目录批量导入执行记录到 RAG 经验库。

用法:
  cd nl2sql_agent
  python scripts/import_traces_to_rag.py --count 20
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent_team.sql_rag import SQLExperienceStore, SQLExperience, classify_error


def main():
    parser = argparse.ArgumentParser(description="导入 traces 到 RAG")
    parser.add_argument("--count", type=int, default=20, help="导入最近 N 条不重复 trace")
    parser.add_argument("--traces-dir", default="agent_team/traces")
    parser.add_argument("--rag-path", default="agent_team/data/sql_experiences.json")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不写入")
    args = parser.parse_args()

    traces_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.traces_dir)
    rag_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.rag_path)

    if not os.path.isdir(traces_dir):
        print(f"ERROR: traces dir not found: {traces_dir}")
        sys.exit(1)

    # 按修改时间排序，取最新的
    files = []
    for fname in os.listdir(traces_dir):
        if fname.endswith(".json"):
            fpath = os.path.join(traces_dir, fname)
            files.append((os.path.getmtime(fpath), fpath, fname))
    files.sort(key=lambda x: x[0], reverse=True)

    # 逐条读取，按 question 去重（保留最新的）
    seen_questions = set()
    entries = []
    for mtime, fpath, fname in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                trace = json.load(f)
        except Exception:
            continue

        q = (trace.get("question") or "").strip()
        if not q or q in seen_questions:
            continue
        seen_questions.add(q)

        sql = (trace.get("pred_sql") or "").strip()
        if not sql:
            continue

        tables_used = []
        rs = trace.get("relevant_schema") or {}
        for t in rs.get("candidate_tables", []):
            tables_used.append(t.get("name", ""))

        success = bool(trace.get("success"))
        exec_ok = bool(trace.get("exec_ok"))
        iterations = int(trace.get("iterations", 1))
        error_msg = trace.get("error") or ""
        exec_error = trace.get("exec_error") or ""

        # 分类失败类型
        error_type = ""
        if not success and error_msg:
            error_type = classify_error(error_msg)
        elif not exec_ok and exec_error:
            error_type = classify_error(exec_error)

        entries.append({
            "question": q,
            "sql": sql,
            "tables_used": tables_used,
            "success": success and exec_ok,
            "error_type": error_type,
            "iteration_count": iterations,
            "fname": fname,
        })

        if len(entries) >= args.count:
            break

    if not entries:
        print("No valid traces found.")
        return

    # ── 预览 ──
    print(f"{'='*70}")
    print(f"  准备导入 {len(entries)} 条经验")
    print(f"{'='*70}")
    for i, e in enumerate(entries):
        status = "OK" if e["success"] else f"FAIL({e['error_type']})"
        print(f"  [{i+1:2d}] [{status}] {e['question'][:80]}")
        print(f"       Tables: {e['tables_used']}, Iters: {e['iteration_count']}")
        print(f"       SQL: {e['sql'][:100]}...")
        print()

    if args.dry_run:
        print("[DRY RUN] 未实际写入。")
        return

    # ── 写入 RAG ──
    store = SQLExperienceStore()
    if os.path.exists(rag_path):
        store.load(rag_path)
        print(f"已加载现有 RAG: {store.count} 条经验")

    added = 0
    for e in entries:
        exp = SQLExperience(
            question=e["question"],
            sql=e["sql"],
            db_id="",
            tables_used=e["tables_used"],
            success=e["success"],
            error_type=e["error_type"],
            iteration_count=e["iteration_count"],
        )
        if store.add(exp):
            added += 1

    os.makedirs(os.path.dirname(rag_path), exist_ok=True)
    store.save(rag_path)
    print(f"\n写入完成: 新增 {added} 条, 总计 {store.count} 条")
    print(f"RAG 数据文件: {rag_path}")


if __name__ == "__main__":
    main()
