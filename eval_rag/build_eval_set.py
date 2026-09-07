# eval_rag/build_eval_set.py
"""从 BIRD 题目中选取 N 题，构建 RAG 评测集。

输出：
  eval_rag/eval_store.json     — SQLExperienceStore 持久化
  eval_rag/eval_queries.json   — 查询元数据（question_id, question, db_id, tables, gold_sql）

用法：
  python -m eval_rag.build_eval_set \
      --questions ../_archive/benchmarks/bird/minidev/MINIDEV/mini_dev_sqlite.json \
      --tables ../_archive/benchmarks/bird/minidev/MINIDEV/dev_tables.json \
      --per-db 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Ensure agent_team is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_team.sql_rag import SQLExperienceStore, SQLExperience


def extract_tables_from_sql(sql: str) -> list[str]:
    """从 SQL 中提取表名（FROM/JOIN 后面的标识符）。"""
    sql_lower = sql.lower()
    tables = []

    # FROM table_name / JOIN table_name / FROM table AS alias
    pattern = r'(?:from|join)\s+(\w+)(?:\s+(?:as\s+)?\w+)?'
    seen = set()
    for m in re.finditer(pattern, sql_lower):
        t = m.group(1)
        # 排除 SQL 关键字
        if t in ('select', 'where', 'on', 'and', 'or', 'group', 'order', 'having', 'limit', 'union', 'intersect', 'except'):
            continue
        if t not in seen:
            seen.add(t)
            tables.append(t)

    return tables


def load_bird_questions(path: str) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_table_names(tables_path: str) -> dict[str, list[str]]:
    """从 BIRD tables.json 提取 {db_id: [table_name, ...]}。"""
    with open(tables_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    result = {}
    for entry in raw:
        db_id = entry["db_id"]
        names = entry.get("table_names", entry.get("table_names_original", []))
        result[db_id] = list(names)
    return result


def select_questions(questions: list[dict], per_db: int = 10) -> list[dict]:
    """每库各选 per_db 题，保证检索有多个相关结果。"""
    by_db: dict[str, list[dict]] = {}
    for q in questions:
        db_id = q.get("db_id", "")
        if db_id:
            by_db.setdefault(db_id, []).append(q)

    selected = []
    for db_id, pool in sorted(by_db.items()):
        take = min(len(pool), per_db)
        selected.extend(pool[:take])
    return selected


def main():
    parser = argparse.ArgumentParser(description="Build RAG eval set from BIRD questions")
    parser.add_argument("--questions", required=True, help="Path to BIRD mini_dev_sqlite.json")
    parser.add_argument("--tables", required=True, help="Path to BIRD dev_tables.json")
    parser.add_argument("--per-db", type=int, default=10, help="Questions per database (default: 10)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: eval_rag/)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(__file__)

    # 加载数据
    all_questions = load_bird_questions(args.questions)
    table_map = load_table_names(args.tables)

    # 选取题目
    selected = select_questions(all_questions, per_db=args.per_db)

    # 构建 RAG store
    store = SQLExperienceStore()
    query_meta = []

    for i, q in enumerate(selected):
        sql = q.get("SQL", "")
        db_id = q.get("db_id", "")
        question = q.get("question", "")

        tables_used = extract_tables_from_sql(sql)

        # 如果 SQL 解析出的表为空，用 tables.json 中的表名做 fallback
        if not tables_used and db_id in table_map:
            # 简单匹配：SQL 中出现哪个表名就纳入
            sql_lower = sql.lower()
            for t in table_map[db_id]:
                if t.lower() in sql_lower:
                    tables_used.append(t)

        exp = SQLExperience(
            question=question,
            sql=sql,
            db_id=db_id,
            tables_used=tables_used,
            success=True,
            iteration_count=1,
        )
        # 不走去重逻辑，直接插入
        store._insert(exp)

        query_meta.append({
            "question_id": q.get("question_id", f"q{i:03d}"),
            "question": question,
            "db_id": db_id,
            "tables_used": tables_used,
            "gold_sql": sql,
            "difficulty": q.get("difficulty", ""),
        })

    # 保存
    store_path = os.path.join(output_dir, "eval_store.json")
    queries_path = os.path.join(output_dir, "eval_queries.json")

    store.save(store_path)
    with open(queries_path, 'w', encoding='utf-8') as f:
        json.dump(query_meta, f, ensure_ascii=False, indent=2)

    # 统计
    db_counts = {}
    for m in query_meta:
        db_counts[m["db_id"]] = db_counts.get(m["db_id"], 0) + 1

    print(f"Built eval set: {len(query_meta)} questions from {len(db_counts)} databases")
    for db_id, count in sorted(db_counts.items()):
        print(f"  {db_id}: {count} questions")
    print(f"Store saved to: {store_path}")
    print(f"Queries saved to: {queries_path}")


if __name__ == "__main__":
    main()
