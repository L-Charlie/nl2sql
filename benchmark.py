# nl2sql_agent/benchmark.py
"""BIRD Mini-Dev benchmark runner for agent_team pipeline.

Uses BIRD SQLite databases with complete ground truth.

Supports 10-way concurrency via ThreadPoolExecutor (LLM calls are I/O-bound).

Usage:
  cd nl2sql_agent
  python benchmark.py \
    --data-dir ../_archive/benchmarks/bird/minidev/MINIDEV \
    --limit 50 \
    --seed 42 \
    --workers 10
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from openai import OpenAI

from agent_team.executor import SQLiteReadOnlyExecutor
from agent_team.orchestrator import Orchestrator


# ─── BIRD schema loading ────────────────────────────────────────────────────────

def _convert_bird_schema(tables_entry: dict) -> list[dict]:
    """Convert a BIRD tables.json entry to agent_team schema format."""
    table_names = tables_entry.get("table_names_original", [])
    column_names = tables_entry.get("column_names_original", [])
    column_types = tables_entry.get("column_types", [])
    raw_pks = tables_entry.get("primary_keys", [])
    primary_keys = set()
    for pk in raw_pks:
        if isinstance(pk, list):
            primary_keys.update(pk)
        else:
            primary_keys.add(pk)
    foreign_key_markers: dict[int, list[str]] = {}
    foreign_keys_by_table: dict[int, list[dict]] = {
        i: [] for i in range(len(table_names))
    }
    for left, right in tables_entry.get("foreign_keys", []):
        if left >= len(column_names) or right >= len(column_names):
            continue
        source_table_idx, source_column = column_names[left]
        target_table_idx, target_column = column_names[right]
        if source_table_idx < 0 or target_table_idx < 0:
            continue
        target_table = table_names[target_table_idx]
        foreign_key_markers.setdefault(left, []).append(
            f"FK->{target_table}.{target_column}"
        )
        foreign_keys_by_table[source_table_idx].append({
            "from": source_column,
            "to_table": target_table,
            "to_column": target_column,
        })

    by_table: dict[int, list[dict]] = {i: [] for i in range(len(table_names))}
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx < 0:
            continue
        markers = []
        if col_idx in primary_keys:
            markers.append("PK")
        for fk_note in foreign_key_markers.get(col_idx, []):
            markers.append(fk_note)
        desc = " ".join(markers) if markers else ""
        col_type = column_types[col_idx] if col_idx < len(column_types) else "text"
        by_table[table_idx].append({
            "col": col_name,
            "type": col_type.upper(),
            "description": desc,
        })

    return [
        {
            "table_name": table_names[i],
            "table_description": "",
            "columns": by_table.get(i, []),
            "foreign_keys": foreign_keys_by_table.get(i, []),
        }
        for i in range(len(table_names))
        if by_table.get(i)
    ]


def load_bird_schema_map(tables_path: str, db_dir: str = "") -> dict[str, list[dict]]:
    """Load BIRD tables.json and return {db_id: agent_team_schema} mapping."""
    with open(tables_path, "r", encoding="utf-8") as f:
        raw_tables = json.load(f)

    result = {}
    for entry in raw_tables:
        db_id = entry["db_id"]
        col_descriptions = {}
        if db_dir:
            desc_dir = os.path.join(db_dir, db_id, "database_description")
            if os.path.isdir(desc_dir):
                for csv_file in sorted(os.listdir(desc_dir)):
                    if not csv_file.endswith('.csv'):
                        continue
                    csv_path = os.path.join(desc_dir, csv_file)
                    table_name = os.path.splitext(csv_file)[0]
                    try:
                        with open(csv_path, 'r', encoding='utf-8') as fh:
                            reader = csv.reader(fh)
                            header = next(reader, [])
                            for row in reader:
                                if len(row) < 2:
                                    continue
                                orig_name = row[0].strip() if len(row) > 0 else ""
                                col_name = row[1].strip() if len(row) > 1 else ""
                                desc = row[2].strip() if len(row) > 2 else ""
                                values = row[4].strip() if len(row) > 4 else ""
                                name = orig_name or col_name
                                if not name:
                                    continue
                                parts = []
                                if desc:
                                    parts.append(desc)
                                if values and "Commonsense evidence" not in values and "Normal range" not in values:
                                    parts.append(f"Values: {values}")
                                if parts:
                                    col_descriptions[(table_name, name)] = "; ".join(parts)
                    except Exception:
                        pass

        schema = _convert_bird_schema(entry)
        for table in schema:
            tname = table["table_name"]
            for col in table["columns"]:
                key = (tname, col["col"])
                csv_desc = col_descriptions.get(key, "")
                if csv_desc:
                    existing = col.get("description", "")
                    col["description"] = f"{existing}; {csv_desc}" if existing else csv_desc

        result[db_id] = schema

    return result


# ─── Helpers ───────────────────────────────────────────────────────────────────

def normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


def strip_sql_block(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def execute_sqlite(sql: str, db_path: str) -> dict:
    """Execute a final benchmark query in explicit full, read-only mode."""
    if not sql or not sql.strip():
        return {"ok": False, "error": "Empty SQL", "rows": [], "row_count": 0}
    return SQLiteReadOnlyExecutor(db_path).execute(sql, mode="full")


def safe_str(s, maxlen=120):
    if s is None:
        return ""
    s = str(s)[:maxlen]
    return s.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
        sys.stdout.encoding or "utf-8", errors="replace"
    )


def results_equal(rows_a: list[dict], rows_b: list[dict]) -> bool:
    """Compare two result sets — order-independent, position-based value comparison."""
    if len(rows_a) != len(rows_b):
        return False
    if not rows_a:
        return True

    def normalize_rows(rows):
        normalized = []
        for r in rows:
            t = tuple(
                str(round(float(v), 6)) if _is_float(v) else str(v)
                for v in r.values()
            )
            normalized.append(t)
        return sorted(normalized)

    return normalize_rows(rows_a) == normalize_rows(rows_b)


def _is_float(val) -> bool:
    try:
        float(val)
        return "." in str(val)
    except (ValueError, TypeError):
        return False


# ─── BIRD-specific ─────────────────────────────────────────────────────────────

def load_bird_questions(json_path: str) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Single-question runner ────────────────────────────────────────────────────

def run_one_question(
    q: dict,
    idx: int,
    total: int,
    schema_map: dict[str, list[dict]],
    db_dir: str,
    model: str,
    model_client: OpenAI,
    orchestrators: dict[str, Orchestrator],
    orch_lock,
) -> dict:
    """Run a single BIRD question through the agent_team pipeline (thread-safe).

    Shares a single model_client across all Orchestrator instances.
    The shared orchestrators dict is protected by orch_lock.
    """
    qid = q["question_id"]
    db_id = q["db_id"]
    question = q["question"]
    evidence = q.get("evidence", "")
    gold_sql = q["SQL"]
    difficulty = q["difficulty"]

    db_path = str(Path(db_dir) / db_id / f"{db_id}.sqlite")
    if not os.path.exists(db_path):
        return {"question_id": qid, "db_id": db_id, "question": question[:200],
                "difficulty": difficulty, "match": False, "error": f"DB not found: {db_path}",
                "elapsed_sec": 0, "iterations": 0, "pred_sql": "", "gold_sql": gold_sql,
                "pred_exec_ok": False, "gold_exec_ok": True, "pred_rows": 0, "gold_rows": 0}

    if db_id not in schema_map:
        return {"question_id": qid, "db_id": db_id, "question": question[:200],
                "difficulty": difficulty, "match": False, "error": f"Schema not found for {db_id}",
                "elapsed_sec": 0, "iterations": 0, "pred_sql": "", "gold_sql": gold_sql,
                "pred_exec_ok": False, "gold_exec_ok": True, "pred_rows": 0, "gold_rows": 0}

    schema = schema_map[db_id]

    print(f"[{idx}/{total}] Q{qid} | {db_id} | {difficulty} | {question[:80]}")

    start = time.time()
    try:
        # Get or create Orchestrator for this db_id (thread-safe)
        with orch_lock:
            if db_id not in orchestrators:
                orchestrators[db_id] = Orchestrator(
                    schema=schema, model=model, model_client=model_client, domain="generic",
                )
            orchestator = orchestrators[db_id]

        result = orchestator.run(
            question,
            knowledge=evidence,
            db_path=db_path,
            database_executor=SQLiteReadOnlyExecutor(db_path),
        )
        elapsed = time.time() - start

        pred_sql = strip_sql_block((result.sql or "").strip())
        pred_exec = execute_sqlite(pred_sql, db_path)
        gold_exec = execute_sqlite(gold_sql, db_path)

        if pred_exec["ok"] and gold_exec["ok"]:
            match = results_equal(pred_exec["rows"], gold_exec["rows"])
        elif not pred_exec["ok"] and not gold_exec["ok"]:
            match = (pred_exec["error"] == gold_exec["error"])
        else:
            match = False

        status = "PASS" if match else "FAIL"
        print(f"  {status} | pred_exec={pred_exec['ok']} gold_exec={gold_exec['ok']} | "
              f"pred_rows={pred_exec['row_count']} gold_rows={gold_exec['row_count']} | "
              f"time={elapsed:.1f}s iter={result.iterations}")
        if not match:
            print(f"  Pred: {pred_sql[:150]}")
            print(f"  Gold: {gold_sql[:150]}")
            if pred_exec["error"]:
                print(f"  Pred error: {pred_exec['error'][:100]}")
            if gold_exec["error"]:
                print(f"  Gold error: {gold_exec['error'][:100]}")

        return {
            "question_id": qid, "db_id": db_id, "question": question[:200],
            "difficulty": difficulty, "evidence": evidence,
            "pred_sql": pred_sql, "gold_sql": gold_sql,
            "match": match, "pred_exec_ok": pred_exec["ok"], "gold_exec_ok": gold_exec["ok"],
            "pred_rows": pred_exec["row_count"], "gold_rows": gold_exec["row_count"],
            "elapsed_sec": round(elapsed, 1), "iterations": result.iterations,
            "error": result.error or "",
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e}")
        return {
            "question_id": qid, "db_id": db_id, "question": question[:200],
            "difficulty": difficulty, "pred_sql": "", "gold_sql": gold_sql,
            "match": False, "pred_exec_ok": False, "gold_exec_ok": True,
            "error": str(e), "elapsed_sec": round(elapsed, 1),
            "iterations": 0, "evidence": evidence,
            "pred_rows": 0, "gold_rows": 0,
        }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load .env for API keys
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    parser = argparse.ArgumentParser(description="BIRD Mini-Dev benchmark")
    parser.add_argument("--data-dir", required=True, help="Path to MINIDEV directory")
    parser.add_argument("--output-dir", default="benchmark_output")
    parser.add_argument("--limit", type=int, default=50, help="Max questions to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for question selection")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent worker threads")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_API_BASE", "https://api.deepseek.com"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--selection", choices=["head", "random", "hard"], default="head",
                        help="head = first N, random = random N, hard = challenging first")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    questions_file = data_dir / "mini_dev_sqlite.json"
    tables_file = data_dir / "dev_tables_fixed.json"
    if not tables_file.exists():
        tables_file = data_dir / "dev_tables.json"
    db_dir = data_dir / "dev_databases"

    # Load BIRD questions and schema
    questions = load_bird_questions(str(questions_file))
    schema_map = load_bird_schema_map(str(tables_file))

    # Select subset
    import random
    random.seed(args.seed)
    if args.selection == "random":
        selected = random.sample(questions, min(args.limit, len(questions)))
    elif args.selection == "hard":
        hard = [q for q in questions if q["difficulty"] == "challenging"]
        others = [q for q in questions if q["difficulty"] != "challenging"]
        selected = hard[:args.limit] if len(hard) >= args.limit else hard + others[:args.limit - len(hard)]
    else:
        selected = questions[:args.limit]

    print(f"BIRD Mini-Dev SQLite Benchmark")
    print(f"  Selected: {len(selected)}/{len(questions)} (mode={args.selection}, seed={args.seed})")
    diffs = Counter(q["difficulty"] for q in selected)
    print(f"  Difficulty: {dict(diffs)}")
    print(f"  Model: {args.model}")
    print(f"  Workers: {args.workers}")

    # Single shared model_client — OpenAI client is thread-safe
    model_client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared state for concurrent execution
    import threading
    orchestrators: dict[str, Orchestrator] = {}
    orch_lock = threading.Lock()

    # Run all questions concurrently
    results = []
    passed = 0
    exec_fail = 0
    pred_exec_ok = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_one_question,
                q, idx, len(selected),
                schema_map, str(db_dir),
                args.model, model_client,
                orchestrators, orch_lock,
            ): q
            for idx, q in enumerate(selected, start=1)
        }

        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            if r.get("match"):
                passed += 1
            if not r.get("pred_exec_ok"):
                exec_fail += 1
            else:
                pred_exec_ok += 1

    # Sort results by original order (question_id)
    results.sort(key=lambda r: r.get("question_id", 0))

    # Summary
    total = len(results)
    acc = passed / total * 100 if total > 0 else 0
    print(f"\n{'='*70}")
    print(f"BIRD Mini-Dev Benchmark Summary")
    print(f"{'='*70}")

    # By difficulty
    by_diff = {}
    for r in results:
        d = r.get("difficulty", "unknown")
        by_diff.setdefault(d, {"total": 0, "passed": 0})
        by_diff[d]["total"] += 1
        if r["match"]:
            by_diff[d]["passed"] += 1

    print(f"{'Difficulty':<15} {'Count':>6} {'Passed':>6} {'Accuracy':>10}")
    print(f"{'-'*37}")
    for d in ["simple", "moderate", "challenging"]:
        if d in by_diff:
            b = by_diff[d]
            print(f"{d:<15} {b['total']:>6} {b['passed']:>6} {b['passed']/b['total']*100:>9.1f}%")
    print(f"{'-'*37}")
    print(f"{'Total':<15} {total:>6} {passed:>6} {acc:>9.1f}%")
    print(f"\n  Pred exec OK: {pred_exec_ok}/{total}  Pred exec fail: {exec_fail}/{total}")

    # Save
    summary = output_dir / "bird_results.json"
    with open(summary, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "model": args.model, "limit": len(selected),
                "selection": args.selection, "seed": args.seed,
                "workers": args.workers,
            },
            "total": total, "passed": passed, "accuracy": round(acc, 1),
            "by_difficulty": {
                d: {**b, "accuracy": round(b["passed"]/b["total"]*100, 1)}
                for d, b in by_diff.items()
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {summary}")


if __name__ == "__main__":
    main()
