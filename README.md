# NL2SQL Multi-Agent Team

Multi-agent NL2SQL pipeline with deterministic pre-execution safety and a bounded Judge/Refiner loop.

## Architecture

```
SchemaLinker → DataSampler → Builder(SQLArtifact) → SafeExecutor(Gate → read-only DB) → Judge → Refiner
```

### Agents

| Agent | Type | Role |
|-------|------|------|
| SchemaLinker | BM25 + scored graph rules | Table/column retrieval, evidence-graded FK expansion |
| DataSampler | Zero-LLM | Column probing against live databases |
| Builder | LLM | Generate a challengeable DraftIntent and one SQL query |
| SafeExecutor | Deterministic | Gate every SQL, then run it through a read-only database executor |
| Judge | Deterministic + LLM | Skip LLM on Gate/DB failures; run semantic checks only after execution succeeds |
| Refiner | LLM | Stage-aware Gate, database, or semantic repair with bounded re-schema |

Every Builder or Refiner SQL is checked again. Gate blocks only high-certainty safety and structural failures: non-read-only or multiple statements, parse failures, unresolved placeholders, and references outside the full authorized schema. Aggregation, joins, filters, time range, sorting, and Top-N are recorded as SQL signature data for Judge rather than Gate blockers.

## Trace Example: "How many pets does Alice have?"

Full pipeline trace for a single question through all 6 stages:

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                   │
│  Question: "How many pets does Alice have?"                     │
│  Schema: students (id, name, age) + pets (id, student_id, pet)  │
│  DB: SQLite                                                     │
└────────────────────────────────┬────────────────────────────────┘
                                 │
         ┌───────────────────────▼───────────────────────┐
         │ ① SchemaLinker (4ms) — Zero-LLM               │
         │   BM25 + Embedding 检索表/列                    │
         │                                                │
         │  • pets     → score 1.0  (强匹配 "pet")        │
         │  • students → score 0.68 (通过 FK 关系带入)     │
         │                                                │
         │  输出: 2 张候选表, 6 列,                        │
         │       Join Path: pets.student_id ──▶ students.id│
         │       FK 推断置信度: 0.80；扩展分: 1×.80×.85   │
         └───────────────────────┬───────────────────────┘
                                 │
         ┌───────────────────────▼───────────────────────┐
         │ ② DataSampler (4ms) — Zero-LLM                │
         │   对候选表做列探查 (column probing)            │
         │                                                │
         │  • pets:     抽样数据验证列类型                  │
         │  • students: 抽样数据验证列类型                  │
         │                                                │
         │  输出: 2 张表通过采样验证                       │
         └───────────────────────┬───────────────────────┘
                                 │
         ┌───────────────────────▼───────────────────────┐
         │ ③ RAG Retrieval (334ms) — Zero-LLM            │
         │   检索历史相似经验 (sql_experiences.json)       │
         │                                                │
         │  命中 3 条经验:                                 │
         │  1. "How many pets does Alice have?" ✅ (精确)  │
         │  2. "How many flights does JetBlue have?" ✅    │
         │  3. "How many likes does Kyle have?" ✅         │
         │                                                │
         │  输出: 3 条成功经验注入 Builder prompt          │
         └───────────────────────┬───────────────────────┘
                                 │
         ┌───────────────────────▼───────────────────────┐
         │ ④ Builder (10.1s) — LLM (deepseek-v4-pro)     │
         │   组装 Prompt → 生成 SQL                        │
         │                                                │
         │  Prompt 组成 (~2008 tokens):                    │
         │  • System: builder_prompt_generic.txt          │
         │  • Schema DDL (来自 ①②)                        │
         │  • RAG 范例 (来自 ③)                            │
         │  • Question + Instructions                     │
         │                                                │
         │  输出 (25 tokens):                              │
         │  SELECT count(*)                               │
         │  FROM pets                                     │
         │  JOIN students                                 │
         │    ON pets.student_id = students.id            │
         │  WHERE students.name = 'Alice';                │
         └───────────────────────┬───────────────────────┘
                                 │
         ┌───────────────────────▼───────────────────────┐
         │ ⑤ SafeExecutor — Gate + SQLite probe           │
         │   单语句/只读/授权 Schema 检查                   │
         │   URI mode=ro + PRAGMA query_only              │
         │  返回: row_count=1, exact=true, ok=true         │
         │  (Alice 有 1 只宠物: Fluffy)                    │
         └───────────────────────┬───────────────────────┘
                                 │
         ┌───────────────────────▼───────────────────────┐
         │ ⑥ Judge (74.3s) — LLM (deepseek-v4-pro)       │
         │   多维度评估 SQL 质量                            │
         │                                                │
         │  ┌──────────────┬──────┬──────────────────┐   │
         │  │ 维度          │ 得分  │ 说明              │   │
         │  ├──────────────┼──────┼──────────────────┤   │
         │  │ syntax       │   7  │ SQL 语法正确       │   │
         │  │ semantics    │  18  │ 语义匹配问题       │   │
         │  │ logic        │  12  │ JOIN/WHERE 逻辑   │   │
         │  │ result_quality│ 13  │ 结果正确性         │   │
         │  └──────────────┴──────┴──────────────────┘   │
         │                                                │
         │  • Critical Flaws: 无                          │
         │  • Reverse Question: "How many pets does       │
         │    Alice have?" → semantic_similarity=1.0      │
         │  • Overall Confidence: 100 → pass_=true        │
         └───────────────────────┬───────────────────────┘
                                 │
         ┌───────────────────────▼───────────────────────┐
         │              Refiner — SKIPPED ✅              │
         │   Judge pass_=true, 无需修复循环                │
         └───────────────────────┬───────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────┐
│                          OUTPUT                                  │
│  SQL: SELECT count(*) FROM pets JOIN students ON ...             │
│  Success: True | Iterations: 1 | Error: None                     │
│  Result: [{'count(*)': 1}]                                       │
└──────────────────────────────────────────────────────────────────┘
```

### Time Distribution

| Stage | Duration | % |
|-------|----------|---|
| SchemaLinker | 4ms | 0.005% |
| DataSampler | 4ms | 0.005% |
| RAG | 334ms | 0.4% |
| Builder (LLM) | 10,104ms | 11.7% |
| Executor | 1ms | 0.001% |
| **Judge (LLM)** | **74,327ms** | **86.3%** |
| **Total** | **86.2s** | 100% |

> Judge is the bottleneck (86% of total time). Builder takes only 12%.

### Domain Switching

| Domain | Prompt | Use Case |
|--------|--------|----------|
| `generic` | `builder_prompt_generic.txt` | Standard SQLite, BIRD benchmark |
| `enterprise` | `builder_prompt_enterprise.txt` | CTE, window functions, recursive CTE |

## Quick Start

```python
from openai import OpenAI
from agent_team.orchestrator import Orchestrator

schema = [
    {"table_name": "customers", "columns": [
        {"col": "id", "type": "INTEGER", "description": "PK"},
        {"col": "name", "type": "TEXT", "description": ""},
    ]},
]

client = OpenAI(base_url="https://api.deepseek.com", api_key="...")
orch = Orchestrator(schema=schema, model="deepseek-v4-pro",
                    model_client=client, domain="generic")

# Without an executor, SQL is generated but deliberately remains unverified.
result = orch.run("How many customers are there?")
print(result.sql)
assert result.status == "generated_unverified"
assert result.success is False

# db_path enables SQLiteReadOnlyExecutor: URI mode=ro + PRAGMA query_only=ON.
# The repair loop uses bounded probe results rather than loading the full result set.
result = orch.run("How many customers?", db_path="mydb.sqlite")

# A legacy execute_sql callback is still accepted through CallbackExecutorAdapter,
# but the callback cannot claim database-level read-only enforcement.
```

### Live Schema Discovery

SQLite 可以通过只读连接直接发现表、列、主键和数据库声明的外键，再把结果交给
`Orchestrator`。`table_names` 可用于把发现范围限制为调用方明确授权的表：

```python
from agent_team.schema_discovery import discover_sqlite_schema

schema = discover_sqlite_schema(
    "mydb.sqlite",
    table_names=["orders", "customers"],
)
orch = Orchestrator(schema=schema, model="deepseek-v4-pro", model_client=client)
```

FK 图谱按证据分级：数据库结构化外键 `1.00`、Schema 中的
`FK->table.column` 标记 `0.95`、满足主键/唯一键和类型条件的
`{entity}_id -> table.id` 推断 `0.80`。普通同名列只记录为 `0.10` 的诊断候选，
不会扩表，也不会作为可执行 JOIN。扩展分为
`种子表归一化 BM25 × 边置信度 × 0.85`，并限制为一跳、每个种子最多两张、
总候选表最多八张；返回值的 `fk_expansion` 字段保留扩展证据和被拒绝数量。

## Benchmark

```bash
# BIRD Mini-Dev
python benchmark.py \
  --data-dir ../_archive/benchmarks/bird/minidev/MINIDEV \
  --limit 50 --seed 42 --workers 10
```

## Results

| Benchmark | Exec Accuracy | Notes |
|-----------|--------------|-------|
| BIRD Mini-Dev (SQLite, 20Q) | **35%** | generic domain, 100% exec OK |

## Directory

```
nl2sql_agent/
├── agent_team/          # Core multi-agent engine
│   ├── orchestrator.py  # Pipeline orchestration
│   ├── builder.py       # LLM SQL generation
│   ├── contracts.py     # SQLArtifact, GateResult, ExecutionAttempt, retry state
│   ├── pre_execution_gate.py # Deterministic safety/schema Gate
│   ├── executor.py      # SafeExecutor and read-only SQLite executor
│   ├── refiner.py       # Stage-aware Gate/DB/semantic repair
│   ├── schema_linker.py # Table/column retrieval
│   ├── data_sampler.py  # Column probing
│   ├── planner.py       # Query planning (optional, large-schema)
│   ├── nl2sql_judge.py  # Multi-dimension SQL judge
│   ├── prompts/         # System prompts per domain
│   ├── data/            # RAG experience store
│   └── memories/        # Agent memory files
├── benchmark.py         # BIRD Mini-Dev benchmark runner
├── scripts/             # Utility scripts
└── tests/               # Test suite
```
