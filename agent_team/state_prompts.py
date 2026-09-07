"""State contracts injected into the system messages that consume them.

Keep these meanings aligned with contracts.py, executor.py, pre_execution_gate.py,
and NL2SQLJudge's deterministic/structured feedback builders.
"""

EXECUTION_STATE_PROMPT = """
## 执行状态契约（程序事实）
- 以下是字段定义，不代表本轮实际取值；以输入提供的值为准。字段缺失或 null 表示未知/未提供，不等于 false，也不能推断成功。
- gate.passed=true：执行前检查无阻断项，仅允许进入执行器，不保证执行成功或语义正确。false：被 Gate 拦截。
- gate.blockers：阻断执行的问题；gate.warnings：非阻断观察，不能单凭 warning 声称执行被拒绝。code 是机器标识，message/details 是具体原因与证据。
- stage=gate、attempted=false、ok=false：执行前被拦截，数据库执行器未被调用，不存在本次执行结果。
- stage=database、attempted=true：已调用数据库执行器；ok=true 表示执行器报告成功，ok=false 表示执行器报告失败（含执行模式等错误），不保证 SQL 已在数据库内运行。
- ok=true 不等于 Judge 的 pass=true，也不证明 SQL 回答了原问题。失败/未执行时的空样本或 row_count=0 不能当作成功查询返回零行。
- mode=probe：有限取样；mode=full：请求完整结果。sample_rows 是预览，rows 在 probe 模式下可为空，不能据此认定查询无结果。
- truncated=true：结果被截断；row_count_exact=true 才能将 row_count 视为精确行数。false/null/缺失时不可把 row_count 当总行数；mode=full 本身不能替代精确性标记。
- result_profile 是已获取数据的统计，sample_size/sample_null_count/sample_value_types 不代表全表统计。展示的前几行也不代表全部结果。
- read_only_enforced=true：执行器报告强制只读；false：未提供强制只读保证，不表示发生了写入。不要凭 SQL 文本或 Gate 通过推断此值。
- error_code/error 说明失败原因；空错误文本不是执行成功的证据。
"""

GATE_CODE_PROMPT = """
## Gate 与执行错误码
- config.unsupported_dialect：方言不受支持，不能通过改 SQL 修复配置问题。
- syntax.empty_sql：SQL 为空；syntax.parse_error：解析失败。
- structure.multiple_statements：解析得到的语句数不等于 1；structure.unresolved_placeholder：模板占位符未解析。
- safety.non_read_only：不是允许的单条只读查询或包含写操作。
- schema.table_not_authorized / schema.column_not_authorized：引用的表/列不在授权 Schema 中。
- schema.outside_retrieval：使用了授权但不在本次检索结果中的表，是 warning，不等于未授权。
- schema.ambiguous_unqualified_column：未限定表名的列有多个可能归属，是 warning。
- aggregation.grain_observation：非聚合输出未出现在 GROUP BY，是 warning，应结合问题检查粒度，不能解释成 Gate 阻断。
- execution.mode_not_supported：执行模式不受支持；database.execution_error：数据库执行异常，结合 error 判断具体原因。
- gate.blocked：Gate 未提供具体 blocker 时的兜底拦截码。execution.status：执行状态核查；result.schema：SELECT 预期输出列与实际结果列的名称及顺序核查，失败时结合 expected/actual 修复。
- 未列出的 code 必须结合 message/details/evidence 理解；不能编造其含义。
"""

INTENT_STATE_PROMPT = """
## 意图核查状态契约
- intent_comparison.available=false：没有可用的结构化意图比对；match=null/缺失：未确定，不能当作通过或已证实不匹配。
- available=true 只说明比对结构可用，不表示匹配。deterministic_checks.result_schema.matches=true/false/null 分别表示输出列匹配/不匹配/未进行可比核查。
- match=true/false：SQL 意图与原问题匹配/不匹配。draft_intent_match 描述草案相关比对，必须结合三份意图及证据解释，不能代替 match。
- diagnosis=contract_issue：草案与 SQL 一致但共同偏离原问题，应同时修正草案和 SQL。DraftIntent 是可质疑草案，原问题才是最终依据。
- mismatches 中 expected 是期望语义，actual 是当前语义，component 定位差异；severity=critical 是关键缺陷，major 是重要差异，不能忽略或仅靠高相似度覆盖。
- semantic.intent_contract_missing：Judge 未返回要求的结构化意图比对，不等于已证实 SQL 语义错误；semantic.contract_issue：草案和 SQL 共同偏离原问题；semantic.<component>.<序号>：对应组件的语义差异。
- checks 的 v=YES/NO 表示该项核查满足/不满足，e 是证据，f 是修复建议；不是 SQL 执行状态。
- semantic_similarity/semantic_score 是辅助语义相似度信号，不是确定性正确性证明；未进行语义评估时的零值不能视为真实相似度评分。
"""

REPAIR_STATE_PROMPT = """
## Judge 与修复控制状态契约
- judge_mode / failure_stage=gate：执行前拦截；database：执行器失败；semantic：执行成功后进行质量核查，仍可能存在确定性结果结构问题。gate/database 模式由程序直接构造报告，没有进行 LLM 语义评估。
- pass=true 是程序综合判断的通过，pass=false 表示未通过；overall_confidence 不能覆盖确定性阻断或意图不匹配。
- structured_feedback.retryable=true：允许在剩余预算内尝试修复，不保证能修复；false：当前反馈不允许 SQL 重试。预算和路由由程序控制，模型不能自行更改。
- issues 中 code/category 定位问题，severity 表示严重度，message/evidence/suggestion 分别是描述、证据、建议。deterministic_checks 中 blocking=true 或 blocking_issues 表示确定性阻断，不能通过主观评分撤销。
- deterministic_checks.checks 中 passed=false 也可能只是 warning；须结合 blocking 判断是否阻断，不能把它与 GateResult.passed 或最终 pass 混为一谈。
- schema_search.required=true 仅表示建议补检索，不表示已经补表；query_terms/suggested_tables 是检索线索，不是访问授权。max_new_tables 是补表上限。
- schema_retrieval.requested：请求了补检索；triggered：检索回调成功返回，不保证找到新表；added_tables：实际新增表。skipped_reason=reschema_budget_exhausted 表示预算耗尽，schema_retriever_unavailable 表示没有检索器；error 表示检索异常。只能使用本轮实际提供的 Schema。
- regeneration.mode=sql_repair：在当前 Schema 内修复；restricted_reschema：受限补检索后修复。preserve_correct_parts=true 要求保留正确部分，original_question_is_authoritative=true 要求以原问题为准。
- repair_route=gate_sql_repair/database_sql_repair/sql_repair 分别是 Gate/执行错误/语义修复；l2_reschema 表示进入补检索分支，不保证补表成功；unsupported 表示不支持的修复阶段。
- action=fixed 由程序在获得不同的候选 SQL 后设置，只表示产生修复候选，必须重新通过 Gate、执行和 Judge；give_up 表示没有可用修复，error 表示修复异常。不要输出 action 代替本次要求的 SQLArtifact JSON。
"""
