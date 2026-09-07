# nl2sql/agent_team/validation_engine.py
"""
ValidationEngine —— 零大模型（Zero-LLM）的 SQL 验证引擎。

核心目标：
  在不需要 LLM 介入的情况下，通过确定性规则自动验证生成的 SQL 是否符合预期。
  验证结果会被 Refiner（精炼器）用来判断是否需要修改 SQL。

三层验证架构（L1 / L2 / L3）：
  Layer 1 —— Schema 一致性检查（Schema Consistency）
    检查生成的 SQL 中引用的表和列是否确实存在于 relevant_schema 中。
    如果引用了不存在的表或列，说明 SQL 生成有误。

  Layer 2 —— 执行结果检查（Execution Result）
    SQL 执行之后，对返回的结果做一系列质量检查：
      - 执行是否报错（error check）
      - 结果是否为空（zero rows check，给出警告而不是报错）
      - 比例字段是否在 [0, 1] 范围内（ratio range check）
      - 计数类字段是否大于 0（zero value check）

  Layer 3 —— 多候选结果差异检查（Multi-Candidate Divergence）
    如果有两个版本的 SQL 都执行了，比较它们的返回结果：
      - 行数是否一致
      - 列名是否一致
      - 相同行列位置的值是否一致
    如果差异过大，说明两个候选 SQL 的逻辑存在分歧，需要审视。

为什么不需要 LLM？
  这种纯规则的验证方式有以下优点：
    - 确定性：同样的输入永远得到同样的输出
    - 低延迟：不需要调用外部 API，毫秒级完成
    - 可解释：每个检查失败都有明确的描述和详情的错误原因
    - 低成本：没有 token 消耗
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SchemaCheck:
    """
    单个 Schema 一致性检查结果。

    L1 层的检查单元，用于记录一条关于表或列存在性检查的结果。

    属性：
      passed:      检查是否通过
                     True  = 表或列存在
                     False = 引用了不存在的表或列
      description: 检查项的描述（简短的一句话摘要）
      detail:      详细的说明（为什么会失败、有哪些可用的表/列等）

    示例：
      SchemaCheck(
          passed=False,
          description="表 'dws_user_dau' 不在 relevant_schema 中",
          detail="SQL 中引用了表 'dws_user_dau'，但 relevant_schema 的候选表中不存在。可用的表: ['dws_user_act', ...]"
      )
    """
    passed: bool
    description: str
    detail: str = ""


@dataclass
class ResultCheck:
    """
    单个执行结果检查结果。

    L2 层的检查单元，用于记录一条对 SQL 执行结果的质量检查。

    属性：
      passed:      检查是否通过
                     True  = 检查通过（或给出一条警告性通过）
                     False = 检查未通过
      description: 检查项的描述
      detail:      详细的说明

    注意：
      passed=True 可能带有 detail 详情（如"零行"警告），
      这种"通过但需注意"的情况会被标记为 warnings。
    """
    passed: bool
    description: str
    detail: str = ""


@dataclass
class DivergenceCheck:
    """
    单个多候选结果差异检查结果。

    L3 层的检查单元，用于记录两个 SQL 执行结果之间的差异。

    属性：
      passed:      检查是否通过
                     True  = 无差异或差异可接受
                     False = 存在显著差异
      description: 差异的描述
      detail:      详细的差异信息
    """
    passed: bool
    description: str
    detail: str = ""


@dataclass
class LayerReport:
    """
    单个验证层的检查报告。

    每一层（L1/L2/L3）的验证结果都封装在这个对象中。

    属性：
      layer:   层编号（1、2 或 3）
      skipped: 是否被跳过（如果该层的输入数据不可用，则跳过）
      checks:  该层所有的检查结果列表（类型为 SchemaCheck / ResultCheck / DivergenceCheck）

    派生属性：
      is_clean:  该层是否全部通过（无失败项）。跳过也算 clean。
      failures:  该层中所有未通过的检查项列表
      warnings:  该层中 passed=True 但有 detail 的 ResultCheck 列表（仅 L2 有 warning）
    """
    layer: int
    skipped: bool = False
    checks: list = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """该层是否全部通过。跳过的层也算 clean。"""
        if self.skipped:
            return True
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list:
        """获取该层中所有未通过的检查项。"""
        return [c for c in self.checks if not c.passed]

    @property
    def warnings(self) -> list:
        """
        获取该层中"通过但需注意"的检查项。

        目前只对 L2 有意义：
          例如"结果为零行"是 passed=True 但带有 detail 信息，
          它算是一条"可接受的警告"而非"错误"。
        """
        return [c for c in self.checks if isinstance(c, ResultCheck) and c.passed and c.detail]


class ValidationReport:
    """
    跨所有层的聚合验证报告。

    这个对象汇总了 L1、L2、L3 的所有验证结果，提供了便捷的
    属性访问方法来检查整体验证状态。

    使用示例：
      report = ValidationReport()
      report.l1 = l1_result
      report.l2 = l2_result
      report.l3 = l3_result

      if report.is_fully_valid:
          print("所有验证层均通过！")
      elif not report.l1_passed:
          print(f"L1 检查失败: {report.l1.failures}")

    属性的命名约定：
      l1_passed / l1_skipped / l2_passed / l2_skipped / ...
      用于快速检查某层的状态。
    """

    def __init__(self):
        self.l1 = LayerReport(layer=1)
        self.l2 = LayerReport(layer=2)
        self.l3 = LayerReport(layer=3)

    @property
    def l1_passed(self) -> bool:
        return self.l1.is_clean

    @property
    def l1_skipped(self) -> bool:
        return self.l1.skipped

    @property
    def l2_passed(self) -> bool:
        return self.l2.is_clean

    @property
    def l2_skipped(self) -> bool:
        return self.l2.skipped

    @property
    def l3_passed(self) -> bool:
        return self.l3.is_clean

    @property
    def l3_skipped(self) -> bool:
        return self.l3.skipped

    @property
    def is_fully_valid(self) -> bool:
        """所有三层全部通过（或跳过）才算完全有效。"""
        return self.l1.is_clean and self.l2.is_clean and self.l3.is_clean

    def to_dict(self) -> dict:
        """
        将验证报告转换为字典格式，便于序列化（如 JSON 输出）。

        返回结构：
          {
            "l1": { "skipped", "passed", "failure_count", "failures": [...] },
            "l2": { "skipped", "passed", "failure_count", "warning_count",
                    "failures": [...], "warnings": [...] },
            "l3": { "skipped", "passed", "failure_count", "failures": [...] },
            "is_fully_valid": true/false
          }
        """
        return {
            "l1": {
                "skipped": self.l1.skipped,
                "passed": self.l1.is_clean,
                "failure_count": len(self.l1.failures),
                "failures": [{"description": c.description, "detail": c.detail} for c in self.l1.failures],
            },
            "l2": {
                "skipped": self.l2.skipped,
                "passed": self.l2.is_clean,
                "failure_count": len(self.l2.failures),
                "warning_count": len(self.l2.warnings),
                "failures": [{"description": c.description, "detail": c.detail} for c in self.l2.failures],
                "warnings": [{"description": c.description, "detail": c.detail} for c in self.l2.warnings],
            },
            "l3": {
                "skipped": self.l3.skipped,
                "passed": self.l3.is_clean,
                "failure_count": len(self.l3.failures),
                "failures": [{"description": c.description, "detail": c.detail} for c in self.l3.failures],
            },
            "is_fully_valid": self.is_fully_valid,
        }


class ValidationEngine:
    """
    零大模型（Zero-LLM）的 SQL 验证引擎。

    采用三层确定性验证架构，无需 LLM 参与：
      第 1 层：Schema 一致性 —— 验证表和列名的存在性
      第 2 层：执行结果 —— 验证返回数据的质量
      第 3 层：多候选结果差异 —— 比较两个 SQL 版本的输出

    使用方式：
      engine = ValidationEngine()
      report = engine.validate(
          sql="SELECT dtstatdate, dau FROM dws_user_dau",
          relevant_schema={...},       # L1 需要
          exec_result={"status": "ok", "result": [...]},  # L2 需要
          exec_result_alt=[...],       # L3 需要（可选）
          ratio_fields=["dau_rate"],
          count_fields=["dau"],
      )

      if report.is_fully_valid:
          print("验证通过！")
      elif not report.l1_passed:
          # 更新 SQL 中的表/列引用
          ...
      elif not report.l2_passed:
          # 检查执行结果问题
          ...

    设计来源：
      编码自 spider_agent.txt 中的验证规则：
        - L1 规则：确保生成的 SQL 只引用了 relevant_schema 中的实体
        - L2 规则：ZERO_VALUE_HANDLING_STRICT_RULE（第 710-719 行）
                  RESULT_FILTERING_STANDARDS（第 721-744 行）
        - L3 规则：多候选 SQL 的差异比较逻辑
    """

    def __init__(self):
        pass

    # ========================================================================
    # 公共 API
    # ========================================================================

    def validate(
        self,
        sql: str,
        relevant_schema: Optional[dict] = None,
        exec_result: Optional[dict] = None,
        exec_result_alt: Optional[list[dict]] = None,
        ratio_fields: Optional[list[str]] = None,
        count_fields: Optional[list[str]] = None,
    ) -> ValidationReport:
        """
        执行所有可用的验证层。

        这个方法根据提供的输入数据自动判断运行哪些验证层：
          - 如果有 relevant_schema，运行 L1（Schema 一致性检查）
          - 如果有 exec_result，运行 L2（执行结果检查）
          - 如果同时有 exec_result 和 exec_result_alt，运行 L3（差异检查）
          - 如果某层的输入数据缺失，该层被标记为"跳过"

        参数：
          sql:               要验证的 SQL 字符串
          relevant_schema:   SchemaLinker 的输出字典（含 candidate_tables）
          exec_result:       run_sql_direct 的输出字典（含 status 和 result）
          exec_result_alt:   可选的另一个 SQL 执行结果，用于 L3 差异比较
          ratio_fields:      期望值在 [0, 1] 范围内的列名列表
          count_fields:      期望值大于 0 的列名列表

        返回：
          ValidationReport 对象，包含 l1/l2/l3 三层的检查结果。
        """
        report = ValidationReport()

        # ----- Layer 1: Schema 一致性检查 -----
        if relevant_schema:
            l1_report = self.check_schema_consistency(sql, relevant_schema)
            report.l1 = l1_report
        else:
            report.l1.skipped = True

        # ----- Layer 2: 执行结果检查 -----
        if exec_result:
            l2_report = self.check_execution_result(
                exec_result, ratio_fields=ratio_fields, count_fields=count_fields
            )
            report.l2 = l2_report
        else:
            report.l2.skipped = True

        # ----- Layer 3: 多候选结果差异检查 -----
        if exec_result and exec_result_alt:
            result1 = exec_result.get("result", [])
            if isinstance(exec_result_alt, dict):
                result2 = exec_result_alt.get("result", exec_result_alt)
            else:
                result2 = exec_result_alt
            l3_report = self.check_multi_candidate_divergence(result1, result2)
            report.l3 = l3_report
        else:
            report.l3.skipped = True

        return report

    # ========================================================================
    # Layer 1: Schema 一致性检查（Schema Consistency）
    #
    # 目标：验证 SQL 中引用的表和列是否确实存在于 relevant_schema 中。
    #
    # 为什么需要这一层？
    #   生成的 SQL 可能因为幻觉（hallucination）或逻辑错误而引用
    #   不存在的表名或列名。在正式执行之前发现这些错误可以节省大量时间。
    #
    # 实现方式：
    #   使用正则表达式（不是完整的 SQL 解析器）从 SQL 文本中提取
    #   表名和列名，然后与 relevant_schema 中的实体列表做比对。
    #
    # 局限性：
    #   正则解析不是完整的 SQL 解析，可能遗漏某些复杂的 SQL 语法结构。
    #   但对于大多数常见的 SELECT 查询来说已经足够。
    # ========================================================================

    def check_schema_consistency(self, sql: str, relevant_schema: dict) -> LayerReport:
        """
        L1：验证 SQL 中所有表和列在 relevant_schema 中存在。

        检查流程：
          1. 从 relevant_schema 中提取有效的表名和列名集合。
          2. 从 SQL 中提取 CTE（公共表表达式）的表名（这些需要排除检查）。
          3. 从 FROM/JOIN 子句中提取引用的表名。
          4. 从 SELECT 和 WHERE/ON 子句中提取引用的列名。
          5. 逐一检查每个引用是否在有效集合中。

        对 CTE 的特殊处理：
          WITH 子句中定义的临时表名不会被当作"不存在的表"报错。
          例如：
            WITH dau_tmp AS (SELECT ...)
            SELECT * FROM dau_tmp
          这里的 dau_tmp 是 CTE 的临时表名，不是真实存在的表，不应该报错。

        参数：
          sql:             SQL 字符串
          relevant_schema: SchemaLinker 的输出，含 candidate_tables

        返回：
          LayerReport(layer=1)，其中 checks 列表包含所有 SchemaCheck 结果。
        """
        report = LayerReport(layer=1)

        # ====================================================================
        # 第一步：构建有效表名和列名的索引
        # ====================================================================

        valid_tables = set()           # 所有候选表的名称
        valid_columns = {}             # { 表名 -> 该表的列名集合 }
        global_columns = set()         # 所有候选表中所有列名的并集

        for table in relevant_schema.get("candidate_tables", []):
            table_name = table["name"]
            valid_tables.add(table_name)
            cols = {c["name"] for c in table.get("relevant_columns", [])}
            valid_columns[table_name] = cols
            global_columns.update(cols)

        # ====================================================================
        # 第二步：提取 CTE 表名（WITH xxx AS (...)）
        #
        # CTE（Common Table Expression）是在 SQL 中用 WITH 定义的临时表，
        # 只在该 SQL 语句中有效。这些"表名"不需要存在于数据库中，
        # 所以不能把它们当作"不存在的表"来报错。
        # ====================================================================

        cte_names = set()
        cte_pattern = re.compile(r'\bWITH\s+(\w+)\s+AS\s*\(', re.IGNORECASE)
        for match in cte_pattern.finditer(sql):
            cte_names.add(match.group(1).lower())

        # ====================================================================
        # 第三步：从 FROM/JOIN 子句中提取表名
        #
        # 匹配模式：FROM table_name [AS] alias, JOIN table_name [AS] alias
        #
        # 同时处理表的别名：
        #   如果 FROM users u 或 FROM users AS u，
        #   我们把 u 也加入 valid_tables，这样后续对 u.column_name 的检查就能通过。
        # ====================================================================

        table_pattern = re.compile(
            r'(?:FROM|JOIN)\s+(\w+)\s*(?:(?:AS\s+)?(\w+))?\s*',
            re.IGNORECASE
        )

        referenced_tables = set()
        table_aliases = {}  # 别名 -> 原始表名
        for match in table_pattern.finditer(sql):
            table_name = match.group(1)
            alias = match.group(2)
            referenced_tables.add(table_name)
            # 判断捕获到的是否真的是别名（不是 SQL 关键字）
            if alias and alias.upper() not in {
                'ON', 'WHERE', 'AND', 'OR', 'INNER', 'LEFT', 'RIGHT',
                'CROSS', 'FULL', 'OUTER', 'JOIN', 'FROM', 'SELECT'
            }:
                table_aliases[alias] = table_name
                # 把别名也当作"有效表名"，这样后续 t.col 的检查能正常通过
                if table_name in valid_tables:
                    valid_tables.add(alias)
                    if table_name in valid_columns:
                        valid_columns[alias] = valid_columns[table_name]

        # ====================================================================
        # 第四步：检查每个引用的表是否存在
        # ====================================================================

        for table in referenced_tables:
            # CTE 中定义的表名不检查
            if table.lower() in cte_names:
                continue
            if table not in valid_tables:
                # 大小写不敏感再做一轮检查
                found = False
                for vt in valid_tables:
                    if vt.lower() == table.lower():
                        found = True
                        break
                if not found:
                    report.checks.append(SchemaCheck(
                        passed=False,
                        description=f"表 '{table}' 不在 relevant_schema 中",
                        detail=f"SQL 中引用了表 '{table}'，但 relevant_schema 的候选表中不存在。可用的表: {sorted(valid_tables)}",
                    ))

        # ====================================================================
        # 第五步：从 WHERE/ON/JOIN 条件中提取表限定的列名（t.col 格式）
        #
        # 匹配模式：table.column（如 dws_user_dau.dau）
        # ====================================================================

        qualified_col_pattern = re.compile(
            r'(?:ON|WHERE|AND|OR|\(|,)\s*(\w+)\.(\w+)\s*', re.IGNORECASE
        )
        for match in qualified_col_pattern.finditer(sql):
            alias = match.group(1)
            col_name = match.group(2)
            # 检查别名是否匹配一个已知的（或通过别名注册的）表
            matched_table = None
            for vt in valid_tables:
                if alias.lower() == vt.lower():
                    matched_table = vt
                    break
            if matched_table and col_name not in global_columns:
                report.checks.append(SchemaCheck(
                    passed=False,
                    description=f"列 '{col_name}' 不在表 '{matched_table}' 中",
                    detail=f"表 '{matched_table}' 没有列 '{col_name}'。可用列: {sorted(valid_columns.get(matched_table, set()))}",
                ))

        # ====================================================================
        # 第六步：从 SELECT 子句中提取裸列名
        #
        # 这里解析 SELECT 和 FROM 之间的内容，按逗号分割但需要
        # 注意括号嵌套（函数调用的参数列表中的逗号不能作为分隔符）。
        # 对于筛选后的列表达式：
        #   - * 和 func(...) 跳过（不检查函数调用）
        #   - col AS alias 跳过（别名不需要检查）
        #   - t.col 检查最后的列名部分
        #   - 裸列名（如 dau、dtstatdate）检查是否在 global_columns 中
        # ====================================================================

        select_pattern = re.compile(r'SELECT\s+(.*?)\s+FROM', re.IGNORECASE | re.DOTALL)
        select_match = select_pattern.search(sql)
        if select_match:
            select_clause = select_match.group(1)
            # 按逗号分割 SELECT 列，但要尊重括号嵌套
            cols = self._split_select_columns(select_clause)
            for col_expr in cols:
                col_expr = col_expr.strip()
                # 跳过 *（全选）、函数调用（如 SUM(x)）、带别名的表达式
                if col_expr == '*' or '(' in col_expr or ' AS ' in col_expr.upper():
                    continue
                # 如果是 t.col 格式，只取 col 部分
                bare_col = col_expr.split('.')[-1].strip()
                if bare_col and bare_col not in global_columns and not bare_col.isdigit():
                    # 只有符合标识符规则的才检查（避免把字符串文字误判为列名）
                    if re.match(r'^[a-zA-Z_]\w*$', bare_col):
                        if bare_col not in global_columns:
                            report.checks.append(SchemaCheck(
                                passed=False,
                                description=f"列 '{bare_col}' 不在 relevant_schema 中",
                                detail=f"SELECT 中引用了列 '{bare_col}'，但 relevant_schema 的字段中不存在。可用列: {sorted(global_columns)}",
                            ))

        # ====================================================================
        # 第七步：从 WHERE/ON/HAVING 条件中提取列名
        #
        # 匹配模式：WHERE col = value、AND col > 100、ON col IN (...)
        # 注意排除 SQL 关键字本身被误匹配的情况。
        # ====================================================================

        where_col_pattern = re.compile(
            r'(?:WHERE|AND|OR|ON|HAVING)\s+([a-zA-Z_]\w*)(?!\.)\s*(?:=|>|<|>=|<=|!=|LIKE|IN|IS)\s*',
            re.IGNORECASE
        )
        # SQL 关键字列表，用于过滤掉被正则误匹配的关键字
        sql_keywords = {
            'AND', 'OR', 'NOT', 'IN', 'LIKE', 'IS', 'BETWEEN',
            'NULL', 'TRUE', 'FALSE', 'AS', 'ON', 'WHERE', 'HAVING'
        }
        for match in where_col_pattern.finditer(sql):
            col_name = match.group(1)
            if col_name.upper() in sql_keywords:
                continue  # 跳过 SQL 关键字
            if col_name not in global_columns:
                report.checks.append(SchemaCheck(
                    passed=False,
                    description=f"列 '{col_name}' 不在 relevant_schema 中",
                    detail=f"WHERE/ON 条件中引用了列 '{col_name}'，但 relevant_schema 的字段中不存在。可用列: {sorted(global_columns)}",
                ))

        return report

    def _split_select_columns(self, select_clause: str) -> list[str]:
        """
        按逗号分割 SELECT 子句中的列表达式，同时正确保留括号内的内容。

        为什么需要这个方法？
          SELECT 子句中的逗号既可能是列分隔符：
            SELECT col1, col2, col3 FROM ...
          也可能是函数调用参数的分隔符：
            SELECT COALESCE(col1, col2), SUM(col3) FROM ...
          简单使用 str.split(',') 会把函数的参数也错误地分割。

        实现方式：
          遍历字符串中的每个字符，用 depth 变量追踪当前的括号嵌套层级：
            - 遇到 '('  depth += 1
            - 遇到 ')'  depth -= 1
            - 遇到 ',' 且 depth == 0：说明是在括号外，是一个真实的列分隔符

        参数：
          select_clause: SELECT 和 FROM 之间的文本内容

        返回：
          分割后的列表达式列表（每个表达式已被 strip 去除首尾空格）

        示例：
          输入: "a, b + c, MAX(d, e, f)"
          输出: ["a", "b + c", "MAX(d, e, f)"]
        """
        cols = []
        depth = 0
        current = ""
        for ch in select_clause:
            if ch == '(':
                depth += 1
                current += ch
            elif ch == ')':
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                # 只有不在括号内的逗号才是真正的列分隔符
                cols.append(current.strip())
                current = ""
            else:
                current += ch
        if current.strip():
            cols.append(current.strip())
        return cols

    # ========================================================================
    # Layer 2: 执行结果验证（Execution Result）
    #
    # 目标：检验 SQL 执行后返回的数据是否符合质量要求。
    #
    # 检查项：
    #   1. 执行状态：SQL 是否成功执行（没有语法错误或运行时错误）
    #   2. 结果格式：返回的结果是否是一个合法的列表
    #   3. 零行检查：结果为空是一般是正常的，但给出警告提醒
    #   4. 比例范围：比例字段的值应该在 [0, 1] 范围内
    #      如果 > 1，说明分子/分母可能颠倒了
    #   5. 计数值检查：计数类字段应该 > 0
    #      如果为 0，可能没有过滤掉无意义的记录
    #
    # 规则来源：
    #   spider_agent.txt 中的：
    #     - ZERO_VALUE_HANDLING_STRICT_RULE（零值处理铁律，第 710-719 行）
    #     - RESULT_FILTERING_STANDARDS（结果过滤标准，第 721-744 行）
    # ========================================================================

    def check_execution_result(
        self,
        exec_result: dict,
        ratio_fields: Optional[list[str]] = None,
        count_fields: Optional[list[str]] = None,
    ) -> LayerReport:
        """
        L2：验证 SQL 的执行结果。

        参数：
          exec_result: SQL 执行结果的字典，格式：
            {
              "status": "ok" 或 "error",
              "result": [{"col1": val1, "col2": val2}, ...],  # 成功时有
              "error_message": "..."  # 失败时有
            }
          ratio_fields: 比例字段的列名列表，这些字段的值应该在 [0, 1] 之间。
                       例如 ["dau_rate", "pay_rate"]。
          count_fields: 计数类字段的列名列表，这些字段的值应该 > 0。
                       例如 ["dau", "pay_count", "new_users"]。

        返回：
          LayerReport(layer=2)，其中 checks 列表包含所有 ResultCheck 结果。
        """
        report = LayerReport(layer=1)
        ratio_fields = ratio_fields or []
        count_fields = count_fields or []

        # ====================================================================
        # 检查 1：SQL 执行状态
        # 如果执行失败（语法错误、表不存在等），直接返回错误。
        # ====================================================================
        if exec_result.get("status") == "error":
            report.checks.append(ResultCheck(
                passed=False,
                description="SQL execution error: 执行失败",
                detail=exec_result.get("error_message", "Unknown error"),
            ))
            return report

        # ====================================================================
        # 检查 2：结果格式
        # 确保 result 是一个列表（行列表，每行是一个字段 -> 值的字典）。
        # ====================================================================
        rows = exec_result.get("result", [])
        if not isinstance(rows, list):
            report.checks.append(ResultCheck(
                passed=False,
                description="执行结果格式异常",
                detail=f"预期 list，实际 {type(rows).__name__}",
            ))
            return report

        # ====================================================================
        # 检查 3：零行结果（Zero Rows）
        #
        # 如果 SQL 执行成功但返回 0 行数据，这可能是正确的
        # （比如查询的时间范围内确实没有数据），也可能是 SQL 逻辑错误
        # （比如 JOIN 条件写错了导致没有匹配行）。
        #
        # 所以这里不判定为失败（passed=True），但给出详细说明，
        # 让后续流程（人工或 Refiner）来判断。
        # ====================================================================
        if len(rows) == 0:
            report.checks.append(ResultCheck(
                passed=True,
                description="结果为零行",
                detail="SQL 执行成功但返回零行。可能是正确的（目标时间无数据），也可能是 join/过滤条件错误。需要人工判定。",
            ))
            return report

        # ====================================================================
        # 检查 4：比例字段范围检查（Ratio Range Check）
        #
        # 比例字段的值应该在 [0, 1] 范围内。
        # 例如付费率 = 付费用户数 / 活跃用户数，结果应该在 0 到 1 之间。
        #
        # 如果某个比例字段的值 > 1.0，很可能是分子/分母搞反了。
        # 比如用 活跃用户数 / 付费用户数，结果可能远大于 1。
        # ====================================================================
        for rf in ratio_fields:
            if rows and rf in rows[0]:
                for i, row in enumerate(rows):
                    val = row.get(rf)
                    if val is not None and isinstance(val, (int, float)) and val > 1.0:
                        report.checks.append(ResultCheck(
                            passed=False,
                            description=f"比例字段 '{rf}' 在第 {i+1} 行的值为 {val}，超过 1.0",
                            detail=f"比例字段应落在 [0, 1] 区间。值 > 1 表明分子/分母逻辑可能颠倒。",
                        ))
                        break  # 每个字段只报第一个错误

        # ====================================================================
        # 检查 5：计数字段零值检查（Zero Value Check）
        #
        # 有些计数字段的值为 0 是没有意义的（或者代表数据异常），
        # 例如 "人数 = 0" 通常意味着没有正确过滤数据。
        #
        # 根据 spider_agent.txt 零值处理铁律：
        #   统计类问题中，必须过滤掉 total_xxx = 0 的记录。
        # ====================================================================
        for cf in count_fields:
            if rows and cf in rows[0]:
                for i, row in enumerate(rows):
                    val = row.get(cf)
                    if val is not None and isinstance(val, (int, float)) and val <= 0:
                        report.checks.append(ResultCheck(
                            passed=False,
                            description=f"统计字段 '{cf}' 在第 {i+1} 行的值为 {val}，应大于 0",
                            detail=f"spider_agent.txt 零值处理铁律：统计类问题必须过滤 total_{cf} = 0 的记录。",
                        ))
                        break  # 每个字段只报第一个错误

        return report

    # ========================================================================
    # Layer 3: 多候选结果差异检查（Multi-Candidate Divergence）
    #
    # 目标：比较两个 SQL 版本的执行结果，发现显著差异。
    #
    # 为什么需要比较两个结果？
    #   有时系统会生成多个候选 SQL（如使用不同 JOIN 路径或不同的聚合方式），
    #   然后分别执行。如果两个结果存在显著差异，说明至少有一个版本有问题。
    #
    # 检查维度：
    #   1. 行数是否一致
    #   2. 列名是否一致
    #   3. 相同行列位置的值是否一致
    #
    # 注意：
    #   L3 只做结构性的差异检查，不判断哪个结果是正确的。
    #   存在差异只是"需要关注"的信号，具体决策由 Refiner 来做。
    # ========================================================================

    def check_multi_candidate_divergence(
        self,
        result1: list[dict],
        result2: list[dict],
    ) -> LayerReport:
        """
        L3：比较两个 SQL 执行结果的差异。

        参数：
          result1: 第一个 SQL 的执行结果，行列表（每行是一个字典）
          result2: 第二个 SQL 的执行结果，行列表（每行是一个字典）

        返回：
          LayerReport(layer=3)，其中 checks 列表包含所有 DivergenceCheck 结果。
        """
        report = LayerReport(layer=3)

        # 检查 0：两个结果都是空 -> 一致（通过）
        if not result1 and not result2:
            report.checks.append(DivergenceCheck(
                passed=True, description="两个结果均为空", detail=""
            ))
            return report

        # ====================================================================
        # 差异检查 1：行数差异
        # 例如 result1 返回了 10 行，result2 返回了 5 行，
        # 说明两个 SQL 的过滤条件或 JOIN 逻辑不同。
        # ====================================================================
        if len(result1) != len(result2):
            report.checks.append(DivergenceCheck(
                passed=False,
                description=f"行数不一致: result1={len(result1)}, result2={len(result2)}",
                detail="两个候选 SQL 返回不同的行数，可能 join 或过滤逻辑不同。",
            ))

        # ====================================================================
        # 差异检查 2：列名差异
        # 检查两个结果的列集合是否相同。
        # 例如 result1 有 [date, dau]，result2 有 [date, wau]，
        # 说明两个 SQL SELECT 的列不同。
        # ====================================================================
        cols1 = set(result1[0].keys()) if result1 else set()
        cols2 = set(result2[0].keys()) if result2 else set()
        if cols1 != cols2:
            only_1 = cols1 - cols2
            only_2 = cols2 - cols1
            detail_parts = []
            if only_1:
                detail_parts.append(f"仅 result1 有: {sorted(only_1)}")
            if only_2:
                detail_parts.append(f"仅 result2 有: {sorted(only_2)}")
            report.checks.append(DivergenceCheck(
                passed=False,
                description="输出列不一致",
                detail="; ".join(detail_parts),
            ))

        # ====================================================================
        # 差异检查 3：值差异
        # 只有当行数和列名都一致时，才按行列逐一比较每个单元格的值。
        # 如果有值不同，统计差异数量。
        #
        # 为什么要限制行列数一致才做值比较？
        #   如果行数或列名都不一样，逐行比较没有意义（对应关系不明确）。
        # ====================================================================
        if len(result1) == len(result2) and cols1 == cols2 and cols1:
            sorted_cols = sorted(cols1)
            differences = 0
            for i, (r1, r2) in enumerate(zip(result1, result2)):
                for col in sorted_cols:
                    v1 = r1.get(col)
                    v2 = r2.get(col)
                    if v1 != v2:
                        differences += 1
            if differences > 0:
                report.checks.append(DivergenceCheck(
                    passed=False,
                    description=f"值差异: {differences} 处不同",
                    detail=f"相同行列位置但值不同。两个候选 SQL 的聚合/计算逻辑可能存在差异。",
                ))

        return report
