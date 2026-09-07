# nl2sql/agent_team/time_field_resolver.py
"""
TimeFieldResolver —— 时间字段解析器。

为什么时间字段在 NL2SQL 中如此重要？
  在数据分析场景中，绝大多数问题都涉及时间维度：
    - "上周的 DAU 是多少？"      -> 需要按周过滤
    - "本月累计付费金额"          -> 需要按月范围查询
    - "今天下午3点的在线人数"     -> 需要精确到时分秒
    - "对比去年同期的活跃用户"    -> 需要年同比计算

  如果选错了时间字段，SQL 查询会返回完全错误的结果。
  例如把 dteventtime（事件发生时间）当成 dtstatdate（分区统计日期）使用，
  可能导致一天的数据被分散到多个分区中。

  这个模块的核心任务就是：对于给定的表和问题，确定应该使用哪个时间字段。

表后缀命名规范：
  数据仓库中的表通常通过后缀标识其类型，这直接影响了时间字段的选择：
    - _di（Daily Increment）：日增量表，每天一个分区，使用 dtstatdate
    - _df（Daily Full）：日全量表，只保留最近一天的快照，使用 dtstatdate
    - _hi（Hourly Increment）：小时增量表，每小时一个分区，使用 dteventtime

  此外还有 DWS（汇总层）和 DWD（明细层）的区分：
    - DWS 表：汇总层，通常使用 dtstatdate（统计日期）
    - DWD 表：明细层，通常使用 dteventtime（事件发生时间）

紧凑时间格式：
  用户在问题中可能使用紧凑格式（如 "20250101-20250131"）而不是
  标准格式（"2025-01-01 至 2025-01-31"）。系统需要识别这种格式，
  并在 SQL 生成时转换为数据库兼容的格式。

参考来源：
  该模块的决策规则编码自 spider_agent.txt 第 91-137 行的时间字段选择决策树。
"""

import re
from typing import Optional


class TimeFieldResolver:
    """
    时间字段解析器 —— 为给定的表和问题确定最佳时间字段。

    核心决策规则：
      1. DWS/DIM 表只要有 dtstatdate 就是用 dtstatdate（分区日期，最可靠）。
      2. DWD 表（明细表，通常含 _hi 后缀）：
         - 如果问题是行为分析类（如"点击"、"发生"），优先用 dteventtime（事件时间）。
         - 如果问题是统计范围类（如"统计"、"期间"），优先用 tdbank_imp_date（入库时间）。
      3. _df 表只支持单日快照（全量），只能用 dtstatdate。
      4. 紧凑时间格式（如 20250101）需要对 dteventtime 做边界转换。

    使用示例：
      resolver = TimeFieldResolver()
      result = resolver.resolve(table, "上周的 DAU 趋势如何？")
      # result["primary"] -> {"table": "dws_user_dau", "field": "dtstatdate", "type": "partition_date"}
    """

    # ========================================================================
    # DATE_COLUMN_PATTERNS —— 日期/时间字段名称模式列表
    #
    # 格式：[ (正则表达式, 字段类型标签), ... ]
    #
    # 这个列表定义了如何根据列名判断它是否是时间字段。
    # 每个元素包含一个正则表达式和一个类型标签，
    # 当列名匹配正则时，就被识别为对应类型的日期/时间字段。
    #
    # 类型标签含义：
    #   - partition_date:   分区日期（最常用，对应 dtstatdate）
    #   - event_time:       事件发生时间（对应 dteventtime）
    #   - ingest_time:      数据入库时间（对应 tdbank_imp_date）
    #   - registration_date: 注册日期
    #   - generic_date:     通用日期字段
    #   - generic_time:     通用时间字段
    #
    # 注意：
    #   列表顺序很重要！排在越前面的模式优先级越高。
    #   例如 dtstatdate 排在第一个，因为它是最重要的时间字段。
    # ========================================================================
    DATE_COLUMN_PATTERNS = [
        (r"dtstatdate", "partition_date"),      # 分区统计日期，DWS 表默认
        (r"dteventtime", "event_time"),          # 事件发生时间，DWD 表默认
        (r"tdbank_imp_date", "ingest_time"),     # 数据入库时间，回退选项
        (r"dregdate", "registration_date"),       # 注册日期（数字格式）
        (r"iregdate", "registration_date"),       # 注册日期（整数格式）
        (r"dt\w*date", "partition_date"),         # 其他 dtxxxdate 格式
        (r"date\w*", "generic_date"),             # 通用日期字段
        (r"time\w*", "generic_time"),             # 通用时间字段
    ]

    # ========================================================================
    # BEHAVIORAL_PATTERNS —— 行为分析类问题的关键词模式
    #
    # 当问题匹配这些模式时，说明用户关心的是"事件发生的时间"，
    # 应该优先使用 dteventtime（事件时间）。
    #
    # 示例：
    #   - "用户点击了哪个按钮？"  -> 行为类，用事件时间
    #   - "昨天3点发生了多少次登录？" -> 行为类，用事件时间
    # ========================================================================
    BEHAVIORAL_PATTERNS = [
        r"(时刻|实时|点击|行为|事件|发生|触发|进行)",
        r"(在.*?(时候|时分|瞬间|时点))",
    ]

    # ========================================================================
    # STATISTICAL_PATTERNS —— 统计/范围分析类问题的关键词模式
    #
    # 当问题匹配这些模式时，说明用户关心的是"某段时间内的汇总数据"，
    # 应该优先使用 tdbank_imp_date（数据入库时间）。
    #
    # 为什么统计问题要用入库时间？
    #   对于 DWD 明细表，事件时间（dteventtime）可能因为延迟上报等原因
    #   导致数据不完整。使用入库时间（tdbank_imp_date）能确保统计的完整性。
    #
    # 示例：
    #   - "统计上个月的DAU"  -> 统计类，用入库时间
    #   - "过去7天的付费总额" -> 统计类，用入库时间
    # ========================================================================
    STATISTICAL_PATTERNS = [
        r"(统计|汇总|聚合|范围|期间|区间|过去|最近|历史|每日|每天|上周|本月)",
    ]

    def identify_date_columns(self, table: dict) -> list[dict]:
        """
        识别一张表中所有可能的日期/时间字段。

        实现方式：
          1. 遍历表的所有列。
          2. 对每个列名，依次与 DATE_COLUMN_PATTERNS 中的模式匹配。
          3. 如果匹配成功，记录该列的字段类型标签。
          4. 返回所有匹配到的日期/时间字段列表。

        注意：
          - 一个列只会匹配第一个成功的模式（break 跳出循环）。
          - 匹配是大小写不敏感的（re.IGNORECASE）。

        参数：
          table: 表字典，包含：
            - name:    表名
            - layer:   数据层级（DWS/DWD/DIM）
            - columns: 列列表，每列有 name、type、description

        返回：
          日期/时间字段的列表，每个字段包含：
            - name:     列名
            - type:     字段类型标签（如 partition_date、event_time）
            - col_type: 原始数据类型（如 string、bigint）
            - description: 列描述
        """
        date_cols = []
        for col in table.get("columns", []):
            col_name = col["name"]
            col_type = col.get("type", "").lower()
            desc = col.get("description", "")

            # 先做一个粗筛：如果类型明显不是日期/时间，可以跳过
            # 但这里为了保守起见，只检查是否包含 date/time/timestamp/varchar 类型关键词
            if any(t in col_type for t in ("date", "time", "timestamp", "varchar")):
                pass  # 可能是日期字段，继续检查列名模式

            # 用列名模式逐一匹配
            for pattern, field_type in self.DATE_COLUMN_PATTERNS:
                if re.search(pattern, col_name, re.IGNORECASE):
                    date_cols.append({
                        "name": col_name,
                        "type": field_type,              # 字段类型标签
                        "col_type": col.get("type", "unknown"),  # 数据库类型
                        "description": desc,
                    })
                    break  # 匹配到第一个模式就停止，避免重复添加

        return date_cols

    def resolve(self, table: dict, question: str) -> dict:
        """
        为给定的表和问题确定最佳时间字段。

        这是 TimeFieldResolver 的核心方法，执行完整的决策流程：
          1. 识别表中所有可能的时间字段。
          2. 根据问题类型（行为类/统计类）对时间字段排序。
          3. 选择最优的时间字段作为主字段。
          4. 如果有次优字段，也将其作为备选（fallback）。
          5. 根据表后缀和问题特征添加业务提醒。

        参数：
          table:    表字典，包含表名、层级、列列表
          question: 用户的自然语言问题

        返回：
          字典，包含：
            - primary:       主时间字段 {table, field, type}
            - fallback:      备选时间字段 {table, field, type}（可选）
            - all_candidates: 所有候选时间字段列表
            - notes:         业务提醒列表（如格式转换提醒、表特性说明）

        决策树逻辑（编码自 spider_agent.txt）：
          Step 1: 如果有 dtstatdate，默认用它（DWS/DIM 表的首选）。
          Step 2: 如果是行为分析问题，优先选 dteventtime。
          Step 3: 如果是统计范围问题，在 DWD 表上优先选 tdbank_imp_date。
          Step 4: 其他情况用默认排序。
        """
        # 识别表中的所有时间字段
        date_cols = self.identify_date_columns(table)
        table_name = table["name"]
        layer = table.get("layer", "")

        result = {
            "primary": {"table": table_name, "field": None, "type": "none"},
            "all_candidates": date_cols,
            "notes": [],
        }

        # 边界情况：表中没有时间字段
        if not date_cols:
            result["notes"].append(f"表 {table_name} 无日期字段，无法做时间过滤")
            return result

        # 对候选时间字段按优先级排序
        ranked = self.rank_time_fields(date_cols, question)

        if not ranked:
            return result

        # 选择排序第一的作为主时间字段
        primary = ranked[0]
        result["primary"] = {"table": table_name, "field": primary["name"], "type": primary["type"]}

        # 如果有第二个候选，作为备选时间字段
        if len(ranked) > 1:
            fallback = ranked[1]
            result["fallback"] = {"table": table_name, "field": fallback["name"], "type": fallback["type"]}

        # ====================================================================
        # 根据表后缀添加业务提醒
        #
        # 表后缀约定：
        #   _di（Daily Increment）：每日增量
        #     每天新增/变更的数据，保留历史所有分区。
        #     适合时间范围查询。
        #
        #   _df（Daily Full）：每日全量
        #     每天一个全量快照，通常只保留最近一天。
        #     不适合多日范围查询，需要用 _di 表代替。
        #
        #   _hi（Hourly Increment）：每小时增量
        #     每小时一个分区，时间精度高。
        #     但可能需要用 tdbank_imp_date 作为补充。
        # ====================================================================
        suffix = table_name.split("_")[-1] if "_" in table_name else ""

        # _df 表只支持单日快照
        if suffix == "df":
            result["notes"].append(
                "_df 表仅支持单日快照，多日范围查询需使用 _di 表或 DWD 明细表"
            )

        # _hi 表且使用事件时间时，提醒可能数据缺失
        if suffix == "hi" and primary["type"] == "event_time":
            result["notes"].append(
                "DWD 明细表使用 dteventtime。若数据在目标日期缺失，应切换到 tdbank_imp_date"
            )

        # 紧凑时间格式提醒：如果问题中有 "20250101" 这种格式，
        # 但主时间字段是 dteventtime，需要提醒转换格式
        if self.has_compact_time(question) and primary["type"] == "event_time":
            result["notes"].append(
                "问题使用紧凑时间格式，dteventtime 边界需转为 'YYYY-MM-DD HH:MM:SS' 格式"
            )

        # 比例计算提醒
        if self._is_ratio_question(question):
            result["notes"].append("比例计算需确保分子分母使用相同时间字段")

        return result

    def rank_time_fields(self, date_cols: list[dict], question: str) -> list[dict]:
        """
        对时间字段按问题上下文进行优先级排序。

        这是决策规则的核心实现。通过 priority() 函数为每个时间字段打分，
        分数越低优先级越高（类似于排名，第一名的分数是 0）。

        排序优先级（从高到低）：
          优先级 0（最高）：
            - dtstatdate：DWS 表的通用分区日期，永远优先
            - dteventtime：当问题是行为分析类时优先
            - tdbank_imp_date：当问题是统计范围类时优先
          优先级 1：
            - dteventtime：作为 DWD 表的默认选择
          优先级 2：
            - tdbank_imp_date：作为通用回退选择
          优先级 3：
            - 注册日期类字段
          优先级 4（最低）：
            - 其他未归类的时间字段

        参数：
          date_cols: 候选时间字段列表
          question:  用户问题（用于判断行为/统计类型）

        返回：
          按优先级排序后的时间字段列表（最优的在最前面）
        """
        # 检查问题是行为分析类还是统计范围类
        is_behavioral = any(re.search(p, question) for p in self.BEHAVIORAL_PATTERNS)
        is_statistical = any(re.search(p, question) for p in self.STATISTICAL_PATTERNS)

        def priority(col: dict) -> int:
            """
            计算单个时间字段的优先级分数。

            分数越低 = 优先级越高（0 是最优）。
            排序结果将按分数从小到大排列。
            """
            name = col["name"]
            ftype = col["type"]

            # 【优先级 0】dtstatdate 是 DWS 表的通用默认值
            # 几乎所有汇总层（DWS）表都有这个字段
            if name == "dtstatdate":
                return 0

            # 【优先级 0】行为分析问题优先用事件时间
            if is_behavioral and ftype == "event_time":
                return 0
            elif is_behavioral and name == "dteventtime":
                return 0

            # 【优先级 0】统计问题优先用入库时间
            if is_statistical and ftype == "ingest_time":
                return 0
            elif is_statistical and name == "tdbank_imp_date":
                return 0

            # 【优先级 1】dteventtime 是 DWD 明细表的默认选择
            if ftype == "event_time":
                return 1
            if name == "dteventtime":
                return 1

            # 【优先级 2】tdbank_imp_date 作为回退选择
            if ftype == "ingest_time" or name == "tdbank_imp_date":
                return 2

            # 【优先级 3】注册日期类字段
            if ftype in ("registration_date",):
                return 3

            # 【优先级 4】其他未归类的日期字段
            return 4

        ranked = sorted(date_cols, key=priority)
        return ranked

    def has_compact_time(self, question: str) -> bool:
        """
        检查问题中是否使用了紧凑时间格式。

        什么是紧凑时间格式？
          用户可能输入 "20250101" 而不是 "2025-01-01"。
          这种格式（连续 8 位数字）在数据分析中很常见，因为它：
            - 在文件名和分区名中常用
            - 排序时就是自然的字典序
            - 没有分隔符，方便程序处理

        当使用紧凑格式时，需要在 SQL 中做格式转换，
        例如将 '20250101' 转为 '2025-01-01 00:00:00' 才能与 dteventtime 比较。

        参数：
          question: 用户问题

        返回：
          True 如果问题中包含类似 20250101 的 8 位数字序列
        """
        return bool(re.search(r"\d{8}", question))

    def _is_ratio_question(self, question: str) -> bool:
        """
        判断问题是否涉及比例/百分比计算。

        比例计算需要特别注意时间字段的一致性：
          分子和分母都必须使用相同的时间字段进行过滤，
          否则计算出的比例会是错误的。

        例如："付费率是多少？"
          - 分子：付费用户数
          - 分母：活跃用户数
          如果分子用 dteventtime 过滤，分母用 dtstatdate 过滤，
          两个数字对应的时间范围不一致，付费率就是错误的。

        参数：
          question: 用户问题

        返回：
          True 如果问题涉及比例计算
        """
        ratio_keywords = ["比例", "占比", "百分比", "率", "/"]
        return any(kw in question for kw in ratio_keywords)
