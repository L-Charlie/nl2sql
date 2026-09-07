"""
============================================================================
  build_indices.py —— 知识图谱 JSON 和嵌入索引构建脚本（一次性使用）
============================================================================

【功能概述】
本脚本负责从源文件构建 NL2SQL 系统所需的三大索引：
  1. 知识图谱（knowledge graph）—— 将数据库 Schema + 规则转化为图结构
  2. 嵌入索引（embedding index）—— 为字段名/表名/业务术语生成向量嵌入
  3. 案例库（case library）—— 从历史案例目录加载已解决的 SQL 案例

【输出文件说明】
  输出目录（由 --output-dir 指定）中包含以下文件：
  ┌──────────────────────────────┬─────────────────────────────────────┐
  │ 文件名                       │ 用途                                 │
  ├──────────────────────────────┼─────────────────────────────────────┤
  │ knowledge_graph.json         │ 知识图谱：节点(表/列/规则) + 边关系  │
  │ embedding_index.json         │ 嵌入索引：字段名和业务术语的向量索引 │
  │ case_library.json            │ 案例库：历史已解决的 SQL 案例集合    │
  └──────────────────────────────┴─────────────────────────────────────┘

【源文件与输出索引的关系】
  源文件                              →  输出索引
  ─────────────────────────────────────────────────────────
  schema.json（数据库 Schema 定义）    →  知识图谱的"表/列"节点
                                        嵌入索引的字段名条目
  SQL 规则手册（可选）                 →  知识图谱的"规则"节点
  common_knowledge.md（业务知识文档）  →  嵌入索引的业务术语条目
  cases/ 目录（历史案例文件夹）        →  case_library.json

【使用方式】
  在项目根目录下执行：
    python -m agent_team.build_indices ^
        --schema ../path/to/schema.json ^
        --knowledge ../path/to/common_knowledge.md ^
        --cases ../path/to/cases/ ^
        --output-dir agent_team/data/

【参数说明】
  --schema     （必需）数据库 Schema 的 JSON 文件路径
  --rules      （可选）SQL 规则文本文件路径
  --cases      （必需）历史案例目录路径
  --knowledge  （可选）通用业务知识文档路径（Markdown 格式）
  --output-dir （必需）构建产物的输出目录
"""

import argparse      # 用于解析命令行参数
import os            # 用于文件和路径操作
import sys           # 用于系统相关功能
import json          # 用于 JSON 文件的读写

# 导入本包的自定义构建器
from agent_team.knowledge_graph_builder import KnowledgeGraphBuilder   # 知识图谱构建器
from agent_team.embedding_index import EmbeddingIndex                  # 嵌入索引构建器
from agent_team.case_library_loader import CaseLibraryLoader           # 案例库加载器


def main():
    """
    主函数：按顺序执行知识图谱、嵌入索引和案例库的构建流程。

    执行步骤：
    1. 解析命令行参数
    2. 创建输出目录（如果不存在）
    3. 构建知识图谱（从 schema.json + spider_agent.txt）
    4. 构建嵌入索引（从 schema.json + common_knowledge.md）
    5. 加载案例库（从 cases/ 目录）
    6. 打印输出文件清单
    """
    # ----- 第一步：解析命令行参数 -----
    parser = argparse.ArgumentParser(description="构建 NL2SQL 多智能体知识索引")
    parser.add_argument("--schema", required=True, help="Schema JSON 文件的路径")
    parser.add_argument("--rules", default=None, help="SQL 规则文件的路径（可选）")
    parser.add_argument("--cases", required=True, help="cases/ 案例目录的路径")
    parser.add_argument("--knowledge", default=None, help="common_knowledge.md 业务知识文档的路径（可选）")
    parser.add_argument("--output-dir", required=True, help="构建产物的输出目录路径")
    args = parser.parse_args()

    # 确保输出目录存在，如果不存在则自动创建
    os.makedirs(args.output_dir, exist_ok=True)

    # ===== 步骤 1：构建知识图谱 =====
    print("正在构建知识图谱...")
    kg_builder = KnowledgeGraphBuilder.from_files(args.schema, args.rules or "")
    # 执行构建：内部会解析 Schema 创建表/列节点，解析规则创建规则节点
    graph = kg_builder.build()
    # 将构建好的图结构导出为 JSON 文件，供后续使用
    graph_path = os.path.join(args.output_dir, "knowledge_graph.json")
    kg_builder.export_json(graph_path)
    print(f"  -> 知识图谱已保存至 {graph_path}")
    print(f"    {len(graph.nodes)} 个节点, {len(graph.edges)} 条边")

    # ===== 步骤 2：构建嵌入索引 =====
    # 嵌入索引将文本（字段名、表名、业务术语）转换为向量，
    # 使得语义相似的词在向量空间中距离更近。
    # SchemaLinker 利用这个索引将用户问题中的自然语言词汇匹配到数据库字段。
    print("正在构建嵌入索引...")
    # 读取 Schema JSON 文件，获取表名和列名
    with open(args.schema, 'r', encoding='utf-8') as f:
        schema = json.load(f)

    # 如果提供了业务知识文档，则从中提取业务术语（如"毛利率"、"同比增长"等）
    # 这些术语会被加入嵌入索引，帮助系统理解业务语境中的特殊词汇
    business_terms = None
    if args.knowledge and os.path.exists(args.knowledge):
        with open(args.knowledge, 'r', encoding='utf-8') as f:
            knowledge_text = f.read()
        # _extract_business_terms 是一个内部方法，从纯文本中提取业务术语
        business_terms = EmbeddingIndex._extract_business_terms(knowledge_text)

    # 创建嵌入索引实例，基于 Schema 和业务术语构建索引
    idx = EmbeddingIndex()
    idx.build_from_schema(schema, business_terms=business_terms)
    # 保存嵌入索引到 JSON 文件
    index_path = os.path.join(args.output_dir, "embedding_index.json")
    idx.save(index_path)
    print(f"  -> 嵌入索引已保存至 {index_path}")
    print(f"    {len(idx.entries)} 个条目已索引")

    # ===== 步骤 3：加载案例库 =====
    # 案例库存储了"之前已经解决过的 NL2SQL 转换案例"。
    # Refiner 在修复 SQL 时，可以参考相似的历史案例来找到修复模式。
    # 每个案例包含：原始问题、生成的 SQL、是否正确、如果有错是怎么修复的。
    print("正在加载案例库...")
    loader = CaseLibraryLoader()
    # 从 cases/ 目录中递归加载所有案例文件
    case_lib = loader.load_from_directory(args.cases)
    case_lib_path = os.path.join(args.output_dir, "case_library.json")
    # 将案例库序列化为 JSON 并保存
    with open(case_lib_path, 'w', encoding='utf-8') as f:
        json.dump(case_lib.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"  -> 案例库已保存至 {case_lib_path}")
    print(f"    {case_lib.summary()}")

    # ===== 步骤 4：冷启动 SQL 经验 RAG =====
    # 将案例库中的全部案例灌入 RAG（SQL 经验记忆），让 Builder 在
    # 每次生成 SQL 前都能检索到相似的历史案例作为参考。
    print("正在冷启动 SQL 经验 RAG...")
    try:
        from agent_team.sql_rag import load_from_case_library
        rag_store = load_from_case_library(case_lib_path, db_id="", skip_dedup=True)
        rag_path = os.path.join(args.output_dir, "sql_experiences.json")
        rag_store.save(rag_path)
        print(f"  -> RAG 经验库已保存至 {rag_path}")
        print(f"    总计 {rag_store.count} 条经验 "
              f"（成功 {rag_store.success_count} / 失败 {rag_store.count - rag_store.success_count}）")
    except Exception as e:
        print(f"  -> RAG 冷启动失败（非致命）: {e}")

    # ----- 输出完成信息及文件清单 -----
    print("\n构建完成。输出文件清单：")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {fpath} ({size_kb:.1f} KB)")


# 入口：当该脚本被直接运行时（而非作为模块导入），调用 main() 函数
if __name__ == "__main__":
    main()
