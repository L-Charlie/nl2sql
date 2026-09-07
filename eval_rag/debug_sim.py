"""Debug: bge-m3 同库相似度分布 vs 0.7 阈值"""
import json, sys
sys.path.insert(0, ".")
from agent_team.sql_rag import SQLExperienceStore, _get_encoder, _BGE_INSTRUCTION
import numpy as np
from collections import defaultdict

store = SQLExperienceStore()
store.load("eval_rag/eval_store.json")
queries = json.load(open("eval_rag/eval_queries.json"))

encoder = _get_encoder()
print(f"Encoder: {type(encoder).__name__}")

# 统计所有 query 对同库文档的相似度
same_db_sims = []
cross_db_sims = []
self_sims = []

for qm in queries:
    q = qm["question"]
    q_emb = np.array(encoder.encode([q], normalize_embeddings=True,
                                     prompt=_BGE_INSTRUCTION))[0]
    scores = np.dot(store._question_embeddings, q_emb)

    for i, exp in enumerate(store._experiences):
        s = float(scores[i])
        if exp.question == q:
            self_sims.append(s)
        elif exp.db_id == qm["db_id"]:
            same_db_sims.append(s)
        else:
            cross_db_sims.append(s)

print(f"\nSelf-match similarity: {np.mean(self_sims):.4f} ± {np.std(self_sims):.4f}")
print(f"Same-db similarity:    {np.mean(same_db_sims):.4f} ± {np.std(same_db_sims):.4f}")
print(f"Cross-db similarity:   {np.mean(cross_db_sims):.4f} ± {np.std(cross_db_sims):.4f}")

print(f"\nSame-db above 0.7: {sum(1 for s in same_db_sims if s > 0.7)} / {len(same_db_sims)} ({100*sum(1 for s in same_db_sims if s > 0.7)/len(same_db_sims):.1f}%)")
print(f"Same-db above 0.5: {sum(1 for s in same_db_sims if s > 0.5)} / {len(same_db_sims)} ({100*sum(1 for s in same_db_sims if s > 0.5)/len(same_db_sims):.1f}%)")

# 直方图
bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print(f"\nSame-db similarity histogram:")
hist, _ = np.histogram(same_db_sims, bins=bins)
for i in range(len(hist)):
    print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]}")

print(f"\nCross-db similarity histogram:")
hist, _ = np.histogram(cross_db_sims, bins=bins)
for i in range(len(hist)):
    print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]}")
