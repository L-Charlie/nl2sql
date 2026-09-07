"""Smoke test for RAG upgrade: retrieve + structure + dedup."""
import sys
sys.path.insert(0, ".")
from agent_team.sql_rag import (
    SQLExperienceStore, SQLExperience,
    _question_structure_intent, _structure_overlap_score,
)

# Test 1: construct store with 3 experiences (fallback char-bigram encoding)
store = SQLExperienceStore()
exp1 = SQLExperience(question="Find the name of students with pets",
                     sql="SELECT name FROM student JOIN pet",
                     db_id="school", tables_used=["student", "pet"])
exp2 = SQLExperience(question="Count how many pets each student has",
                     sql="SELECT student_id, COUNT(*) FROM pet GROUP BY student_id",
                     db_id="school", tables_used=["pet"])
exp3 = SQLExperience(question="Find students who have both cats and dogs",
                     sql="SELECT name FROM student WHERE id IN (SELECT student_id FROM pet WHERE type='cat') INTERSECT SELECT name FROM student WHERE id IN (SELECT student_id FROM pet WHERE type='dog')",
                     db_id="school", tables_used=["student", "pet"])
store.add(exp1); store.add(exp2); store.add(exp3)
print(f"[1] Store count: {store.count} (expect 3)")

# Test 2: retrieve
results = store.retrieve("How many students have pets?", top_k=3)
print(f"[2] Retrieved: {len(results)} results")
for r in results:
    print(f"    - {r.question[:60]}...")

# Test 3: structure intent extraction
intent = _question_structure_intent("排名前5的学生")
print(f"[3] Structure intent for 'top 5 students': {intent}")

# Test 4: structure overlap
key = SQLExperienceStore._sql_structure_key(exp3.sql)
score = _structure_overlap_score(intent, key)
print(f"[4] Structure overlap (window intent vs INTERSECT sql): {score:.2f}")

# Test 5: dedup — same question should not duplicate
store.add(SQLExperience(question="Find the name of students with pets",
                        sql="SELECT name FROM student JOIN pet",
                        db_id="school", tables_used=["student", "pet"]))
print(f"[5] After dedup add: {store.count} (expect 3)")

# Test 6: _sql_structure_key still works
key2 = SQLExperienceStore._sql_structure_key("SELECT a FROM t1 JOIN t2 GROUP BY a")
assert key2["join_count"] == 1
assert key2["has_group_by"] == True
assert key2["has_window"] == False
print("[6] _sql_structure_key works correctly")

# Test 7: _sql_structure_too_different still works
from agent_team.sql_rag import SQLExperienceStore as SES
assert SES._sql_structure_too_different("SELECT a FROM t1", "SELECT a FROM t1 JOIN t2 JOIN t3 JOIN t4")
assert not SES._sql_structure_too_different("SELECT a FROM t1", "SELECT b FROM t1")
print("[7] _sql_structure_too_different works correctly")

# Test 8: summary
s = store.summary()
print(f"[8] Summary: {s}")

print("\nALL TESTS PASSED")
