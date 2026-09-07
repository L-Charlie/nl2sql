import os, sys, json, sqlite3, time
sys.path.insert(0, '.')

# load .env
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI
from agent_team.orchestrator import Orchestrator

# minimal schema
schema = [
    {'table_name': 'students', 'table_description': 'student info', 'columns': [
        {'col': 'id', 'type': 'INTEGER', 'description': 'PK'},
        {'col': 'name', 'type': 'TEXT', 'description': ''},
        {'col': 'age', 'type': 'INTEGER', 'description': ''},
    ]},
    {'table_name': 'pets', 'table_description': 'pet ownership', 'columns': [
        {'col': 'id', 'type': 'INTEGER', 'description': 'PK'},
        {'col': 'student_id', 'type': 'INTEGER', 'description': 'FK->students.id'},
        {'col': 'pet_name', 'type': 'TEXT', 'description': ''},
    ]},
]

client = OpenAI(base_url='https://api.deepseek.com', api_key=os.getenv('OPENAI_API_KEY'))
print('Client created, starting orchestrator...')

# create temp db
db_path = 'test_temp.sqlite'
conn = sqlite3.connect(db_path)
conn.execute('CREATE TABLE students (id INTEGER, name TEXT, age INTEGER)')
conn.execute('CREATE TABLE pets (id INTEGER, student_id INTEGER, pet_name TEXT)')
conn.execute("INSERT INTO students VALUES (1, 'Alice', 20)")
conn.execute("INSERT INTO students VALUES (2, 'Bob', 22)")
conn.execute("INSERT INTO pets VALUES (1, 1, 'Fluffy')")
conn.execute("INSERT INTO pets VALUES (2, 2, 'Rex')")
conn.commit()
conn.close()

orch = Orchestrator(schema=schema, model='deepseek-v4-pro', model_client=client, domain='generic')

def execute(sql):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(sql).fetchall()]
        conn.close()
        return {'ok': True, 'error': '', 'rows': rows, 'row_count': len(rows), 'sample_rows': rows[:5]}
    except Exception as e:
        conn.close()
        return {'ok': False, 'error': str(e), 'rows': [], 'row_count': 0, 'sample_rows': []}

print('Running...')
start = time.time()
result = orch.run('How many pets does Alice have?', execute_sql=execute, db_path=db_path)
print(f'Done in {time.time()-start:.1f}s')
print(f'SQL: {result.sql}')
print(f'Success: {result.success}')
print(f'Iterations: {result.iterations}')
print(f'Error: {result.error}')
print(f'Conversation log entries: {len(result.conversation_log)}')
for entry in result.conversation_log:
    print(f"  [{entry.get('module')}:{entry.get('stage')}] {entry.get('ok','')}")

os.remove(db_path)
