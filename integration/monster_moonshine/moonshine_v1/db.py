
import os, sqlite3, json, time, math
from typing import List, Tuple, Dict, Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  created REAL NOT NULL,
  meta_json TEXT NOT NULL,
  vec BLOB NOT NULL,
  norm REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS charts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0
);
CREATE TABLE IF NOT EXISTS item_charts (
  item_id TEXT NOT NULL,
  chart_id INTEGER NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0,
  PRIMARY KEY (item_id, chart_id),
  FOREIGN KEY(item_id) REFERENCES items(id) ON DELETE CASCADE,
  FOREIGN KEY(chart_id) REFERENCES charts(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  op TEXT NOT NULL,
  notes TEXT NOT NULL
);
"""

def connect(path: str):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  con = sqlite3.connect(path, check_same_thread=False)
  con.execute("PRAGMA journal_mode=WAL;")
  con.execute("PRAGMA synchronous=NORMAL;")
  con.executescript(SCHEMA)
  return con

def l2norm(v):
  return math.sqrt(sum(x*x for x in v))

def add_item(con, *, item_id: str, kind: str, vec: list, meta: dict=None, chart_names: list=None):
  meta = meta or {}
  chart_names = chart_names or []
  norm = l2norm(vec)
  con.execute("INSERT OR REPLACE INTO items(id,kind,created,meta_json,vec,norm) VALUES(?,?,?,?,?,?)",
              (item_id, kind, time.time(), json.dumps(meta), json.dumps(vec), norm))
  for name in chart_names:
    cid = ensure_chart(con, name)
    con.execute("INSERT OR REPLACE INTO item_charts(item_id, chart_id, weight) VALUES(?,?,?)",
                (item_id, cid, 1.0))
  con.commit()

def ensure_chart(con, name: str) -> int:
  cur = con.execute("SELECT id FROM charts WHERE name=?", (name,))
  r = cur.fetchone()
  if r: return r[0]
  con.execute("INSERT INTO charts(name, weight) VALUES(?,?)", (name, 1.0))
  con.commit()
  return con.execute("SELECT id FROM charts WHERE name=?", (name,)).fetchone()[0]

def get_item(con, item_id: str):
  cur = con.execute("SELECT id, kind, created, meta_json, vec, norm FROM items WHERE id=?", (item_id,))
  r = cur.fetchone()
  if not r: return None
  return {"id": r[0], "kind": r[1], "created": r[2], "meta": json.loads(r[3]), "vec": json.loads(r[4]), "norm": r[5]}

def list_items(con, limit=100, offset=0):
  cur = con.execute("SELECT id,kind,created FROM items ORDER BY created DESC LIMIT ? OFFSET ?", (limit, offset))
  return [{"id":i, "kind":k, "created":c} for (i,k,c) in cur.fetchall()]

def cosine(a, b, anorm=None, bnorm=None):
  if anorm is None: anorm = l2norm(a)
  if bnorm is None: bnorm = l2norm(b)
  if anorm == 0 or bnorm == 0: return 0.0
  return sum(x*y for x,y in zip(a,b)) / (anorm*bnorm)

def search(con, vec: list, topk=10, chart_name: str=None):
  anorm = l2norm(vec)
  params = ()
  if chart_name:
    q = """
SELECT items.id, items.vec, items.norm
FROM items JOIN item_charts ON items.id=item_charts.item_id
JOIN charts ON item_charts.chart_id=charts.id
WHERE charts.name=?
"""
    params = (chart_name,)
  else:
    q = "SELECT id, vec, norm FROM items"
  sims = []
  for item_id, vjson, vnorm in con.execute(q, params):
    v = json.loads(vjson)
    s = cosine(vec, v, anorm, vnorm)
    sims.append((s, item_id))
  sims.sort(reverse=True)
  return [{"id": iid, "score": float(s)} for (s,iid) in sims[:topk]]

def log(con, op: str, notes: dict):
  con.execute("INSERT INTO logs(ts, op, notes) VALUES(?,?,?)", (time.time(), op, json.dumps(notes)))
  con.commit()

def stats(con):
  c_items = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
  c_charts = con.execute("SELECT COUNT(*) FROM charts").fetchone()[0]
  return {"items": c_items, "charts": c_charts}
