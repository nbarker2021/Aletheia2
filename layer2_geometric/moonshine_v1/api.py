
import json, time, uuid, os
from typing import Any, Dict, List
from wsgiref.simple_server import make_server
from urllib.parse import parse_qs
from db import connect, add_item, get_item, list_items, search, log, stats
from embedding.voa_moonshine import moonshine_feature, fuse
from embedding.geometry_bridge import radial_angle_hist
from embedding.cqe_channels import summarize_lane

DB_PATH = "./data/monster_moonshine.db"

os.makedirs("./data", exist_ok=True)
con = connect(DB_PATH)

def read_json(environ):
    try:
        length = int(environ.get('CONTENT_LENGTH', '0'))
    except (ValueError):
        length = 0
    body = environ['wsgi.input'].read(length) if length > 0 else b'{}'
    return json.loads(body.decode('utf-8') or "{}")

def respond(start_response, status: str, obj: dict, ctype="application/json"):
    data = json.dumps(obj).encode("utf-8")
    headers = [('Content-Type', ctype), ('Content-Length', str(len(data)))]
    start_response(status, headers)
    return [data]

def app(environ, start_response):
    path = environ.get('PATH_INFO', '/')
    method = environ.get('REQUEST_METHOD', 'GET')

    if path == "/api/stats":
        return respond(start_response, '200 OK', stats(con))

    if path == "/api/list":
        q = parse_qs(environ.get('QUERY_STRING', ''))
        limit = int(q.get('limit',['100'])[0]); offset = int(q.get('offset',['0'])[0])
        return respond(start_response, '200 OK', {"items": list_items(con, limit, offset)})

    if path == "/api/get":
        q = parse_qs(environ.get('QUERY_STRING', ''))
        iid = q.get('id', [''])[0]
        item = get_item(con, iid)
        if not item: return respond(start_response, '404 NOT FOUND', {"error":"not found"})
        return respond(start_response, '200 OK', item)

    if path == "/api/add" and method == "POST":
        payload = read_json(environ)
        kind = payload.get("kind","geom")
        meta = payload.get("meta",{})
        chart_names = payload.get("charts",["moonshine","geom","cqe"])
        parts = {
            "moonshine": moonshine_feature(dim=32),
            "geom": radial_angle_hist(payload.get("points", []), rbins=16, abins=16),
            "cqe": summarize_lane(meta),
        }
        vec = fuse(parts)
        item_id = payload.get("id") or str(uuid.uuid4())
        add_item(con, item_id=item_id, kind=kind, vec=vec, meta=meta, chart_names=chart_names)
        log(con, "add", {"id": item_id, "kind": kind})
        return respond(start_response, '200 OK', {"id": item_id, "dim": len(vec)})

    if path == "/api/search" and method == "POST":
        payload = read_json(environ)
        parts = {
            "moonshine": moonshine_feature(dim=32),
            "geom": radial_angle_hist(payload.get("points", []), rbins=16, abins=16),
            "cqe": summarize_lane(payload.get("meta", {})),
        }
        vec = fuse(parts)
        res = search(con, vec, topk=int(payload.get("topk",10)), chart_name=payload.get("chart"))
        return respond(start_response, '200 OK', {"results": res})

    if path == "/" or path.startswith("/static/"):
        if path == "/": path = "/static/index.html"
        try:
            with open("."+path, "rb") as f:
                data = f.read()
            ctype = "text/html"
            if path.endswith(".js"): ctype = "text/javascript"
            if path.endswith(".css"): ctype = "text/css"
            start_response('200 OK', [('Content-Type', ctype)])
            return [data]
        except Exception:
            start_response('404 NOT FOUND', [('Content-Type','text/plain')])
            return [b'Not found']

    start_response('404 NOT FOUND', [('Content-Type','text/plain')])
    return [b'Unknown route']

def serve(host="127.0.0.1", port=8765):
    httpd = make_server(host, port, app)
    print(f"Serving Monster/Moonshine DB on http://{host}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    serve()
