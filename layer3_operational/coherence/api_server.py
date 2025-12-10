
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coherence/Decoherence API (personal server)"""
import json
from wsgiref.simple_server import make_server
from urllib.parse import parse_qs
from coherence_metrics import composite_coherence, collapse_detector, embedding_alignment
from state_store import StateStore

store = StateStore("./deco_states")

def read_json(environ):
    try:
        length = int(environ.get('CONTENT_LENGTH', '0'))
    except (ValueError): length = 0
    body = environ['wsgi.input'].read(length) if length > 0 else b'{}'
    return json.loads(body.decode('utf-8') or "{}")

def respond(start_response, status: str, obj: dict):
    data = json.dumps(obj).encode("utf-8")
    headers = [('Content-Type','application/json'), ('Content-Length', str(len(data)))]
    start_response(status, headers)
    return [data]

def app(environ, start_response):
    path = environ.get('PATH_INFO', '/')
    method = environ.get('REQUEST_METHOD', 'GET')

    if path == "/api/state/save" and method == "POST":
        payload = read_json(environ)
        rid = payload.get("receipt")
        if not rid: return respond(start_response, '400 BAD REQUEST', {"error":"missing receipt"})
        store.save(receipt=rid, points=payload.get("points"), tokens=payload.get("tokens"), embedding=payload.get("embedding"), meta=payload.get("meta"))
        return respond(start_response, '200 OK', {"ok": True, "receipt": rid})

    if path == "/api/state/get":
        q = parse_qs(environ.get('QUERY_STRING','')); rid = q.get('receipt',[''])[0]
        doc = store.load(rid) or {}
        return respond(start_response, '200 OK', doc if doc else {"error":"not found"})

    if path == "/api/state/list":
        return respond(start_response, '200 OK', {"items": store.list()})

    if path == "/api/metrics/coherence" and method == "POST":
        payload = read_json(environ)
        pts = payload.get("points") or []
        return respond(start_response, '200 OK', composite_coherence(pts))

    if path == "/api/metrics/collapse" and method == "POST":
        payload = read_json(environ)
        A = payload.get("prev_points") or []
        B = payload.get("curr_points") or []
        return respond(start_response, '200 OK', collapse_detector(A,B))

    if path == "/api/metrics/align" and method == "POST":
        payload = read_json(environ)
        A = payload.get("emb_a") or []
        B = payload.get("emb_b") or []
        return respond(start_response, '200 OK', {"alignment": embedding_alignment(A,B)})

    start_response('404 NOT FOUND', [('Content-Type','application/json')])
    return [b'{}']

def serve(host="127.0.0.1", port=8787):
    httpd = make_server(host, port, app)
    print(f"Serving Coherence API on http://{host}:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    serve()
