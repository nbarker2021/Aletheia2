import json, base64, urllib.request, urllib.error
class RPCError(Exception): pass
class NodeAdapter:
    def __init__(self, rpc_url: str, rpc_user: str, rpc_pass: str, timeout: float = 120.0):
        self.url = rpc_url; self.timeout = timeout
        import base64 as b64; self.auth = b64.b64encode(f"{rpc_user}:{rpc_pass}".encode()).decode()
        self._id = 0
    def _call(self, method: str, params: list):
        self._id += 1
        body = json.dumps({"jsonrpc":"2.0","id":self._id,"method":method,"params":params}).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "Content-Type":"application/json","Authorization":"Basic "+self.auth})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RPCError(f"HTTP {e.code}: {e.read()}")
        except Exception as e: raise RPCError(str(e))
        if data.get("error"): raise RPCError(str(data["error"]))
        return data["result"]
    def getblocktemplate(self): return self._call("getblocktemplate", [{"rules":["segwit"]}])
    def submitblock(self, hexblock: str): return self._call("submitblock", [hexblock])
