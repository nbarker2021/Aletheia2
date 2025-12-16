
import os, json
from typing import Dict, Any
from .utils import ensure_dir, now_iso

class Receipts:
    def __init__(self, dir_path: str):
        self.dir = dir_path
        ensure_dir(self.dir)
        self._open_files = {}

    def _file_for(self, run_id: str):
        path = os.path.join(self.dir, f"{run_id}.jsonl")
        if run_id not in self._open_files:
            self._open_files[run_id] = open(path, "a", encoding="utf-8")
        return self._open_files[run_id]

    def emit(self, run_id: str, record: Dict[str, Any]):
        record.setdefault("timestamp", now_iso())
        f = self._file_for(run_id)
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()

    def close_all(self):
        for f in self._open_files.values():
            try:
                f.close()
            except Exception:
                pass
        self._open_files.clear()
