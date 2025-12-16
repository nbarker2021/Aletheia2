from __future__ import annotations
def validate_file(path: str, schema: str = "snap"):
    data = json.loads(pathlib.Path(path).read_text())
    if schema == "e8":
        Draft202012Validator(E8_SCHEMA).validate(data)
    else:
        # allow SNAP to $ref E8
        Draft202012Validator(SNAP_SCHEMA, resolver=InMemoryResolver()).validate(data)
    print("OK:", path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate.py <file> [e8|snap]")
        sys.exit(1)
    path = sys.argv[1]
    which = sys.argv[2] if len(sys.argv) > 2 else "snap"
    validate_file(path, which)

from dataclasses import dataclass
from typing import Dict, List
import math

