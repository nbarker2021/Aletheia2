
import json, sys, pathlib
from jsonschema import Draft202012Validator, RefResolver

BASE = pathlib.Path(__file__).resolve().parent.parent

E8_SCHEMA = json.loads((BASE / "E8_Addressing_Schema_v0.1.json").read_text())
SNAP_SCHEMA = json.loads((BASE / "snap_manifest_v2.schema.json").read_text())

class InMemoryResolver(RefResolver):
    def __init__(self):
        super().__init__(base_uri=E8_SCHEMA.get("$id",""), referrer=E8_SCHEMA)
        self.store = {
            E8_SCHEMA["$id"]: E8_SCHEMA,
            SNAP_SCHEMA["$id"]: SNAP_SCHEMA
        }

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
