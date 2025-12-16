# Merge receipts into a unified ledger stream.
import sys, json
from pathlib import Path

def main(out_path="merged_receipts.jsonl", *paths):
    out = Path(out_path)
    with out.open("w", encoding="utf-8") as f_out:
        for p in paths:
            pp = Path(p)
            if not pp.exists(): 
                continue
            for line in pp.read_text(encoding="utf-8").splitlines():
                if not line.strip(): 
                    continue
                f_out.write(line.strip()+"\n")
    print(str(out))

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        # default: try known local files relative to data/
        base = Path(__file__).resolve().parents[1]/"data"
        defaults = [
            base/"slices_receipts.jsonl",
            base/"slices_receipts_v2.jsonl",
            base/"complex_inclusions_receipts.jsonl",
        ]
        main("merged_receipts.jsonl", *[str(x) for x in defaults])
    else:
        main(args[0], *args[1:])
