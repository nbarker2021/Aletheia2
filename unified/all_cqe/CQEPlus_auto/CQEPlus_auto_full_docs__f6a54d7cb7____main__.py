import json
from pathlib import Path
from .cqe_slices import evaluate_slices

def main():
    out = evaluate_slices()
    print(json.dumps(out))

if __name__ == "__main__":
    main()
