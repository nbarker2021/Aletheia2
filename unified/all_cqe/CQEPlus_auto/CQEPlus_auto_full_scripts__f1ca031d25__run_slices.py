# Run the slices evaluate entrypoint
import json
from cqeplus.cqe_slices import evaluate_slices
if __name__ == "__main__":
    print(json.dumps(evaluate_slices(), indent=2))
