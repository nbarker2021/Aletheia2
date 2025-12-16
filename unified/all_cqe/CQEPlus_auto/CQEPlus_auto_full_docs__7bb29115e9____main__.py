"""CQE module entrypoint.

Allows `python -m cqe` as a cross-platform fallback to the CLI scripts.
Equivalent to: `python -m cqe.cli.harness_cli`.
"""
from .cli.harness_cli import main

if __name__ == "__main__":
    main()
