"""
Command-line interface for running the CQE harness.

This module wraps the smoke harness CLI defined in the testing suite
and exposes it under the `cqe.cli` namespace.  The harness CLI accepts
text input and writes receipts to the specified output directory.
"""

from testing.harness.harness_cli import main as main  # type: ignore

__all__ = ["main"]

if __name__ == "__main__":
    # Allow running `python -m cqe.cli.harness_cli` directly
    main()
