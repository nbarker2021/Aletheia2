"""
Validation Manager – ensures that whenever more than 5 pending validation
items exist, validation becomes the highest‑priority task immediately
(as per user protocol).

Usage:
    from cqe_core.validation import validator
    validator.register("check_world", lambda: world.validate())
    ...
    # No need to call run() manually; it auto‑fires when >5 items queued.
"""
from __future__ import annotations
from typing import Callable, List, Dict, Any
import threading, logging, importlib

logger = logging.getLogger("cqe.validation")

class ValidationManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._queue: List[tuple[str, Callable[[], Any]]] = []

    def register(self, name: str, fn: Callable[[], Any]) -> None:
        """
        Add a validation callable. If the queue size exceeds 5, validations
        are executed immediately (blocking).
        """
        with self._lock:
            self._queue.append((name, fn))
            if len(self._queue) > 5:
                self._run_all()

    def _run_all(self):
        logger.info("Validation threshold exceeded – executing %d tests", len(self._queue))
        while self._queue:
            name, fn = self._queue.pop(0)
            try:
                result = fn()
                logger.info("✔ validation %s passed: %s", name, result)
            except Exception as e:
                logger.error("✘ validation %s failed: %s", name, e)
                # Could raise or record failure; here we raise to stop pipeline
                raise

    # Manual trigger
    def run_pending(self):
        with self._lock:
            self._run_all()

# Singleton exposed
validator = ValidationManager()

__all__ = ["validator", "ValidationManager"]
