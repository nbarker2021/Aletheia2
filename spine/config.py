"""
Config - Path and Environment Configuration

All paths are resolved relative to the project root or via environment variables.
No hardcoded absolute paths.

Environment Variables:
- CQE_ROOT: Root directory for CQE data (default: current working directory)
- CQE_LEDGER_PATH: Path for ledger storage (default: {CQE_ROOT}/ledger)
- CQE_STORAGE_PATH: Path for atom storage (default: {CQE_ROOT}/storage)
- CQE_CACHE_PATH: Path for cache (default: {CQE_ROOT}/cache)
"""

import os
from pathlib import Path
from typing import Optional


def get_root() -> Path:
    """
    Get the CQE root directory.
    
    Resolution order:
    1. CQE_ROOT environment variable
    2. Current working directory
    """
    root = os.environ.get("CQE_ROOT")
    if root:
        return Path(root)
    return Path.cwd()


def get_ledger_path() -> Path:
    """
    Get the ledger storage path.
    
    Resolution order:
    1. CQE_LEDGER_PATH environment variable
    2. {CQE_ROOT}/ledger
    """
    path = os.environ.get("CQE_LEDGER_PATH")
    if path:
        return Path(path)
    return get_root() / "ledger"


def get_storage_path() -> Path:
    """
    Get the atom storage path.
    
    Resolution order:
    1. CQE_STORAGE_PATH environment variable
    2. {CQE_ROOT}/storage
    """
    path = os.environ.get("CQE_STORAGE_PATH")
    if path:
        return Path(path)
    return get_root() / "storage"


def get_cache_path() -> Path:
    """
    Get the cache path.
    
    Resolution order:
    1. CQE_CACHE_PATH environment variable
    2. {CQE_ROOT}/cache
    """
    path = os.environ.get("CQE_CACHE_PATH")
    if path:
        return Path(path)
    return get_root() / "cache"


def ensure_path(path: Path) -> Path:
    """Ensure a path exists, creating directories if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(relative_path: str, base: Optional[Path] = None) -> Path:
    """
    Resolve a relative path against a base.
    
    If base is None, uses CQE_ROOT.
    """
    if base is None:
        base = get_root()
    return base / relative_path


class Config:
    """
    Configuration singleton for the CQE system.
    
    All paths are resolved lazily and can be overridden via environment variables.
    """
    
    _instance: Optional['Config'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._root: Optional[Path] = None
        self._ledger_path: Optional[Path] = None
        self._storage_path: Optional[Path] = None
        self._cache_path: Optional[Path] = None
    
    @property
    def root(self) -> Path:
        if self._root is None:
            self._root = get_root()
        return self._root
    
    @root.setter
    def root(self, value: Path):
        self._root = Path(value)
        # Reset derived paths
        self._ledger_path = None
        self._storage_path = None
        self._cache_path = None
    
    @property
    def ledger_path(self) -> Path:
        if self._ledger_path is None:
            self._ledger_path = get_ledger_path()
        return self._ledger_path
    
    @ledger_path.setter
    def ledger_path(self, value: Path):
        self._ledger_path = Path(value)
    
    @property
    def storage_path(self) -> Path:
        if self._storage_path is None:
            self._storage_path = get_storage_path()
        return self._storage_path
    
    @storage_path.setter
    def storage_path(self, value: Path):
        self._storage_path = Path(value)
    
    @property
    def cache_path(self) -> Path:
        if self._cache_path is None:
            self._cache_path = get_cache_path()
        return self._cache_path
    
    @cache_path.setter
    def cache_path(self, value: Path):
        self._cache_path = Path(value)
    
    def to_dict(self) -> dict:
        return {
            "root": str(self.root),
            "ledger_path": str(self.ledger_path),
            "storage_path": str(self.storage_path),
            "cache_path": str(self.cache_path),
        }


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
