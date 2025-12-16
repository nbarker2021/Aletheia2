"""
CQE Bootstrap — ensures required dirs exist for home and local workspace.
"""
from __future__ import annotations
from cqe.utils.config import Settings

def ensure_dirs(paths):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def main() -> None:
    cfg = Settings.from_env()
    ensure_dirs([
        cfg.home, cfg.home/"runs", cfg.home/"logs", cfg.home/"cache", cfg.home/"data",
        cfg.runs_dir, cfg.logs_dir, cfg.data_dir
    ])
    print("✅ CQE bootstrap complete.")
    print(f"Home → {cfg.home}")
    print(f"Local → {cfg.runs_dir}, {cfg.logs_dir}, {cfg.data_dir}")

if __name__ == "__main__":
    main()
