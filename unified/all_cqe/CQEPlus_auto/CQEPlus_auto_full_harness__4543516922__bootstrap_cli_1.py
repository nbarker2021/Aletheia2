"""CQE Bootstrap CLI: creates ~/.cqe and local runs/logs/data folders."""
from __future__ import annotations
from cqe.utils.config import Settings

def ensure_dirs(paths):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def main():
    cfg = Settings.from_env()
    ensure_dirs([cfg.home, cfg.home/"runs", cfg.home/"logs", cfg.home/"cache", cfg.home/"data"])
    ensure_dirs([cfg.runs_dir, cfg.logs_dir, cfg.data_dir])
    print("CQE bootstrap complete.")
    print(f"Home stash -> {cfg.home}")
    print(f"Local dirs  -> {cfg.runs_dir}/, {cfg.logs_dir}/, {cfg.data_dir}/")
