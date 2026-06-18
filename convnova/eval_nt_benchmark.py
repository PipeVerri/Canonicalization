#!/usr/bin/env python3
"""
Run train.py sequentially for each NT benchmark task.
Replaces the Hydra multirun to ensure each task gets its own W&B run.
"""

import subprocess
import sys
from datetime import datetime

TASKS = [
    "enhancers",
    "enhancers_types",
    "H3",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K9ac",
    "H3K14ac",
    "H3K36me3",
    "H3K79me3",
    "H4",
    "H4ac",
    "promoter_all",
    "promoter_no_tata",
    "promoter_tata",
    "splice_sites_acceptors",
    "splice_sites_donors",
    "splice_sites_all",
]


def run_task(task: str, extra_args: list[str] = []) -> bool:
    cmd = [
        sys.executable, "train.py",
        "experiment=canon/nt_benchmark",
        f"dataset.dataset_name={task}",
        *extra_args,
    ]

    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting task: {task}")
    print(f"CMD: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[ERROR] Task '{task}' failed with exit code {result.returncode}")
        return False

    print(f"[OK] Task '{task}' completed successfully")
    return True


def main():
    failed = []
    succeeded = []

    for task in TASKS:
        ok = run_task(task)
        (succeeded if ok else failed).append(task)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {len(TASKS)} tasks total")
    print(f"  ✓ Succeeded ({len(succeeded)}): {', '.join(succeeded) or 'none'}")
    print(f"  ✗ Failed    ({len(failed)}): {', '.join(failed) or 'none'}")
    print('='*60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()