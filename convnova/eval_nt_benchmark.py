#!/usr/bin/env python3
"""
Run train.py sequentially for each NT benchmark task.
Replaces the Hydra multirun to ensure each task gets its own W&B run.
"""

import subprocess
import sys
from datetime import datetime

TASKS = [
    "H2AFZ",
    "H3K27ac",
    "H3K27me3",
    "H3K36me3",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K9ac",
    "H3K9me3",
    "H4K20me1",
    "promoter_all",
    "promoter_tata",
    "promoter_no_tata",
    "splice_sites_all",
    "splice_sites_donors",
    "splice_sites_acceptors",
    "enhancers",
    "enhancers_types",
]


def run_task(task: str, exp_name, extra_args: list[str] = []) -> bool:
    cmd = [
        sys.executable, "train.py",
        exp_name,
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


def main(exp):
    failed = []
    succeeded = []

    for task in TASKS:
        ok = run_task(task, exp)
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
    exp = sys.argv[1]
    if not exp.startswith("experiment="):
        raise ValueError("Experiment not specified")

    main(exp)