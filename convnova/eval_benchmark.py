#!/usr/bin/env python3
"""
Run train.py sequentially for each NT benchmark task.
Replaces the Hydra multirun to ensure each task gets its own W&B run.
"""

import subprocess
import sys
from datetime import datetime
import argparse

NT_BENCHMARK_TASKS = (
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
    "enhancers_types"
)

GENOMIC_BENCHMARK_TASKS = (
    "dummy_mouse_enhancers_ensembl", 
    "demo_coding_vs_intergenomic_seqs", 
    "demo_human_or_worm",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl"
)

def run_task(task: str, exp_name, extra_args: list[str] = []) -> bool:
    cmd = [
        sys.executable, "train.py",
        f"experiment={exp_name}",
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

def main(tasks, experiment):
    failed = []
    succeeded = []

    for task in tasks:
        ok = run_task(task, experiment)
        (succeeded if ok else failed).append(task)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY — {len(tasks)} tasks total")
    print(f"  ✓ Succeeded ({len(succeeded)}): {', '.join(succeeded) or 'none'}")
    print(f"  ✗ Failed    ({len(failed)}): {', '.join(failed) or 'none'}")
    print('='*60)

    if failed:
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True)
    parser.add_argument("-t", "--task", required=True)
    args = parser.parse_args()

    if args.task == "nt-benchmark":
        to_eval = NT_BENCHMARK_TASKS
    elif args.task == "genomic-benchmark":
        to_eval = GENOMIC_BENCHMARK_TASKS
    else:
        raise ValueError("Task not recognized")

    main(to_eval, args.experiment)