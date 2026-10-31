from multiprocessing.pool import RUN
from tkinter import NO

import wandb
import math
import json
from pathlib import Path
import pandas as pd
import numpy as np

api = wandb.Api()
runs = runs = api.runs("pipe-verri-gh/Canonicalization", filters={
    "group": "baseline"
})

dataset_results = {}
metric_names = {}

def find_metric_name(run):
    metric = None
    for split in ("val"): # I don't want to monitor and optimize based off the test scores. Test scores are only used for validation
        if f"{split}/mcc" in run.summary.keys():
            metric = f"{split}/mcc"
            break
        elif f"{split}/auroc" in run.summary.keys():
            metric = f"{split}/auroc"
            break
        elif f"{split}/accuracy" in run.summary.keys():
            metric = f"{split}/accuracy"
            break
        elif f"{split}/loss" in run.summary.keys():
            metric = f"{split}/loss"
            break
    if metric is None:
        raise RuntimeError("Metric not found!")
    return metric

# Collect the metrics
for run in runs:
    if run.config["wandb"]["job_type"] == "pretraining":
        continue

    if run.config["wandb"]["job_type"] == "chromatin_profile_prediction":
        dataset_name = "chromatin_profile_prediction"
    else:
        dataset_name = run.config["dataset"]["dataset_name"]

    if not dataset_name in dataset_results:
        dataset_results[dataset_name] = []

    try:
        m = run.summary["best_score"] # wandb's summary returns the last score, not the best one, so I just ran another script to add that best_score value to the summary
        assert not math.isnan(m)
        dataset_results[dataset_name].append(m)
    except Exception as e:
        print(dataset_name, find_metric_name(run))
        print(run.summary)
        raise e

if __name__ == "__main__":
    # Compute the mean and std
    dataset_metrics = {}
    for name, metrics in dataset_results.items():
        metrics_mean = sum(metrics) / len(metrics)
        dataset_metrics[name] = {
            "mean": metrics_mean,
            "std": math.sqrt(sum((m - metrics_mean) ** 2 for m in metrics) / (len(metrics) - 1)), # BESSEL CORRECTION
            "metric": metric_names[name]
        }

    # Store
    with open(Path(__file__).parent / "baseline_metrics.json", "w") as f:
        json.dump(dataset_metrics, f)
    with open(Path(__file__).parent / "baseline_results.json", "w") as f:
        json.dump(dataset_results, f)