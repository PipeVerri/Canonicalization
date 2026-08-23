import wandb
import math
import json
from pathlib import Path

api = wandb.Api()
runs = runs = api.runs("pipe-verri-gh/Canonicalization", filters={
    "group": "baseline"
})

dataset_results = {}
metric_names = {}

# Collect the metrics
for run in runs:
    if run.config["wandb"]["job_type"] == "pretraining":
        continue

    if run.config["wandb"]["job_type"] == "chromatin_profile_prediction":
        dataset_name = "chromatin_profile_prediction"
    else:
        dataset_name = run.config["dataset"]["dataset_name"]

    metric = "val/accuracy" if dataset_name != "pretraining" else "val/loss"
    if "val/mcc" in run.summary.keys():
        metric = "val/mcc"
    elif "val/auroc" in run.summary.keys():
        metric = "val/auroc"
    metric_names[dataset_name] = metric

    if not dataset_name in dataset_results:
        dataset_results[dataset_name] = []

    try:
        m = run.summary[metric]
        assert not math.isnan(m)
        dataset_results[dataset_name].append(m)
    except Exception as e:
        print(dataset_name, metric)
        print(run.summary)
        raise e

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