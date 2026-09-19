import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import wandb
from tqdm import tqdm

PROJECT = "pipe-verri-gh/Canonicalization"
MAX_WORKERS = 8
SKIP_EXISTING = True  # no recalcular si el run ya tiene best_score

api = wandb.Api()
runs = list(api.runs(PROJECT, filters={"group": "baseline_paper"}))


def process(run):
    job_type = run.config["wandb"]["job_type"]
    if job_type == "pretraining":
        return run.id, "skipped (pretraining)"

    if SKIP_EXISTING and "best_score" in run.summary.keys():
        return run.id, "skipped (ya tiene best_score)"

    if job_type == "chromatin_profile_prediction":
        dataset_name = "chromatin_profile_prediction"
    else:
        dataset_name = run.config["dataset"]["dataset_name"]

    metric = "val/accuracy" if dataset_name != "pretraining" else "val/loss"
    if "val/mcc" in run.summary.keys():
        metric = "val/mcc"
    elif "val/auroc" in run.summary.keys():
        metric = "val/auroc"

    vals = [
        row[metric]
        for row in run.scan_history(keys=[metric])
        if row.get(metric) is not None and not math.isnan(row[metric])
    ]
    if not vals:
        raise ValueError(f"sin valores para {metric}")

    best = min(vals) if metric == "val/loss" else max(vals)

    run.summary["best_score"] = best
    run.summary.update()
    return run.id, f"{metric} -> {best:.6f}"


failed = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(process, run): run for run in runs}
    with tqdm(total=len(futures), desc="Runs", unit="run") as pbar:
        for fut in as_completed(futures):
            run = futures[fut]
            try:
                run_id, msg = fut.result()
                tqdm.write(f"[ok] {run_id}: {msg}")
            except Exception as e:
                failed.append(run.id)
                tqdm.write(f"[error] {run.id}: {e!r}")
            pbar.update(1)
            pbar.set_postfix(errores=len(failed))

print(f"\nTerminado. {len(runs) - len(failed)}/{len(runs)} ok, {len(failed)} con error.")
if failed:
    print("Fallidos:", failed)