import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import wandb
from tqdm import tqdm
from .compute_baseline_z_scores import find_metric_name

PROJECT = "pipe-verri-gh/Canonicalization"
MAX_WORKERS = 8
SKIP_EXISTING = True  # no recalcular si el run ya tiene best_score

api = wandb.Api()
runs = list(api.runs(PROJECT, filters={"group": "partial_downstream_canonicalization"}))

def process(run):
    job_type = run.config["wandb"]["job_type"]
    if job_type == "pretraining":
        return run.id, "skipped (pretraining)"

    if SKIP_EXISTING and "best_score" in run.summary.keys():
        return run.id, "skipped (ya tiene best_score)"

    metric = find_metric_name(run)

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


if __name__ == "__main__":
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