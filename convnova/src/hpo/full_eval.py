from pathlib import Path
from .benchmarks import run_experiment, benchmark_tasks, eval_model_on_tasks

ROOT_DIR = Path(__file__).resolve().parents[1]

def run_all_tasks(call_overrides, runs_dir_name: str, run_name: str, output_json: str | None = None, task_specific_overrides={}, skip_tasks=set(), failed_checkpoint_id=None):
    global ROOT_DIR

    # Pretrain
    checkpoint_dir = ROOT_DIR / "outputs" / runs_dir_name / (run_name if failed_checkpoint_id is None else failed_checkpoint_id)
    pretraining_overrides = [
        f"hydra.run.dir={str(checkpoint_dir.absolute())}",
        *call_overrides
    ]
    if task_specific_overrides.get("pretraining"):
        pretraining_overrides += task_specific_overrides["pretraining"]

    if not "pretraining" in skip_tasks:
        run_experiment("canon/pretrain", pretraining_overrides, str(output_json), f"{run_name}.pretraining")
    checkpoint_file_str = str((checkpoint_dir / "checkpoints/test/loss.ckpt").absolute())

    base_downstream_task_overrides = ( # Make it a tuple so I don't accidentally edit it
        f"train.pretrained_model_path={checkpoint_file_str}",
        *call_overrides
    )

    # NT benchmark
    nt_benchmark_tasks = list(set(benchmark_tasks["nt_benchmark"]) - skip_tasks)
    nt_benchmark_overrides = list(base_downstream_task_overrides)
    if task_specific_overrides.get("nt_benchmark"):
        nt_benchmark_overrides += task_specific_overrides["nt_benchmark"]
    eval_model_on_tasks("canon/nt_benchmark", nt_benchmark_tasks, nt_benchmark_overrides, str(output_json), f"{run_name}.nt_benchmark")

    # Genomic benchmark
    genomic_benchmark_tasks = list(set(benchmark_tasks["genomic_benchmark"]) - skip_tasks)
    genomic_benchmark_overrides = list(base_downstream_task_overrides)
    if task_specific_overrides.get("genomic_benchmark"):
        genomic_benchmark_overrides += task_specific_overrides["genomic_benchmark"]
    eval_model_on_tasks("canon/genomic_benchmark", genomic_benchmark_tasks, genomic_benchmark_overrides, str(output_json), f"{run_name}.genomic_benchmark")

    # BEND benchmark
    bend_benchmark_tasks = list(set(benchmark_tasks["bend_benchmark"]) - skip_tasks)
    bend_benchmark_overrides = list(base_downstream_task_overrides)
    if task_specific_overrides.get("bend_benchmark"):
        bend_benchmark_overrides += task_specific_overrides["bend_benchmark"]
    eval_model_on_tasks("canon/bend_benchmark", bend_benchmark_tasks, bend_benchmark_overrides, str(output_json), f"{run_name}.bend_benchmark")

    # DeepSEA
    deepsea_overrides = call_overrides.copy()
    if task_specific_overrides.get("chromatin_profile"):
        deepsea_overrides += task_specific_overrides["chromatin_profile"]
    if not "chromatin_profile_prediction" in skip_tasks:
        run_experiment("canon/chromatin_profile", deepsea_overrides, str(output_json), f"{run_name}.chromatin_profile_prediction")
