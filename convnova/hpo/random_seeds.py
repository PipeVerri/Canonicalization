from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from eval_benchmark import benchmark_tasks, eval_model_on_tasks, run_experiment


RANDOM_SEEDS = 3

def run_all_tasks(call_overrides, runs_dir: Path, run_name: str, output_json=None | str, task_specific_overrides={}):
    # Pretrain
    checkpoint_dir = PROJECT_ROOT / "outputs" / runs_dir / run_name
    pretraining_overrides = [
        f"hydra.run.dir={str(checkpoint_dir.absolute())}",
        *call_overrides
    ]
    if task_specific_overrides.get("pretraining"):
        pretraining_overrides += task_specific_overrides["pretraining"]

    run_experiment("canon/pretrain", pretraining_overrides, str(output_json), f"{run_name}.pretraining")
    checkpoint_file_str = str((checkpoint_dir / "checkpoints/test/loss.ckpt").absolute())

    base_downstream_task_overrides = ( # Make it a tuple so I don't accidentally edit it
        f"train.pretrained_model_path={checkpoint_file_str}",
        *call_overrides
    )

    # NT benchmark
    nt_benchmark_tasks = benchmark_tasks["nt_benchmark"]
    nt_benchmark_overrides = list(base_downstream_task_overrides)
    if task_specific_overrides.get("nt_benchmark"):
        nt_benchmark_overrides += task_specific_overrides["nt_benchmark"]
    eval_model_on_tasks("canon/nt_benchmark", nt_benchmark_tasks, nt_benchmark_overrides, str(output_json), f"{run_name}.nt_benchmark")

    # Genomic benchmark
    genomic_benchmark_tasks = benchmark_tasks["genomic_benchmark"]
    genomic_benchmark_overrides = list(base_downstream_task_overrides)
    if task_specific_overrides.get("genomic_benchmark"):
        genomic_benchmark_overrides += task_specific_overrides["genomic_benchmark"]
    eval_model_on_tasks("canon/genomic_benchmark", genomic_benchmark_tasks, genomic_benchmark_overrides, str(output_json), f"{run_name}.genomic_benchmark")

    # BEND benchmark
    bend_benchmark_tasks = benchmark_tasks["bend_benchmark"]
    bend_benchmark_overrides = list(base_downstream_task_overrides)
    if task_specific_overrides.get("bend_benchmark"):
        bend_benchmark_overrides += task_specific_overrides["bend_benchmark"]
    eval_model_on_tasks("canon/bend_benchmark", bend_benchmark_tasks, bend_benchmark_overrides, str(output_json), f"{run_name}.bend_benchmark")

    # DeepSEA
    deepsea_overrides = call_overrides.copy()
    if task_specific_overrides.get("chromatin_profile"):
        deepsea_overrides += task_specific_overrides["chromatin_profile"]
    run_experiment("canon/chromatin_profile", deepsea_overrides, str(output_json), f"{run_name}.chromatin_profile_prediction")

for _ in range(RANDOM_SEEDS):
    seed = random.randint(0, 10_000)
    while seed == 2222 or seed == 2805:
        seed = random.randint(0, 10_000)

    run_all_tasks(
        [f"train.seed={seed}"], 
        "pretraining/baseline/random_seeds",
        str(seed),
        output_json="./random_seeds.json"
    )
