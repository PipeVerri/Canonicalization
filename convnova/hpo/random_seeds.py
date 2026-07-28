from eval_benchmark import benchmark_tasks, eval_model_on_tasks, run_experiment
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANDOM_SEEDS = 2

def run_all_tasks(call_overrides, runs_dir, run_name, output_json=None):
    # Pretrain
    checkpoint_dir = PROJECT_ROOT / "outputs" / runs_dir / run_name
    overrides = [
        f"hydra.run.dir={str(checkpoint_dir.absolute())}",
        *call_overrides
    ]
    run_experiment("canon/pretrain", overrides, str(output_json), run_name)

    checkpoint_file_str = str((checkpoint_dir / "checkpoints/test/loss.ckpt").absolute())

    # NT benchmark
    nt_benchmark_tasks = benchmark_tasks["nt_benchmark"]
    overrides = [
        f"train.pretrained_model_path={checkpoint_file_str}",
        *call_overrides
    ]
    eval_model_on_tasks("canon/nt_benchmark", nt_benchmark_tasks, overrides, str(output_json), run_name)

    # Genomic benchmark
    genomic_benchmark_tasks = benchmark_tasks["genomic_benchmark"]
    eval_model_on_tasks("canon/genomic_benchmark", genomic_benchmark_tasks, overrides, str(output_json), run_name)

    # BEND benchmark
    bend_benchmark_tasks = benchmark_tasks["bend_benchmark"]
    eval_model_on_tasks("canon/bend_benchmark", bend_benchmark_tasks, overrides, str(output_json), run_name)

    # DeepSEA
    run_experiment("canon/chromatin_profile", call_overrides, str(output_json), run_name)

for _ in range(RANDOM_SEEDS):
    seed = random.randint(0, 10_000)
    run_all_tasks(
        [f"train.seed={seed}"], 
        "pretraining/baseline/random_seeds",
        str(seed)
    )