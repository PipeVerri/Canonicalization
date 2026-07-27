from src.utils.hpo_runner import run_model_with_config_and_overrides
from ..eval_benchmark import benchmark_tasks, eval_model_on_tasks
from pathlib import Path
import random

RANDOM_SEEDS = 2

for _ in range(RANDOM_SEEDS):
    seed = random.randint(0, 10_000)
    output_dir = Path(__file__).parents[1] / "outputs"

    # Pretrain
    checkpoint_dir = output_dir / f"pretraining/baseline/random_seeds/{seed}/"
    overrides = [
        f"train.seed={seed}",
        f"hydra.run.dir={str(checkpoint_dir.absolute())}"
    ]
    run_model_with_config_and_overrides("canon/pretrain", overrides)

    checkpoint_file_str = str((checkpoint_dir / "checkpoints/test/loss.ckpt").absolute())

    # NT benchmark
    nt_benchmark_tasks = benchmark_tasks["nt_benchmark"]
    overrides = [
        f"train.seed={seed}",
        f"train.pretrained_model_path={checkpoint_file_str}"
    ]
    eval_model_on_tasks("canon/nt_benchmark", nt_benchmark_tasks, overrides)

    # Genomic benchmark
    genomic_benchmark_tasks = benchmark_tasks["genomic_benchmark"]
    eval_model_on_tasks("canon/genomic_benchmark", genomic_benchmark_tasks, overrides)

    # BEND benchmark
    bend_benchmark_tasks = benchmark_tasks["bend_benchmark"]
    eval_model_on_tasks("canon/bend_benchmark", bend_benchmark_tasks, overrides)

    # DeepSEA
    overrides = [
        f"train.seed={seed}"
    ]
    run_model_with_config_and_overrides("canon/chromatin_profile", overrides)