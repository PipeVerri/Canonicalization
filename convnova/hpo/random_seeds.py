import subprocess
import sys
import os
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RANDOM_SEEDS = 2

def run_experiment(experiment, overrides, output_json=None):
    """Run a single experiment via the hpo_runner CLI."""
    cmd = [
        sys.executable, "-m", "src.utils.hpo_runner",
        experiment,
        *overrides,
    ]
    if output_json:
        cmd.extend(["--output-json", output_json])
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")
    
    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed with code {result.returncode}")

for _ in range(RANDOM_SEEDS):
    import os
    seed = random.randint(0, 10_000)
    output_dir = PROJECT_ROOT / "outputs"
    output_json = output_dir / f"random_seeds_results_{seed}.json"

    # Pretrain
    checkpoint_dir = output_dir / f"pretraining/baseline/random_seeds/{seed}/"
    overrides = [
        f"train.seed={seed}",
        f"hydra.run.dir={str(checkpoint_dir.absolute())}"
    ]
    run_experiment("canon/pretrain", overrides, str(output_json))

    checkpoint_file_str = str((checkpoint_dir / "checkpoints/test/loss.ckpt").absolute())

    # NT benchmark
    from eval_benchmark import benchmark_tasks, eval_model_on_tasks
    nt_benchmark_tasks = benchmark_tasks["nt_benchmark"]
    overrides = [
        f"train.seed={seed}",
        f"train.pretrained_model_path={checkpoint_file_str}"
    ]
    eval_model_on_tasks("canon/nt_benchmark", nt_benchmark_tasks, overrides, str(output_json))

    # Genomic benchmark
    genomic_benchmark_tasks = benchmark_tasks["genomic_benchmark"]
    eval_model_on_tasks("canon/genomic_benchmark", genomic_benchmark_tasks, overrides, str(output_json))

    # BEND benchmark
    bend_benchmark_tasks = benchmark_tasks["bend_benchmark"]
    eval_model_on_tasks("canon/bend_benchmark", bend_benchmark_tasks, overrides, str(output_json))

    # DeepSEA
    overrides = [
        f"train.seed={seed}"
    ]
    run_experiment("canon/chromatin_profile", overrides, str(output_json))