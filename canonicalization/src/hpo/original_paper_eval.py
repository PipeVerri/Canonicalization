from pathlib import Path
from .benchmarks import run_experiment, eval_model_on_tasks, benchmark_tasks

ROOT_DIR = Path(__file__).resolve().parents[2]

def run_on_select_tasks(overrides_list: list, runs_dir_name: str, run_name: str, experiments_folder: str, benchmark_name: str, eval_only_on_tasks: None | list = None, skip_pretrain=False):
    global ROOT_DIR
    if not benchmark_name in ("nt_benchmark", "genomic_benchmark", "bend_benchmark", "chromatin_profile_prediction"):
        raise RuntimeError(f"{benchmark_name} is not a recognized benchmark")
    
    checkpoint_dir = ROOT_DIR / "outputs" / runs_dir_name / run_name
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pretraining_overrides = [
        f"hydra.run.dir={str(checkpoint_dir.absolute())}",
        *overrides_list
    ]

    if not skip_pretrain:
        run_experiment(f"{experiments_folder}/pretrain_{benchmark_name}", pretraining_overrides)
    
    checkpoint_file_str = str((checkpoint_dir / "checkpoints/test/loss.ckpt").absolute())

    base_downstream_task_overrides = ( # Make it a tuple so I don't accidentally edit it
        f"train.pretrained_model_path={checkpoint_file_str}",
        *overrides_list
    )

    if benchmark_name != "chromatin_profile_prediction":
        if eval_only_on_tasks is None:
            eval_model_on_tasks(f"{experiments_folder}/{benchmark_name}", benchmark_tasks[benchmark_name], base_downstream_task_overrides)
        else:
            eval_model_on_tasks(f"{experiments_folder}/{benchmark_name}", eval_only_on_tasks, base_downstream_task_overrides)
    else:
         run_experiment(f"{experiments_folder}/chromatin_profile_prediction", pretraining_overrides)