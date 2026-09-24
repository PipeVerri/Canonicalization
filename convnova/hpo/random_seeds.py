import argparse
from src.hpo.full_eval import run_all_tasks
from src.hpo.original_paper_eval import run_on_select_tasks

RANDOM_SEEDS = [1938, 205, 8262, 1480, 4436]
#RANDOM_SEEDS = [8262, 1480, 4436]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-folder", type=str, required=True, help="The configs experiments folder path(excluding /.../experiments/)")
    parser.add_argument("--checkpoints-folder", type=str, required=True, help="Something like pretraining/raw_control_signal/random_seeds")
    parser.add_argument("--eval-mode", type=str, required=True, help="'paper' or 'hpo' mode")
    parser.add_argument("--selected-benchmark", type=str, help="For the paper mode, the name of the benchmark")
    parser.add_argument("--selected-tasks", type=str, help="For the paper mode, separated by comma. Leave empty for the whole benchmark")
    
    args, overrides = parser.parse_known_args()
    # Validate the overrides
    for o in overrides:
        if not "=" in o:
            raise ValueError(f"Invalid override {o}")

    if args.eval_mode not in ("paper", "hpo"):
        raise ValueError(f"{args.eval_mode} eval mode not recognized")

    for seed in RANDOM_SEEDS:
        if args.eval_mode == "hpo":
            run_all_tasks(
                call_overrides=[f"train.seed={seed}", *overrides], 
                runs_dir_name=args.checkpoints_folder,
                run_name=str(seed),
                experiments_folder=args.experiments_folder
            )
        elif args.eval_mode == "paper":
            assert args.selected_benchmark is not None
            selected_tasks = None
            if args.selected_tasks is not None:
                selected_tasks = args.selected_tasks.split(",")
            run_on_select_tasks(
                overrides_list=[f"train.seed={seed}", *overrides],
                runs_dir_name=args.checkpoints_folder,
                run_name=str(seed),
                experiments_folder=args.experiments_folder,
                benchmark_name=args.selected_benchmark,
                eval_only_on_tasks=selected_tasks
            )