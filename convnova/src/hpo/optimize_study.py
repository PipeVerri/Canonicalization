import optuna
from .optuna_trial import objective
from pathlib import Path
import json
from .optuna_db_cleanup import remove_two_failed_trials_in_a_row

def optimize_study(sampler: optuna.samplers.BaseSampler, study_name: str, hpo_dir: Path, checkpoints_dir_name: str, baseline_metrics_path: Path, overrides_dict_method, n_trials: int):
    # Load the baseline_metrics
    with open(baseline_metrics_path, "r") as f:
        baseline_metrics = json.load(f)
        
    # Clean up the sqlite file
    optuna_db_path = hpo_dir / "optuna.db"
    if not optuna_db_path.exists():
        print("Warning!!!! optuna db path doesn't exist")
        optuna_db_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        remove_two_failed_trials_in_a_row(study_name, optuna_db_path)

    study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{optuna_db_path}", load_if_exists=True, direction="maximize", sampler=sampler)

    # Checks if it should restart a failed run
    last_trial = study.trials[-1] if study.trials else None
    json_file = hpo_dir / "outputs.json"

    if (last_trial is not None) and (last_trial.state == optuna.trial.TrialState.FAIL):
        # Rename the json so it uses the new trial number
        with open(json_file, "r") as f:
            runs_scores = json.load(f)
        
        # Check if there was even a successful run, if not, skip it
        if str(last_trial.number) in runs_scores.keys():
            if not str(last_trial.number + 1) in runs_scores.keys():
                runs_scores[str(last_trial.number + 1)] = runs_scores[str(last_trial.number)]
            
            with open(json_file, "w") as f:
                json.dump(runs_scores, f)
        
            # Construct the skip_tasks dict
            skip_tasks = []
            if "pretraining" in runs_scores[str(last_trial.number + 1)]:
                skip_tasks.append("pretraining")
            if "chromatin_profile_prediction" in runs_scores[str(last_trial.number + 1)]:
                skip_tasks.append("chromatin_profile_prediction")
            skip_tasks += list(runs_scores[str(last_trial.number + 1)].get("nt_benchmark", {}).keys())
            skip_tasks += list(runs_scores[str(last_trial.number + 1)].get("genomic_benchmark", {}).keys())
            skip_tasks += list(runs_scores[str(last_trial.number + 1)].get("bend_benchmark", {}).keys())
        
            # Schedule the run so it uses the same trial config
            study.enqueue_trial(last_trial.params, user_attrs={
                "resuming_failed": True,
                "skip_tasks": skip_tasks,
            })

    def objective_wrapper(trial: optuna.Trial):
        nonlocal hpo_dir, baseline_metrics, checkpoints_dir_name
        overrides_dict = overrides_dict_method(trial)
        return objective(overrides_dict, trial, hpo_dir, checkpoints_dir_name, baseline_metrics)

    study.optimize(objective_wrapper, n_trials=n_trials)