import optuna
from .random_seeds import run_all_tasks
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parents[1]
json_file = ROOT_DIR / "outputs/hpo/hpo_phase_1_outputs.json"
json_file.parent.mkdir(exist_ok=True, parents=True)

def objective(trial: optuna.Trial):
    overrides = {
        "canonicalization": {
            "annotations_level": trial.suggest_categorical("annotations_level", [1, 2, 3]),
            "include_cdjv_segments": trial.suggest_categorical("include_cdjv_segments", ["true", "false"]),
            "include_pseudogenes": trial.suggest_categorical("include_pseudogenes", ["true", "false"]),
            "orient_all_segments": trial.suggest_categorical("orient_all_segments", ["true", "false"]),
            "partial_canonicalization_works": trial.suggest_categorical("partial_canonicalization_works", ["true", "false"]),
            "transformation": trial.suggest_categorical("transformation", ["reverse", "complement", "rc"])
        },
        "dataset": {
            "use_control_signal": trial.suggest_categorical("suggest_control_signal", ["true", "false"])
        }
    }
    
    overrides_list = []
    for category in overrides.keys():
        for key, val in overrides[category].items():
            overrides_list.append(f"{category}.{key}={val}")
    
    trial_name = str(trial.number)
    # Run the canonicalization script
    cmd = []

    run_all_tasks(overrides_list, "hpo", trial_name, ROOT_DIR / "outputs", output_json=str(json_file.absolute()))

    # Read the json and compute the scores

    # TODO, add control signals to the rest of the datasets
    # TODO. When taking the RC of the reference chromosome at the end, is that correctly applied to the downstream tasks?
    # TODO: Test that I'm getting the same baseline numbers as the original paper 