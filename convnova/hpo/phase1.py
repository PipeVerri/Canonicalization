import optuna

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

    