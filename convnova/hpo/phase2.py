import optuna
from pathlib import Path
from src.hpo.optimize_study import optimize_study

ROOT_DIR = Path(__file__).parents[1]

def build_overrides_dict(trial: optuna.Trial):
    return {
        "canonicalization": {
            "annotations_level": trial.suggest_categorical("annotations_level", [1, 2, 3]),
            "include_cdjv_segments": trial.suggest_categorical("include_cdjv_segments", ["true", "false"]),
            "include_pseudogenes": trial.suggest_categorical("include_pseudogenes", ["true", "false"]),
            "orient_all_segments": trial.suggest_categorical("orient_all_segments", ["true", "false"]),
            "partial_canonicalization_works": trial.suggest_categorical("partial_canonicalization_works", ["true", "false"]),
            "transformation": trial.suggest_categorical("transformation", ["reverse", "complement", "rc"])
        },
        # Skip control signal search. It's only for the ablation
        #"dataset": {
        #    "use_control_signal": trial.suggest_categorical("suggest_control_signal", ["true", "false"])
        #},
        "wandb": {
            "group": "hpo_phase_1",
            "mode": "online"
        }
    }

sampler = optuna.samplers.TPESampler(
    n_startup_trials=6, # 30% of 20 total runs
    multivariate=True, # hparams interactions are really important
    group=False # All hparams coexist
)

optimize_study(
    sampler=sampler,
    study_name="phase_1",
    hpo_dir=ROOT_DIR / "outputs/hpo/phase_1",
    checkpoints_dir_name="hpo_phase_1",
    baseline_metrics_path=Path(__file__).parent / "baseline_metrics.json",
    overrides_dict_method=build_overrides_dict,
    n_trials=20
)