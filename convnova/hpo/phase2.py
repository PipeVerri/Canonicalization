import optuna
from pathlib import Path
from src.hpo.optimize_study import optimize_study

ROOT_DIR = Path(__file__).parents[1]

def build_overrides_dict(trial: optuna.Trial):
    return {
        "canonicalization": {
            "overlap_transformation_behavior": trial.suggest_categorical("overlapping_segments_canonicalization", ["partial_break", "full_break", "both_break"])
        },
        "wandb": {
            "group": "overlapping_segments_ablation",
            "mode": "online"
        }
    }

sampler = optuna.samplers.GridSampler({
    "overlapping_segments_canonicalization": ["partial_break", "full_break", "both_break"],
    "orient_all_segments": [True, False],
    "keep_segments_order": [True, False]
})

optimize_study(
    sampler=sampler,
    study_name="overlapping_segments_ablation",
    hpo_dir=ROOT_DIR / "outputs/hpo/ablations/overlapping_segments",
    checkpoints_dir_name="hpo_overlapping_segments_ablation",
    baseline_metrics_path=Path(__file__).parent / "baseline_metrics.json",
    overrides_dict_method=build_overrides_dict,
    n_trials=3
)