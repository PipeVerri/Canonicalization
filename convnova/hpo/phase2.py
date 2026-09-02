import optuna
from pathlib import Path
from src.hpo.optimize_study import optimize_study

ROOT_DIR = Path(__file__).parents[1]

def build_overrides_dict(trial: optuna.Trial):
    overlap_transformation_behavior = trial.suggest_categorical("overlapping_segments_canonicalization", ["partial_break", "full_break", "both_break"])
    if overlap_transformation_behavior == "partial_break":
        canon_dict = {"overlap_transformation_behavior": "partial_break"}
    elif overlap_transformation_behavior == "full_break":
        orient_all_segments = trial.suggest_categorical("orient_all_segments", [True, False]),
        if orient_all_segments:
            canon_dict = {
                "overlap_transformation_behavior": "full_break", 
                "orient_all_segments": orient_all_segments,
                "keep_segments_order": trial.suggest_categorical("keep_segments_order", [True, False])
            }
        else:
            canon_dict = {
                "overlap_transformation_behavior": "full_break", 
                "orient_all_segments": orient_all_segments
            }
    else:
        canon_dict = {"overlap_transformation_behavior": "both_break"}

    return {
        "canonicalization": canon_dict,
        "wandb": {
            "group": "overlapping_segments_ablation",
            "mode": "online"
        }
    }

# Exhaustively search the hyperparameter space. Grid search doesn't take into account the conditional hyperparameter selection, and ends up doing repeats
sampler = optuna.samplers.BruteForceSampler()

optimize_study(
    sampler=sampler,
    study_name="overlapping_segments_ablation",
    hpo_dir=ROOT_DIR / "outputs/hpo/ablations/overlapping_segments",
    checkpoints_dir_name="hpo_overlapping_segments_ablation",
    baseline_metrics_path=Path(__file__).parent / "baseline_metrics.json",
    overrides_dict_method=build_overrides_dict,
    n_trials=6 # 1 for partial break + 4 for full break + 1 for both break = 6
)