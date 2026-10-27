from regex import T

import optuna
from pathlib import Path
from src.hpo.optimize_study import optimize_study

ROOT_DIR = Path(__file__).parents[1]

def build_overrides_dict(trial: optuna.Trial):
    overlap_transformation_behavior = trial.suggest_categorical("overlapping_segments_canonicalization", ["partial_break", "full_break", "both_break"])
    if overlap_transformation_behavior == "partial_break":
        overlaps_overrides = {
            "overlap_transformation_behavior": "partial_break",
        }
    elif overlap_transformation_behavior == "full_break":
        orient_all_segments = trial.suggest_categorical("orient_all_segments", [True, False])
        if orient_all_segments:
            overlaps_overrides = {
                "overlap_transformation_behavior": "full_break", 
                "full_break_config": {
                    "orient_all_segments": orient_all_segments,
                    "keep_segments_order": trial.suggest_categorical("keep_segments_order", [True, False]),
                }
            }
        else:
            overlaps_overrides = {
                "overlap_transformation_behavior": "full_break", 
                "full_break_config": {
                    "orient_all_segments": False
                    # keep_segments_order doesn't matter
                }
            }
    else:
        overlaps_overrides = {
            "overlap_transformation_behavior": "both_break",
        }

    return {
        "canonicalization": {
            "annotations_level": trial.suggest_categorical("annotations_level", [1, 2, 3]),
            "include_cdjv_segments": trial.suggest_categorical("include_cdjv_segments", ["true", "false"]),
            "include_pseudogenes": trial.suggest_categorical("include_pseudogenes", ["true", "false"]),
            "transformation": trial.suggest_categorical("transformation", ["reverse", "complement", "rc"]),
            "should_rc_at_end": trial.suggest_categorical("should_rc_at_end", [True, False]),
            "partial_downstream_canonicalization": trial.suggest_categorical("partial_downstream_canonicalization", [True, False])
        } | overlaps_overrides,
        "model": {
            "switch_strands": trial.suggest_categorial("switch_strands", [True, False])
        },
        "wandb": {
            "group": "hpo",
            "mode": "online"
        },
        "train": {
            "seed": 2222
        }
    }

sampler = optuna.samplers.TPESampler(
    n_startup_trials=5,
    multivariate=True, # hparams interactions are really important
    group=True # Not all hparams coexist
)

optimize_study(
    sampler=sampler,
    study_name="hpo",
    hpo_dir=ROOT_DIR / "outputs/hpo/fixed_hpo",
    checkpoints_dir_name="hpo_checkpoints",
    baseline_metrics_path=Path(__file__).parent / "baseline_metrics.json",
    overrides_dict_method=build_overrides_dict,
    n_trials=20
)