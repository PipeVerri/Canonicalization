import optuna
from .random_seeds import run_all_tasks
from pathlib import Path
import sys
import subprocess
import os
import json

ROOT_DIR = Path(__file__).parents[1]
json_file = ROOT_DIR / "outputs/hpo/hpo_phase_1_outputs.json"
json_file.parent.mkdir(exist_ok=True, parents=True)

def to_absolute(path):
    return str(path.absolute())

def compute_z_score(dataset_name, summary):
    global baseline_metrics
    score = summary[baseline_metrics[dataset_name]["metric"]]
    dataset_metrics = baseline_metrics[dataset_name]
    return (score - dataset_metrics["mean"]) / dataset_metrics["std"]

with open(Path(__file__).parent / "baseline_metrics.json", "r") as f:
    baseline_metrics = json.load(f)

TASKS_GROUPS = {
    "histone": [
        "H2AFZ",
        "H3K27ac",
        "H3K27me3",
        "H3K36me3",
        "H3K4me1",
        "H3K4me2",
        "H3K4me3",
        "H3K9ac",
        "H3K9me3",
        "H4K20me1",
    ],
    "promoter": [
        "promoter_all",
        "promoter_tata",
        "promoter_no_tata",
        "human_nontata_promoters",
    ],
    "enhancers": [
        "enhancers",
        "enhancers_types",
        "dummy_mouse_enhancers_ensembl",
        "human_enhancers_cohn",
        "human_enhancers_ensembl",
        "human_ensembl_regulatory",
        "human_ocr_ensembl",
        "drosophila_enhancers_stark"
    ],
    "splicing": [
        "splice_sites_all",
        "splice_sites_donors",
        "splice_sites_acceptors",
    ],
    "general_genomic_content": [
        "demo_coding_vs_intergenomic_seqs",
        "demo_human_or_worm",
    ],
    "gene_finding": ["gene_finding"],
    "cpg_methylation": ["cpg_methylation"],
    "chromatin_profile_prediction": ["chromatin_profile_prediction"]
}

def objective(trial: optuna.Trial):
    global ROOT_DIR, json_file

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
        },
        "model": {
            "use_control_signal": trial.suggest_categorical("suggest_control_signal", ["true", "false"])
        },
        "wandb": {
            "group": "hpo_phase_1"
        }
    }
    
    overrides_list = []
    for category in overrides.keys():
        for key, val in overrides[category].items():
            overrides_list.append(f"{category}.{key}={val}")

    trial_name = str(trial.number)

    # Run the canonicalization script
    cmd = [sys.executable, ROOT_DIR / "scripts/canonicalize_genomes.py", "experiment='canon/pretrain_canonicalized'", *overrides_list]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR) + ":" + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=ROOT_DIR, env=env, text=True)

    # Eval the model on all tasks
    run_all_tasks(overrides_list, ROOT_DIR / "outputs", trial_name, output_json=str(json_file.absolute()), task_specific_overrides={
        "pretraining": {
            "dataset.fasta_file='{}'".format(to_absolute(ROOT_DIR / "data/parsed/canonicalized.fasta"))
        },
        "nt_benchmark": {
            "dataset.dest_path='{}'".format(to_absolute(ROOT_DIR / "data/nucleotide_transformer_parsed/"))
        },
        "genomic_benchmark": {
            "dataset.dest_path='{}'".format(to_absolute(ROOT_DIR / "data/genomic_benchmark_parsed/"))
        },
        "bend_benchmark": {
            "dataset.dataset_dir='{}'".format(to_absolute(ROOT_DIR / "data/bend_benchmark_parsed/"))
        }
        "chromatin_profile": {
            "dataset.dataset_path='{}'".format(to_absolute(ROOT_DIR / "data/deepsea_parsed/"))
        }
    })

    # Read the json and compute the scores
    with open(json_file, "r") as f:
        runs_scores = json.load(f)[trial_name]

    runs_z_scores = {}
    for task_name, val in runs_scores.items():
        if type(val) == dict:
            for dataset_name, summary in val.items():
                runs_z_scores[dataset_name] = compute_z_score(dataset_name, summary)
        else:
            runs_z_scores[task_name] = compute_z_score(task_name, val)

    # Because there are a lot of similar tasks in NT benchmark and genomic benchmark compared to the NT, I will do a grouped mean for each task type(enhancers, histones, etc)
    groups_means = []
    for group_tasks in TASKS_GROUPS.values():
        mean = 0
        for task in group_tasks:
            mean += runs_z_scores[task]
        groups_means.append(mean / len(group_tasks))

    global_mean = sum(groups_means) / len(groups_means)
    return global_mean