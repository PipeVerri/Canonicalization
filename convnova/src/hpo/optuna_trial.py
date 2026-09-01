import optuna
from pprint import pprint
import sys
import os
import subprocess
from pathlib import Path
from .full_eval import run_all_tasks
import json

ROOT_DIR = Path(__file__).parents[2]

def to_absolute(path):
    return str(path.absolute())

def compute_z_score(baseline_metrics: dict, dataset_name: str, summary):
    #print(f"Summary is:")
    #pprint(summary)
    #print("="*50)
    #print("baseline_metrics is:")
    #pprint(baseline_metrics)
    #print("="*50)
    #print("baseline_metrics, metric is:")
    #pprint(baseline_metrics[dataset_name])

    dataset_metrics = baseline_metrics[dataset_name]
    score = summary[0][dataset_metrics["metric"]]
    return (score - dataset_metrics["mean"]) / dataset_metrics["std"]

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

def compute_trial_grouped_z_score(json_file: Path, baseline_metrics: dict, trial_name: str):
    with open(json_file, "r") as f:
        runs_scores = json.load(f)[trial_name]

    runs_z_scores = {}
    for task_name, val in runs_scores.items():
        if task_name == "pretraining":
            continue

        if type(val) == dict:
            for dataset_name, summary in val.items():
                runs_z_scores[dataset_name] = compute_z_score(baseline_metrics, dataset_name, summary)
        else:
            runs_z_scores[task_name] = compute_z_score(baseline_metrics, task_name, val)

    # Because there are a lot of similar tasks in NT benchmark and genomic benchmark compared to the NT, I will do a grouped mean for each task type(enhancers, histones, etc)
    groups_means = []
    for group_tasks in TASKS_GROUPS.values():
        mean = 0
        for task in group_tasks:
            mean += runs_z_scores[task]
        groups_means.append(mean / len(group_tasks))

    global_mean = sum(groups_means) / len(groups_means)
    return global_mean

def recursively_build_overrides_list(current_overrides_dict: dict, prefix: str) -> list:
    overrides = []
    for key, val in current_overrides_dict.items():
        if type(val) == dict:
            if prefix != "":
                new_prefix = f"{prefix}.{key}"
            else:
                new_prefix = key
            overrides.extend(recursively_build_overrides_list(val, new_prefix))
        else:
            if prefix != "":
                new_prefix = f"{prefix}.{key}"
            else:
                new_prefix = key
            overrides.append(f"{new_prefix}={val}")
    return overrides

def _objective(overrides_dict: dict, trial: optuna.Trial, hpo_dir: Path, checkpoints_dir_name: str, baseline_metrics: dict):
    global ROOT_DIR

    json_file = hpo_dir / "outputs.json"
    json_file.parent.mkdir(exist_ok=True, parents=True)

    print("="*50)
    print("Overrides are:")
    pprint(overrides_dict)
    print("="*50)

    trial_name = str(trial.number)
    
    overrides_list = [
        "+canonicalization=default", 
        #f"+model.use_control_signal={str(overrides['dataset']['use_control_signal']).lower()}",
        f"+wandb.tags=['trial_{trial_name}']"
    ]
    overrides_list.extend(recursively_build_overrides_list(overrides_dict, ""))

    failed_trial = trial.user_attrs.get("resuming_failed", False)
    if not failed_trial:
        # Run the canonicalization script
        cmd = [sys.executable, ROOT_DIR / "scripts/canonicalize_genomes.py", "experiment='canon/pretrain'", *overrides_list]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT_DIR) + ":" + env.get("PYTHONPATH", "")
        subprocess.run(cmd, cwd=ROOT_DIR, env=env, text=True)

    # Eval the model on all tasks
    skip_tasks = set(trial.user_attrs.get("skip_tasks", []))
    run_all_tasks(overrides_list, checkpoints_dir_name, trial_name, output_json=str(json_file.absolute()), task_specific_overrides={
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
        },
        "chromatin_profile": {
            "dataset.dataset_path='{}'".format(to_absolute(ROOT_DIR / "data/deepsea_parsed/"))
        }
    }, skip_tasks=skip_tasks, failed_checkpoint_id=str(trial.number - 1) if failed_trial else None)

    # Read the json and compute the scores
    return compute_trial_grouped_z_score(json_file=json_file, baseline_metrics=baseline_metrics, trial_name=trial_name)

class FdTee:
    def __init__(self, log_path):
        self.log_path = log_path

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()

        self._original_fd1 = os.dup(1)
        self._original_fd2 = os.dup(2)

        self._tee = subprocess.Popen(["tee", str(self.log_path)], stdin=subprocess.PIPE)

        os.dup2(self._tee.stdin.fileno(), 1)  # type: ignore # stdout -> tee
        os.dup2(self._tee.stdin.fileno(), 2)  # type: ignore # stderr -> tee también
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()

        os.dup2(self._original_fd1, 1)
        os.dup2(self._original_fd2, 2)
        os.close(self._original_fd1)
        os.close(self._original_fd2)

        self._tee.stdin.close() # type: ignore
        self._tee.wait()

def objective(overrides_dict: dict, trial: optuna.Trial, hpo_dir: Path, checkpoints_dir_name: str, baseline_metrics: dict):
    log_path = hpo_dir / f"logs/trial_{trial.number}.log"
    log_path.parent.mkdir(exist_ok=True)
    with FdTee(log_path):
        return _objective(overrides_dict, trial, hpo_dir, checkpoints_dir_name, baseline_metrics)