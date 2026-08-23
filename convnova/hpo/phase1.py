import optuna
from random_seeds import run_all_tasks
from pathlib import Path
import sys
import subprocess
import os
import json
from pprint import pprint
import sqlite3

ROOT_DIR = Path(__file__).parents[1]
HPO_DIR = ROOT_DIR / "outputs/hpo/phase_1"
json_file = HPO_DIR / "outputs.json"
json_file.parent.mkdir(exist_ok=True, parents=True)

def to_absolute(path):
    return str(path.absolute())

def compute_z_score(dataset_name, summary):
    global baseline_metrics
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
        # Skip control signal search. It's only for the ablation
        #"dataset": {
        #    "use_control_signal": trial.suggest_categorical("suggest_control_signal", ["true", "false"])
        #},
        "wandb": {
            "group": "hpo_phase_1",
            "mode": "online"
        }
    }

    print("="*50)
    print("Overrides are:")
    pprint(overrides)
    print("="*50)

    trial_name = str(trial.number)
    
    overrides_list = [
        "+canonicalization=default", 
        #f"+model.use_control_signal={str(overrides['dataset']['use_control_signal']).lower()}",
        f"+wandb.tags=['trial_{trial_name}']"
    ]
    for category in overrides.keys():
        for key, val in overrides[category].items():
            overrides_list.append(f"{category}.{key}={val}")

    failed_trial = trial.user_attrs.get("resuming_failed", False)
    if not failed_trial:
        # Run the canonicalization script
        cmd = [sys.executable, ROOT_DIR / "scripts/canonicalize_genomes.py", "experiment='canon/pretrain'", *overrides_list]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT_DIR) + ":" + env.get("PYTHONPATH", "")
        subprocess.run(cmd, cwd=ROOT_DIR, env=env, text=True)

    # Eval the model on all tasks
    skip_tasks = set(trial.user_attrs.get("skip_tasks", []))
    run_all_tasks(overrides_list, "hpo_phase_1_checkpoints", trial_name, output_json=str(json_file.absolute()), task_specific_overrides={
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
    with open(json_file, "r") as f:
        runs_scores = json.load(f)[trial_name]

    runs_z_scores = {}
    for task_name, val in runs_scores.items():
        if task_name == "pretraining":
            continue

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


class FdTee:
    def __init__(self, log_path):
        self.log_path = log_path

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()

        self._original_fd1 = os.dup(1)
        self._original_fd2 = os.dup(2)

        self._tee = subprocess.Popen(["tee", str(self.log_path)], stdin=subprocess.PIPE)

        os.dup2(self._tee.stdin.fileno(), 1)  # stdout -> tee
        os.dup2(self._tee.stdin.fileno(), 2)  # stderr -> tee también
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()

        os.dup2(self._original_fd1, 1)
        os.dup2(self._original_fd2, 2)
        os.close(self._original_fd1)
        os.close(self._original_fd2)

        self._tee.stdin.close()
        self._tee.wait()

def objective_stdout_wrapper(trial: optuna.Trial):
    global HPO_DIR
    log_path = HPO_DIR / f"logs/trial_{trial.number}.log"
    log_path.parent.mkdir(exist_ok=True)
    with FdTee(log_path):
        return objective(trial)

# Before creating the study, check if I have 2 failed runs in a row, if that's the case, then delete the last failed run before loading the study
optuna_db_path = HPO_DIR/'optuna.db'
conn = sqlite3.connect(optuna_db_path)
cursor = conn.execute("SELECT * FROM trials ORDER BY trial_id DESC LIMIT 2")
trials = cursor.fetchall()
pprint(f"Trials: {trials}")
if trials is not None and len(trials) >= 2:
    tables_to_delete = [
        "trials",
        "trial_user_attributes",
        "trial_system_attributes",
        "trial_params",
        "trial_values",
        "trial_intermediate_values",
        "trial_heartbeats",
    ]

    if trials[0][3] == "FAIL" and trials[1][3] == "FAIL":
        for t in tables_to_delete:
            conn.execute(f"DELETE FROM '{t}' WHERE trial_id = {trials[0][0]};")

conn.commit()
conn.close()

sampler = optuna.samplers.TPESampler(
    n_startup_trials=6, # 30% of 20 total runs
    multivariate=True, # hparams interactions are really important
    group=False # All hparams coexist
)
study = optuna.create_study(study_name="phase_1", storage=f"sqlite:///{optuna_db_path}", load_if_exists=True, direction="maximize", sampler=sampler)

# Checks if it should restart a failed run
last_trial = study.trials[-1]
if last_trial.state == optuna.trial.TrialState.FAIL:
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

study.optimize(objective_stdout_wrapper, n_trials=20) # Low trials until representative tasks are found
print(f"Best params are:")
print(study.best_params)
