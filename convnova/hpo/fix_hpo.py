import optuna
import wandb
import sys
import os
from pathlib import Path
import subprocess
import json
from pprint import pprint

ROOT_DIR = Path(__file__).parents[1]

study = optuna.load_study(storage=f"sqlite:///{ROOT_DIR / 'outputs/hpo/phase_1/optuna.db'}", study_name="phase_1")
successful_trials = []

for trial in study.trials:
    if trial.state == optuna.trial.TrialState.COMPLETE:
        successful_trials.append(trial)

api = wandb.Api()
deepsea_wandb = api.runs("pipe-verri-gh/Canonicalization", filters={
    "group": "hpo_phase_1",
    "jobType": "chromatin_profile_prediction"
})

tags = [r.tags[0] for r in deepsea_wandb]

pretrained_model_paths = []

def to_absolute(path):
    return str(path.absolute())

def run_cmd(cmd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR) + ":" + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=ROOT_DIR, env=env, text=True)

for trial_tag in tags:
    genbench_wandb = api.runs("pipe-verri-gh/Canonicalization", filters={
        "group": "hpo_phase_1",
        "jobType": "genomic_benchmark",
        "tags": {"$in": [trial_tag]}
    })
    for run in genbench_wandb:
        pretrained_model_paths.append(run.config["train"]["pretrained_model_path"])
        break

for trial, pretrained_path in zip(successful_trials, pretrained_model_paths):
    overrides = [
        "+canonicalization=default",
        f"train.pretrained_model_path='{pretrained_path}'",
    ]
    for param, value in trial.params.items():
        overrides.append(f"canonicalization.{param}={value}")

    # Delete the entries from the json so the hpo_runner.py script writes into it
    # Optuna trial scores will be updated later
    with open(ROOT_DIR / "outputs/hpo/phase_1/outputs.json", "r") as f:
        all_runs_json = json.load(f)
    all_runs_json[str(trial.number)]["bend_benchmark"].pop("gene_finding", None)
    all_runs_json[str(trial.number)]["genomic_benchmark"].pop("human_ensembl_regulatory", None)
    with open(ROOT_DIR / "outputs/hpo/phase_1/outputs.json", "w") as f:
        json.dump(all_runs_json, f)

    print("OVERRIDES LIST IS")
    pprint(overrides)

    # Run the canonicalization
    cmd = [sys.executable, ROOT_DIR / "scripts/canonicalize_genomes.py", "experiment='canon/pretrain'", *overrides]
    run_cmd(cmd)

    # Run the gene finding and the human_ensembl_regulatory tasks
    cmd = [
        sys.executable, "-m", "src.utils.hpo_runner", 
        "--experiment", "canon/bend_benchmark",
        "--output-json", str(to_absolute(ROOT_DIR / "outputs/hpo/phase_1/outputs.json")),
        "--run-name", f"{str(trial.number)}.bend_benchmark.gene_finding"
        "dataset.dataset_name=gene_finding",
        "dataset.dataset_dir='{}'".format(to_absolute(ROOT_DIR / "data/bend_benchmark_parsed/")), 
        *overrides
    ]
    run_cmd(cmd)

    cmd = [
        sys.executable, "-m", "src.utils.hpo_runner", 
        "--experiment", "canon/genomic_benchmark",
        "--output-json", str(to_absolute(ROOT_DIR / "outputs/hpo/phase_1/outputs.json")),
        "--run-name", f"{str(trial.number)}.genomic_benchmark.human_ensembl_regulatory"
        "dataset.dataset_name=human_ensembl_regulatory",
        "dataset.dest_path='{}'".format(to_absolute(ROOT_DIR / "data/genomic_benchmark_parsed/")),
        *overrides
    ]
    run_cmd(cmd)