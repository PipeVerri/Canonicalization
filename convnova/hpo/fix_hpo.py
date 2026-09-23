import optuna
import wandb
import sys
import os
from pathlib import Path
import subprocess
import json
from pprint import pprint

ROOT_DIR = Path(__file__).parents[1]

study = optuna.load_study(storage=f"sqlite:///{ROOT_DIR / 'outputs/hpo/fixed_hpo/optuna.db'}", study_name="hpo")
successful_trials = []

for trial in study.trials:
    if trial.state == optuna.trial.TrialState.COMPLETE:
        successful_trials.append(trial)

api = wandb.Api()
deepsea_wandb = api.runs("pipe-verri-gh/Canonicalization", filters={
    "group": "hpo",
    "jobType": "chromatin_profile_prediction"
})

pretrained_weights = [] # (absolute_pretrained_model_path)
for run in deepsea_wandb:
    pretrained_weights.append(run.config["train"]["pretrained_model_path"])

def to_absolute(path):
    return str(path.absolute())

def run_cmd(cmd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR) + ":" + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=ROOT_DIR, env=env, text=True)

for (trial, pretrained_path) in list(zip(successful_trials, pretrained_weights))[12:]:
    overrides = [
        f"train.pretrained_model_path='{pretrained_path}'",
        "wandb.group=hpo",
        f"+wandb.tags=['trial_{trial.number}']"
    ]
    for param, value in trial.params.items():
        if param == "overlapping_segments_canonicalization":
            param = "overlap_transformation_behavior"
        elif param == "orient_all_segments" or param == "keep_segments_order":
            param = "full_break_config." + param
        
        overrides.append(f"canonicalization.{param}={value}")

    # Delete the entries from the json so the hpo_runner.py script writes into it
    # Optuna trial scores will be updated later
    with open(ROOT_DIR / "outputs/hpo/fixed_hpo/outputs.json", "r") as f:
        all_runs_json = json.load(f)
    all_runs_json[str(trial.number)]["bend_benchmark"].pop("cpg_methylation", None)
    with open(ROOT_DIR / "outputs/hpo/fixed_hpo/outputs.json", "w") as f:
        json.dump(all_runs_json, f)

    print("OVERRIDES LIST IS")
    pprint(overrides)

    # Run the canonicalization
    cmd = [sys.executable, ROOT_DIR / "scripts/canonicalize_genomes.py", "experiment='hpo/canonicalized/pretrain'", *overrides]
    run_cmd(cmd)

    # Run both bend_benchmark tasks
    cmd = [
        sys.executable, "-m", "src.utils.hpo_runner", 
        "--experiment", "hpo/canonicalized/bend_benchmark",
        "--output-json", str(to_absolute(ROOT_DIR / "outputs/hpo/fixed_hpo/outputs.json")),
        "--run-name", f"{str(trial.number)}.bend_benchmark.gene_finding",
        "dataset.dataset_name=gene_finding",
        *overrides
    ]
    #run_cmd(cmd)

    cmd = [
            sys.executable, "-m", "src.utils.hpo_runner", 
            "--experiment", "hpo/canonicalized/bend_benchmark",
            "--output-json", str(to_absolute(ROOT_DIR / "outputs/hpo/fixed_hpo/outputs.json")),
            "--run-name", f"{str(trial.number)}.bend_benchmark.cpg_methylation",
            "dataset.dataset_name=cpg_methylation",
            *overrides
        ]
    run_cmd(cmd)
