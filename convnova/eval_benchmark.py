import argparse
import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

NT_BENCHMARK_TASKS = (
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
    "promoter_all",
    "promoter_tata",
    "promoter_no_tata",
    "splice_sites_all",
    "splice_sites_donors",
    "splice_sites_acceptors",
    "enhancers",
    "enhancers_types"
)

GENOMIC_BENCHMARK_TASKS = (
    "dummy_mouse_enhancers_ensembl", 
    "demo_coding_vs_intergenomic_seqs", 
    "demo_human_or_worm",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl"
)

BEND_BENCHMARK_TASKS = (
    "cpg_methylation",
    "gene_finding"
)

benchmark_tasks = {
    "nt_benchmark": NT_BENCHMARK_TASKS,
    "genomic_benchmark": GENOMIC_BENCHMARK_TASKS,
    "bend_benchmark": BEND_BENCHMARK_TASKS
}

def run_experiment(experiment, overrides, output_json=None):
    """Run a single experiment via the hpo_runner CLI."""
    cmd = [
        sys.executable, "-m", "src.utils.hpo_runner",
        experiment,
        *overrides,
    ]
    if output_json:
        cmd.extend(["--output-json", output_json])
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")
    
    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed with code {result.returncode}")
    return result.stdout

def eval_model_on_tasks(experiment, tasks, call_overrides, output_json=None):
    outputs = {}
    for task in tasks:
        overrides = [
            f"dataset.dataset_name={task}",
            *call_overrides
        ]
        run_experiment(experiment, overrides, output_json)
        outputs[task] = {"status": "completed"}
    return outputs

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True)
    parser.add_argument("-t", "--task", required=True)
    parser.add_argument("--output-json", help="Path to JSON file to append results")
    args, overrides = parser.parse_known_args()

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override arg {item} not recognized")

    if args.task in benchmark_tasks:
        task_list = benchmark_tasks[args.task]
    else:
        raise ValueError(f"{args.task} not recognized")

    eval_model_on_tasks(args.experiment, task_list, overrides, args.output_json)