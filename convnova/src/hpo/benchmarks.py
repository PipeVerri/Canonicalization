import subprocess
import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]

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

OLD_NT_BENCHMARK_TASKS = (
    "enhancers",
    "enhancers_types",
    "H3",
    "H3K14ac",
    "H3K36me3",
    "H3K4me1",
    "H3K4me2",
    "H3K4me3",
    "H3K79me3",
    "H3K9ac",
    "H4",
    "H4ac",
    "promoter_all",
    "promoter_no_tata",
    "promoter_tata",
    "splice_sites_acceptors",
    "splice_sites_all",
    "splice_sites_donors",
)

GENOMIC_BENCHMARK_TASKS = (
    "dummy_mouse_enhancers_ensembl", 
    "demo_coding_vs_intergenomic_seqs", 
    "demo_human_or_worm",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
    "drosophila_enhancers_stark"
)

BEND_BENCHMARK_TASKS = (
    "cpg_methylation",
    "gene_finding"
)

benchmark_tasks = {
    "nt_benchmark": NT_BENCHMARK_TASKS,
    "nt_benchmark_old": OLD_NT_BENCHMARK_TASKS,
    "genomic_benchmark": GENOMIC_BENCHMARK_TASKS,
    "bend_benchmark": BEND_BENCHMARK_TASKS
}

def run_experiment(experiment, overrides, output_json=None, run_name=None):
    global ROOT_DIR

    """Run a single experiment via the hpo_runner CLI."""
    cmd = [
        sys.executable, "-m", "src.utils.hpo_runner",
        "-e", experiment,
        *overrides,
    ]
    if output_json:
        assert run_name is not None
        cmd.extend(["--output-json", output_json, "--run-name", run_name])
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR) + ":" + env.get("PYTHONPATH", "")
    
    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT_DIR, env=env, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed with code {result.returncode}")

def eval_model_on_tasks(experiment, tasks, call_overrides, output_json=None, run_name=None):
    outputs = {}
    for task in tasks:
        overrides = [
            f"dataset.dataset_name={task}",
            *call_overrides
        ]

        run_experiment(experiment, overrides, output_json, f"{run_name}.{task}")
        outputs[task] = {"status": "completed"}
    return outputs