import argparse
from hydra import initialize_config_dir, compose
from pathlib import Path
from train import process_config_and_run
from hydra.core.utils import run_job
from hydra.core.global_hydra import GlobalHydra
from hydra.types import HydraContext
from hydra._internal.callbacks import Callbacks
from omegaconf import OmegaConf

# Resolve the project root from PROJECT_ROOT (set in train.py) so this script
# works regardless of the directory it is launched from.
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
    "gene_finding" # Excluded variant_effects task on purpose. Can't do attention on a CNN
)

def eval_model_on_task(task_name, experiment, overrides):
    if task_name == "nt-benchmark":
        tasks_list = NT_BENCHMARK_TASKS
    elif task_name == "genomic-benchmark":
        tasks_list = GENOMIC_BENCHMARK_TASKS
    elif task_name == "bend-benchmark":
        tasks_list = BEND_BENCHMARK_TASKS
    else:
        raise ValueError("Task not recognized")

    outputs = {}

    config_dir = (PROJECT_ROOT / "configs").absolute()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config_loader = GlobalHydra.instance().hydra.config_loader

        for task in tasks_list:
            config = compose(
                "config",
                overrides=[
                    f"experiment={experiment}",
                    *overrides,
                    f"dataset.dataset_name={task}"
                ],
                return_hydra_config=True, # run_job needs config["hydra"]["run"]["dir"]
            )

            if config.train.get("pretrained_model_path") is not None:
                p = Path(config.train.pretrained_model_path)
                if not p.is_absolute():
                    config.train.pretrained_model_path = str((PROJECT_ROOT / p).resolve())

            # Anchor hydra.run.dir to the project root with version_base=None so
            # it is independent of the launch directory, and force job.chdir=True
            # so run_job chdir's into the per-task output dir before executing
            # process_config_and_run (and restores the original cwd on exit).
            # This matches the behavior of `python train.py ...` invoked from the
            # project root, so config_tree.txt, checkpoints/, and the wandb
            # save_dir="." all land inside the per-task hydra run dir instead of
            # the launch directory.
            OmegaConf.set_struct(config, False)
            config.hydra.run.dir = str(
                PROJECT_ROOT / "outputs" / config.wandb.job_type
                / config.wandb.group
                / "${now:%Y-%m-%d}" / "${now:%H-%M-%S}-${dataset.dataset_name}"
            )
            config.hydra.job.chdir = True
            OmegaConf.set_struct(config, True)

            hydra_context = HydraContext(
                config_loader=config_loader,
                callbacks=Callbacks(),
            )

            job_return = run_job(
                task_function=process_config_and_run,
                config=config,
                job_dir_key="hydra.run.dir",
                job_subdir_key=None,
                hydra_context=hydra_context,
            )

            outputs[task] = job_return.return_value

    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True)
    parser.add_argument("-t", "--task", required=True)
    args, overrides = parser.parse_known_args()

    # Validate overrides
    for item in overrides:
        if not "=" in item:
            raise ValueError(f"Override arg {item} not recognized")

    eval_model_on_task(args.task, args.experiment, overrides)