from src.hpo.full_eval import run_all_tasks
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

def to_absolute(path):
    return str(path.absolute())

overrides = ["train.seed=2222", "wandb.group=partial_downstream_canonicalization"]

#run_all_tasks(overrides, "pretraining/baseline/partial_downstream_canonicalization", "partial_downstream_canonicalization")
run_all_tasks(overrides, "partial_downstream_canonicalization", "partial_downstream_canonicalization", task_specific_overrides={
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
}, skip_tasks={"pretraining"}, checkpoint_path_override=to_absolute(ROOT_DIR / "outputs/hpo_checkpoints/20/checkpoints/test/loss.ckpt"))