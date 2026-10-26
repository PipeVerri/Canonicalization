# ConvNova Project Guide

## Purpose

This repository contains research code for ConvNova, a convolutional DNA sequence model, and extensions for genome canonicalization. The main workflows are:

- Pretraining on the human reference genome (HG38).
- Fine-tuning and evaluation on Nucleotide Transformer tasks, Genomic Benchmarks, BEND, and DeepSEA chromatin-profile prediction.
- Canonicalizing reference genomes and downstream datasets, including generating associated control signals.
- Running benchmark sweeps and hyperparameter-optimization experiments.

This is a research codebase, not a stable packaged API. Preserve experiment reproducibility and check config inheritance before changing behavior.

## Repository Layout

Run project commands from `convnova/`, the directory containing `train.py` and `configs/`.

- `convnova/train.py`: main Hydra/PyTorch Lightning training entrypoint.
- `convnova/eval_benchmark.py`: evaluates one experiment config across a named benchmark's tasks.
- `convnova/configs/`: Hydra configuration tree.
  - `experiment/`: runnable experiment families and their variants.
  - `pipeline/`: combines dataset, task, encoder, decoder, optimizer, and callbacks.
  - `dataset/`, `task/`, `model/`, `trainer/`, `optimizer/`, `scheduler/`: reusable config groups.
  - `canonicalization/`: genome and downstream canonicalization settings.
- `convnova/src/models/`: model implementations; ConvNova is under `ConvNova/`.
- `convnova/src/dataloaders/`: datamodules, dataset implementations, tokenizers, and collation.
- `convnova/src/tasks/`: task forwards, encoders, decoders, losses, and metrics.
- `convnova/src/utils/canonicalization/`: canonicalization logic and downstream-task processors.
- `convnova/src/hpo/` and `convnova/hpo/`: benchmark execution and HPO utilities.
- `convnova/scripts/`: data preparation, canonicalization, and specialized evaluation scripts.
- `convnova/tests/`: pytest suite.
- `convnova/data/` and `convnova/outputs/`: local datasets and generated runs; both are git-ignored.

The training entrypoint imports modules as `src.*`. Similarly named files also exist under `src/convnova/`; check actual import paths before editing rather than assuming both copies are active.

## Environment

The expected environment is Python 3.10 with PyTorch 2.1, PyTorch Lightning 1.8, Hydra/OmegaConf, Transformers, and the scientific/data dependencies in `requirements.txt`. Some workflows also require external bioinformatics tools such as `minimap2` and NCBI Datasets.

Use the root `README.md`, `environment.yml`, and `install_env.sh` as setup references, and inspect them for version differences before creating an environment. Install dependencies in an isolated environment. Flash Attention is only needed for model paths that use it; ConvNova itself is convolutional.

The training and canonicalization entrypoints set `PROJECT_ROOT` to the `convnova/` directory. Many config paths use this variable through `${oc.env:PROJECT_ROOT}`.

## Configuration Model

Hydra composition starts at `configs/config.yaml`. A typical experiment composes:

1. A pipeline and its reusable config groups.
2. `experiment/base` for shared model/trainer settings.
3. The selected experiment's overrides.
4. Command-line overrides.

The order matters. Inspect the selected experiment's `defaults` list and the fully composed config before assuming where a value comes from. Current experiment names should be taken from `configs/experiment/`; some examples in the README use older paths.

Components are commonly selected with `_name_` and instantiated through registries. Dataset and model attributes can also be injected into encoders and decoders. When changing a constructor or registry entry, trace its callers and config-provided arguments.

Datasets generally return `(x, y)` or `(x, y, extras)`. Task classes route extras through encoders, models, decoders, losses, and metrics. Changes to batch structure must be checked across that whole path. Language-model decoders may expose logits through a named tuple rather than a plain tensor.

## Common Commands

These commands are examples, not requests to launch long jobs automatically. Check dataset paths, checkpoint paths, GPU settings, and W&B configuration first.

```bash
cd convnova

# HG38 pretraining
python train.py experiment=baseline/pretrain

# Individual downstream tasks
python train.py experiment=baseline/nt_benchmark dataset.dataset_name=enhancers
python train.py experiment=baseline/genomic_benchmark dataset.dataset_name=human_ensembl_regulatory
python train.py experiment=baseline/bend_benchmark dataset.dataset_name=cpg_methylation
python train.py experiment=baseline/chromatin_profile_prediction

# Example Hydra overrides
python train.py experiment=baseline/pretrain trainer.devices=1 wandb=null

# Iterate over all tasks registered for a benchmark
python eval_benchmark.py -e baseline/nt_benchmark -t nt_benchmark
```

Canonicalized and raw-control variants live under the corresponding `experiment/paper/` and `experiment/hpo/` directories. They generally inherit baseline experiments, so review the complete defaults chain.

## Data and Canonicalization

Large datasets, genome indexes, checkpoints, and generated outputs are not stored in git. The `Makefile` contains download and indexing targets, while `scripts/prepare_downstream_data.py` prepares several downstream formats. `scripts/canonicalize_genomes.py` generates canonicalized genomes, downstream variants, and control signals using `configs/canonicalization/default.yaml`.

Data preparation can require substantial disk space, memory, network access, and external tools. Some scripts replace generated datasets. Inspect paths and destructive operations before running them, and do not run full preparation merely to validate a small code change.

Keep canonicalization transformations, sequence records, labels, and derived signals synchronized when modifying their shared logic.

## Testing and Validation

Run tests from `convnova/`:

```bash
python -m pytest -q
python -m pytest tests/test_downstream_tasks.py -q
```

For focused changes, run the relevant test module first and then the broader suite when practical. Add regression tests for behavioral changes, preferring small synthetic tensors or temporary fixture files over genome downloads, large datasets, GPUs, or external services.

If runtime dependencies are unavailable, syntax checks such as `python -m compileall` are useful but do not replace behavioral tests. Report which validation was and was not run.

Full training is not a routine validation step: it can require large local datasets, pretrained checkpoints, significant GPU time, and W&B access.

## Development Conventions

- Make focused changes that follow the surrounding Python and YAML style.
- Trace Hydra overrides and runtime imports before editing configuration or similarly named modules.
- Preserve dataset/model/decoder interfaces unless all consumers are updated.
- Use CPU-friendly, deterministic tests for data and model logic where possible.
- Do not commit generated data, run outputs, checkpoints, logs, credentials, or local environment files.
- Do not modify or delete unrelated user work.
- Update the README for user-facing workflow changes and keep this file current when project structure or agent workflows change.
