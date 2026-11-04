# Strand Canonicalization for DNA Foundation Models

Research code for [Strand Canonicalization for DNA Foundation Models: A Proof-of-Concept](https://www.researchgate.net/publication/414832619_Strand_Canonicalization_for_DNA_Foundation_Models_A_Proof-of-Concept), by Felipe Verri.

DNA has two reverse-complementary strands, but transcription can introduce strand-specific asymmetries. This project investigates whether using genome annotations to orient sequences consistently helps a DNA foundation model exploit those asymmetries. The experiments use [ConvNova](https://github.com/aim-uofa/ConvNova), a convolutional model that processes both strands without enforcing complete reverse-complement invariance.

The study compares canonicalized sequences with a standard baseline and a **leakage control** that receives annotation-derived strand information as an explicit input. On the tasks selected for full-scale evaluation, the tested canonicalization approach did not outperform the leakage control. The control improved over the baseline on some tasks, but this does not establish that canonicalization or transcriptional asymmetry caused the improvement. See the paper for the full results and limitations.

## Repository

- `canonicalization/` — training, evaluation, data preparation, and canonicalization code.
- `canonicalization/configs/experiment/` — baseline, paper, and hyperparameter-optimization experiment configurations.
- `canonicalization/configs/canonicalization/default.yaml` — canonicalization options and data paths.
- `canonicalization/tests/` — tests for canonicalization, data loading, and model utilities.
- `paper/` — paper source and figures.

## Setup

Run commands from `canonicalization/`. The supplied Conda environment uses Python 3.10, PyTorch 2.1, and PyTorch Lightning 1.8:

```bash
cd canonicalization
conda env create -f environment.yml
conda activate canonicalization
```

If an environment named `canonicalization` already exists, pass `-n <new-name>` to `conda env create` and activate that name instead.

The environment also installs `minimap2` and NCBI Datasets for data preparation. GPU training requires a compatible PyTorch/CUDA setup; Flash Attention is only needed for model paths that use it, not ConvNova itself.

### Optional: Flash Attention

Building Flash Attention from source can take a long time, so `install_env.sh` recommends using a prebuilt wheel instead:

1. Check the CUDA version used by PyTorch:

   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```

2. Download a compatible wheel from the [Flash Attention prebuilt wheels page](https://mjunya.com/flash-attention-prebuild-wheels/). Match your Python version (3.10), PyTorch version (2.1), PyTorch CUDA version, and platform.
3. Install the downloaded wheel in the activated environment:

   ```bash
   pip install /path/to/flash_attention_wheel.whl
   ```

Your GPU and NVIDIA driver must also support the selected build.

## Data and canonicalization

Genomes, benchmark datasets, indexes, checkpoints, and outputs are not included in the repository. The `Makefile` has targets for downloading and indexing several inputs; `scripts/prepare_downstream_data.py` prepares downstream datasets. The Makefile includes a local `.env` file; create one if needed and review the targets before running them.

Configure paths and options in `configs/canonicalization/default.yaml`. Once the required reference genomes, annotations, indexes, and downstream data are in place, the following generates canonicalized data and control signals:

```bash
python -m scripts.canonicalize_genomes +canonicalization=default
```

Data preparation and canonicalization can require substantial disk space and memory and may replace generated files. Review the scripts and output paths first.

## Training and evaluation

Experiments are configured with Hydra. For example:

```bash
# Pretrain a baseline on HG38
python train.py experiment=baseline/pretrain trainer.devices=1 wandb.mode=disabled

# Fine-tune on a Nucleotide Transformer task using a pretrained checkpoint
python train.py experiment=baseline/nt_benchmark dataset.dataset_name=enhancers \
  train.pretrained_model_path=/path/to/checkpoint.ckpt wandb.mode=disabled

# Evaluate across the Nucleotide Transformer benchmark
python eval_benchmark.py -e baseline/nt_benchmark -t nt_benchmark \
  train.pretrained_model_path=/path/to/checkpoint.ckpt wandb.mode=disabled
```

The repository also supports Genomic Benchmarks, BEND, and DeepSEA chromatin-profile prediction. Paper configurations are under `configs/experiment/paper/`, including `canonicalized/` and `raw_control_signal/` variants; HPO variants are under `configs/experiment/hpo/`. Check config inheritance, data paths, matching pretrained checkpoints, GPU settings, and logging options before running an experiment. The examples above are starting points, not a complete paper reproduction recipe.

## Tests

```bash
python -m pytest -q
```

## Citation and acknowledgements

If you use this work, please cite [the paper](https://www.researchgate.net/publication/414832619_Strand_Canonicalization_for_DNA_Foundation_Models_A_Proof-of-Concept). This project builds on [ConvNova](https://github.com/aim-uofa/ConvNova) and uses several public genomic datasets and benchmarks.
