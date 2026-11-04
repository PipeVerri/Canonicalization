import hydra
from omegaconf import OmegaConf
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import MultilabelAUROC
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.train import process_config
from train import SequenceLightningModule

with open(PROJECT_ROOT / "data/deepsea/predictor_names.txt", "r") as f:
    predictor_names = [line.strip() for line in f.readlines()]

task_indexes = {
    "TF": [],
    "DHS": [],
    "HM": []
}

for idx, t in enumerate(predictor_names):
    task_name = t.split("|")[1]
    if "DNase" in task_name:
        task_indexes["DHS"].append(idx)
    elif any(task_name.startswith(h) for h in ("H1", "H2", "H3", "H4")):
        task_indexes["HM"].append(idx)
    else:
        task_indexes["TF"].append(idx)

# The results match the paper
#print("Task indexes:")
#for task, indexes in task_indexes.items():
#    print(f"  {task}: {len(indexes)}")

@hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
def main(config: OmegaConf):
    config = process_config(config)
    config.wandb.mode = "disabled"  # Disable wandb logging for evaluation
    config.dataset.batch_size = 32

    # ROOT CAUSE FIX: the experiment config sets pretrained_model_state_hook._name_=load_backbone.
    # That hook is designed for pretraining→finetuning transfer: it intentionally discards all
    # 'decoder.*' keys from the checkpoint and replaces them with freshly-initialised scratch
    # weights.  When loading a *fully-finetuned* checkpoint the decoder IS the trained
    # classification head — discarding it leaves a random linear layer → AUROC ≈ 0.5.
    # Setting _name_ to null disables the hook so the full state dict is loaded verbatim.
    config.train.pretrained_model_state_hook._name_ = None

    model = SequenceLightningModule.load_from_checkpoint(
        config.train.pretrained_model_path,
        config=config,
        strict=config.train.pretrained_model_strict_load,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model._initialize_state()  # Sets state to None; ConvNova's forward doesn't use recurrent state

    # set eval mode so dropout is disabled and batchnorm uses running stats.
    model.eval()

    all_preds = []   # accumulate sigmoid scores over the full dataset
    all_labels = []  # accumulate ground-truth labels over the full dataset

    val_loader = model.train_dataloader()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), desc="Evaluating"):
            x, y, *z = batch
            x = x.to(device)
            y = y.to(device)
            if len(z) == 0:
                z = {}

            x_out, y_out, w = model.forward((x, y, z))
            all_preds.append(torch.sigmoid(x_out).cpu().numpy())   # (B, 919)
            all_labels.append(y_out.cpu().numpy())                 # (B, 919)

    # Compute AUROC over the FULL dataset, not per batch.
    # Per-batch AUROC is unreliable for multi-label imbalanced data: many of the
    # 919 targets have only one label value in a small batch, so roc_auc_score
    # skips them → the reported score is noisy and meaningless.
    all_preds  = np.concatenate(all_preds,  axis=0)   # (N, 919)
    all_labels = np.concatenate(all_labels, axis=0)   # (N, 919)

    def macro_auroc(preds, labels, indices=None):
        """Macro-averaged AUROC over columns that have both label values present."""
        if indices is None:
            indices = range(preds.shape[1])
        aucs = [
            roc_auc_score(labels[:, i], preds[:, i])
            for i in indices
            if len(np.unique(labels[:, i])) > 1
        ]
        return float(np.mean(aucs)) if aucs else float("nan")

    overall = macro_auroc(all_preds, all_labels)
    scores = []
    for group, idxs in task_indexes.items():
        score = macro_auroc(all_preds, all_labels, idxs)
        print(f"  {group:>3s} AUROC ({len(idxs):3d} tasks): {score:.4f}")
        scores.append(score)
    mean_score = sum(scores) / len(scores)
    print(f"\nMean AUROC over task groups: {mean_score:.4f}")
    print(f"\nOverall macro AUROC (919 tasks): {overall:.4f}")

if __name__ == "__main__":
    main()