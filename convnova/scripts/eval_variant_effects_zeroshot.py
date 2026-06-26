#!/usr/bin/env python3
"""
Zero-shot variant effect scoring via cosine similarity of ref/alt embeddings.

Uses the same Hydra config as train.py.  Override dataset.dataset_name to
switch between variant_effects_disease and variant_effects_expression.

Example:
    python eval_variant_effects_zeroshot.py \
        experiment=canon/bend_benchmark \
        dataset.dataset_name=variant_effects_disease \
        +zeroshot.split=train \
        +zeroshot.batch_size=64 \
        +zeroshot.output_csv=results/variant_effects_disease.csv
"""

import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pyfaidx import Fasta
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from train import SequenceLightningModule
import hydra
import src.utils as utils
import src.utils.train
from src.utils import registry
import wandb

# ACGTN → 0-4, same as use_tokenizer=False in BendBenchmarkDataset
_CHAR_TO_ID: dict[str, int] = {
    "A": 0, "C": 1, "G": 2, "T": 3, "N": 4,
    "a": 0, "c": 1, "g": 2, "t": 3, "n": 4,
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def tokenize(seq: str, max_length: int) -> torch.LongTensor:
    """DNA string → left-padded LongTensor of token IDs (0=pad, 0-4=ACGTN)."""
    ids = [_CHAR_TO_ID.get(c, 4) for c in seq]
    if len(ids) > max_length:
        ids = ids[:max_length]
    elif len(ids) < max_length:
        ids = [0] * (max_length - len(ids)) + ids
    return torch.LongTensor(ids)


def load_variants(bed_path: str, dataset_name: str, split: str) -> pd.DataFrame:
    if dataset_name == "variant_effects_disease":
        columns = ("chromosome", "start", "end", "ref", "alt",
                   "split", "label", "full_label", "Consequence")
    else:  # variant_effects_expression
        columns = ("chromosome", "start", "end", "ref", "alt",
                   "split", "label", "Consequence", "actual")

    df = pd.read_csv(bed_path, sep="\t", header=None, names=columns)
    df = df.iloc[1:].reset_index(drop=True)
    df[["start", "end", "label"]] = df[["start", "end", "label"]].astype(int)

    split_key = "valid" if split == "val" else split
    return df[df["split"] == split_key].reset_index(drop=True)


def iter_sequences(df: pd.DataFrame, fasta: Fasta, max_length: int):
    """Yield (ref_ids, alt_ids, label) for each variant, skipping out-of-bounds rows."""
    half = max_length // 2
    skipped = 0

    for _, row in df.iterrows():
        snp_pos  = int(row["start"])
        seq_start = snp_pos - half
        seq_end   = snp_pos + (max_length - half)

        if seq_start < 0:
            skipped += 1
            continue

        try:
            seq_ref = fasta[row["chromosome"]][seq_start:seq_end].seq
        except Exception:
            skipped += 1
            continue

        if len(seq_ref) != (seq_end - seq_start):
            skipped += 1
            continue

        seq_alt = list(seq_ref)
        seq_alt[len(seq_alt) // 2] = row["alt"]
        seq_alt = "".join(seq_alt)

        yield tokenize(seq_ref, max_length), tokenize(seq_alt, max_length), int(row["label"])

    if skipped:
        print(f"  Skipped {skipped} variants (out of chromosome bounds)")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_cosine_similarities(
    backbone: torch.nn.Module,
    df: pd.DataFrame,
    fasta: Fasta,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    """
    Run inference over all variants and return a DataFrame with columns
    ['cosine_similarity', 'label'].
    """
    samples = list(iter_sequences(df, fasta, max_length))
    print(f"  {len(samples)} variants after filtering")

    all_scores: list[float] = []
    all_labels: list[int]   = []

    for i in tqdm(range(0, len(samples), batch_size), desc="Inference"):
        chunk = samples[i : i + batch_size]

        refs   = torch.stack([s[0] for s in chunk]).to(device)  # (B, L)
        alts   = torch.stack([s[1] for s in chunk]).to(device)  # (B, L)
        labels = [s[2] for s in chunk]

        emb_ref, _ = backbone(refs)   # (B, L, d_model)
        emb_alt, _ = backbone(alts)

        emb_ref = emb_ref.mean(dim=1)          # (B, d_model)
        emb_alt = emb_alt.mean(dim=1)

        cos_sim = F.cosine_similarity(emb_ref, emb_alt, dim=-1)  # (B,)

        all_scores.extend(cos_sim.cpu().tolist())
        all_labels.extend(labels)

    return pd.DataFrame({"cosine_similarity": all_scores, "label": all_labels})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="config.yaml", version_base=None)
def main(config: DictConfig) -> None:
    config = src.utils.train.process_config(config)

    bed_path = config.dataset.bed_path
    fasta_path = config.dataset.fasta_path
    dataset_name = config.dataset.dataset_name
    if not (dataset_name == "variant_effects_disease" or dataset_name == "variant_effects_expression"):
        raise ValueError("Wrong dataset name")
    
    max_length = config.dataset.max_length
    batch_size = 3000 # Harcoded bigger batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Model ---------------------------------------------------------------
    module = SequenceLightningModule.load_from_checkpoint(
        config.train.pretrained_model_path,
        config=config,
        strict=config.train.pretrained_model_strict_load,
    )
    backbone = module.model.to(device).eval()

    # --- Data ----------------------------------------------------------------
    df = load_variants(bed_path, dataset_name, "train")
    print(f"  {len(df)} rows in split")

    fasta = Fasta(fasta_path)

    # --- Inference -----------------------------------------------------------
    results = compute_cosine_similarities(backbone, df, fasta, max_length, batch_size, device)

    # --- W&B -----------------------------------------------------------------
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.get("entity", None),
        group=config.wandb.get("group", None),
        job_type=config.wandb.get("job_type", None),
        name=f"zeroshot-{dataset_name}",
        tags=["zeroshot", dataset_name],
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=False),
    )

    # --- AUROC ---------------------------------------------------------------
    n_pos = int(results["label"].sum())
    n_neg = len(results) - n_pos

    print(f"\nResults  [{dataset_name}]")
    print(f"  Samples : {len(results):,}  (pos={n_pos:,}, neg={n_neg:,})")
    print(f"  cos_sim  mean={results['cosine_similarity'].mean():.4f}  "
          f"std={results['cosine_similarity'].std():.4f}")

    if results["label"].nunique() < 2:
        print("  [WARNING] Only one class present — cannot compute AUROC")
        wandb.log({"auroc": float("nan"), "dataset": dataset_name})
    else:
        auroc = roc_auc_score(results["label"], results["cosine_similarity"])
        print(f"  AUROC   : {auroc:.4f}")
        wandb.log({
            f"zeroshot/{dataset_name}/auroc": auroc,
            "n_samples": len(results),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "cos_sim_mean": results["cosine_similarity"].mean(),
            "cos_sim_std": results["cosine_similarity"].std(),
        })

    wandb.finish()


if __name__ == "__main__":
    main()