import random
from src.hpo.full_eval import run_all_tasks

RANDOM_SEEDS = [1938, 205, 8262, 1480, 4436]
#RANDOM_SEEDS = [8262, 1480, 4436]

if __name__ == "__main__":
    for seed in RANDOM_SEEDS:
        run_all_tasks(
            [f"train.seed={seed}"], 
            "pretraining/raw_control_signal/random_seeds",
            str(seed),
            "hpo/raw_control_signal"
        )
