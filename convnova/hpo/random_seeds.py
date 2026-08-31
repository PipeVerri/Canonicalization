import random
from src.hpo.full_eval import run_all_tasks

RANDOM_SEEDS = 3

if __name__ == "__main__":
    for _ in range(RANDOM_SEEDS):
        seed = random.randint(0, 10_000)
        while seed == 2222 or seed == 2805 or seed == 7275 or seed == 5947:
            seed = random.randint(0, 10_000)

        run_all_tasks(
            [f"train.seed={seed}"], 
            "pretraining/baseline/random_seeds",
            str(seed),
        )
