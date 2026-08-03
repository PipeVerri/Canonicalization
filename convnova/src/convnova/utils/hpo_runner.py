import os
import sys
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))
from train import process_config_and_run

def run_model_with_config_and_overrides(experiment, overrides, output_json=None, run_name=None):
    config_dir = PROJECT_ROOT / "configs"
    prev_cwd = os.getcwd()

    # Fix for the pretrained_model_path
    for idx, val in enumerate(overrides):
        split = val.split("=")
        if split[0] == "train.pretrained_model_path":
            cktp_path = Path(prev_cwd) / split[1]
            overrides[idx] = f"train.pretrained_model_path={str(cktp_path.absolute())}"

    with initialize_config_dir(
        config_dir=str(config_dir.resolve()),
        version_base=None,
        job_name="eval_benchmark",
    ):
        config = compose(
            config_name="config",
            overrides=[
                f"experiment={experiment}",
                *overrides,
            ],
            return_hydra_config=True,
        )

        HydraConfig.instance().set_config(config)
        run_dir = config.hydra.run.dir
        run_dir = str((PROJECT_ROOT / run_dir).absolute())

        with open_dict(config):
            del config["hydra"]

        os.makedirs(run_dir, exist_ok=True)
        os.chdir(run_dir)

        try:
            # Reload train module fresh each call to reset wandb state
            result = process_config_and_run(config)
        except Exception as e:
            os.chdir(prev_cwd)
            raise e

        os.chdir(prev_cwd)
        if output_json is not None:
            if output_json.exists():
                with open(output_json, "r") as f:
                    data = json.load(f)
            else:
                data = {}

            to_modify = data
            keys = run_name.split(".")

            for key in keys[:-1]:
                if not key in to_modify:
                    to_modify[key] = {}
                to_modify = to_modify[key]
            to_modify[keys[-1]] = result

            with open(output_json, "w") as f:
                json.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True)
    parser.add_argument("-o", "--output-json")
    parser.add_argument("-n", "--run-name")

    args, overrides = parser.parse_known_args()

    # Validate the overrides
    for o in overrides:
        if not "=" in o:
            raise ValueError(f"Invalid override {o}")

    output_json = None
    if args.output_json is not None:
        assert args.run_name is not None
        output_json = Path(args.output_json).absolute()

    run_model_with_config_and_overrides(args.experiment, overrides, output_json=output_json, run_name=args.run_name)