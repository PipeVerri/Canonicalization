import os
import sys
import time
import importlib
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

def run_model_with_config_and_overrides(experiment, overrides):
    config_dir = PROJECT_ROOT / "configs"
    prev_cwd = os.getcwd()

    # Fix for the pretrained_model_path
    for idx, val in enumerate(overrides):
        split = val.split("=")
        if split[0] == "train.pretrained_model_path":
            cktp_path = Path(prev_cwd) / split[1]
            overrides[idx] = f"train.pretrained_model_path={str(cktp_path.absolute())}"

    # Ensure unique run dir per call by adding timestamp if not explicitly overridden
    has_explicit_run_dir = any(o.startswith("hydra.run.dir=") for o in overrides)
    if not has_explicit_run_dir:
        timestamp = time.strftime("%Y-%m-%d/%H-%M-%S")
        overrides = [*overrides, f"hydra.run.dir=outputs/{experiment.replace('/', '_')}/{timestamp}"]

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
            if "train" in sys.modules:
                importlib.reload(sys.modules["train"])
            else:
                import train
            from train import process_config_and_run
            result = process_config_and_run(config)
        except ValueError as e:
            if "resolver" in str(e) and "already registered" in str(e):
                # Clear resolvers and retry
                OmegaConf.clear_resolver("eval")
                OmegaConf.clear_resolver("div_up")
                if "train" in sys.modules:
                    importlib.reload(sys.modules["train"])
                from train import process_config_and_run
                result = process_config_and_run(config)
            else:
                os.chdir(prev_cwd)
                raise e
        except Exception as e:
            os.chdir(prev_cwd)
            raise e

        os.chdir(prev_cwd)
        return result