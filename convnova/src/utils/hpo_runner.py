#!/usr/bin/env python
"""CLI wrapper to run training with config overrides and capture results."""
import os
import sys
import json
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_experiment(experiment, overrides, output_json=None):
    """Run a single experiment via subprocess and return results."""
    config_dir = PROJECT_ROOT / "configs"
    prev_cwd = os.getcwd()

    # Fix for the pretrained_model_path
    for idx, val in enumerate(overrides):
        split = val.split("=")
        if split[0] == "train.pretrained_model_path":
            cktp_path = Path(prev_cwd) / split[1]
            overrides[idx] = f"train.pretrained_model_path={str(cktp_path.absolute())}"

    # Ensure unique run dir per call
    has_explicit_run_dir = any(o.startswith("hydra.run.dir=") for o in overrides)
    if not has_explicit_run_dir:
        timestamp = time.strftime("%Y-%m-%d/%H-%M-%S")
        overrides = [*overrides, f"hydra.run.dir=outputs/{experiment.replace('/', '_')}/{timestamp}"]

    cmd = [
        sys.executable, "train.py",
        f"experiment={experiment}",
        *overrides,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")

    print(f"Running: {' '.join(cmd)}", flush=True)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            timeout=7200,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Training failed with code {result.returncode}")
        
        output = {"status": "success", "returncode": result.returncode}
        
    except subprocess.TimeoutExpired:
        output = {"status": "timeout", "error": "Training timed out"}
    except Exception as e:
        output = {"status": "error", "error": str(e)}

    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {}
        if output_path.exists():
            try:
                with open(output_path) as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        
        key = time.strftime("%Y%m%d_%H%M%S")
        data[key] = output
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run experiment with overrides")
    parser.add_argument("experiment", help="Experiment name (e.g. canon/nt_benchmark)")
    parser.add_argument("overrides", nargs="*", help="Config overrides as key=value")
    parser.add_argument("--output-json", help="Path to JSON file to append results")
    
    args = parser.parse_args()
    
    result = run_experiment(args.experiment, args.overrides, args.output_json)
    print(f"Result: {result}")