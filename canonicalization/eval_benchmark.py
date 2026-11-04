import argparse
from src.hpo.benchmarks import benchmark_tasks, eval_model_on_tasks

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True)
    parser.add_argument("-t", "--task", required=True)
    parser.add_argument("--output-json", help="Path to JSON file to append results")
    args, overrides = parser.parse_known_args()

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override arg {item} not recognized")

    if args.task in benchmark_tasks:
        task_list = benchmark_tasks[args.task]
    else:
        raise ValueError(f"{args.task} not recognized")

    eval_model_on_tasks(args.experiment, task_list, overrides, args.output_json)