import os
from datasets import load_dataset
from pathlib import Path
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded

dataset = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")

def write_split_into_fasta(output_file, filtered_dataset):
    assert len(filtered_dataset) != 0
    with open(output_file, "w") as f:
        for i, entry in enumerate(filtered_dataset):
            seq = entry['sequence']
            label = entry['label']
            
            # ConvNova's loader extracts the label from the LAST character of the header.
            # Format: >seq_{index}_{label}
            f.write(f">seq_{i}_{label}\n{seq}\n")

def export_to_fasta(task_name, output_dir):
    global dataset
    """
    Downloads a split from Hugging Face and filters by task_name,
    then saves it in the FASTA format expected by ConvNova.
    """
    print(f"Processing task: {task_name}...")
    
    # Create directory structure: data/nucleotide_transformer/{task_name}/
    task_dir = Path(output_dir) / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # Filter the dataset
    filtered_dataset = dataset.filter(lambda x: x["task"] == task_name)

    # Save the train and test splits of the filtered dataset
    write_split_into_fasta(task_dir / f"{task_name}_train.fasta", filtered_dataset["train"])
    write_split_into_fasta(task_dir / f"{task_name}_test.fasta", filtered_dataset["test"])

def download_genomic_benchmark_task(dataset_name, output_path):
    if not is_downloaded(dataset_name, cache_path=output_path):
        print("downloading {} to {}".format(dataset_name, output_path))
        download_dataset(dataset_name, version=0, dest_path=output_path)
    else:
        print("already downloaded {}".format(dataset_name))

if __name__ == "__main__":
    # Get the directory of the current script (convnova/scripts/)
    script_dir = Path(__file__).resolve().parent
    # Set the project root to the parent of the script dir (convnova/)
    project_root = script_dir.parent
    
    # List of tasks defined in convnova/configs/dataset/nucleotide_transformer.yaml
    tasks = set(dataset["train"]["task"])

    # Target directory is always convnova/data/nucleotide_transformer
    output_base = project_root / "data" / "nucleotide_transformer"
    os.system(f"rm -r {str(output_base)}/*")
    
    print(f"Base output directory: {output_base}")

    print(f"Processing NT benchmark downstream tasks...")
    for task in tasks:
        export_to_fasta(task, output_base)
    
    print(f"Processing genomic benchmark downstream tasks...")
    tasks = (
        "dummy_mouse_enhancers_ensembl", 
        "demo_coding_vs_intergenomic_seqs", 
        "demo_human_or_worm",
        "human_enhancers_cohn",
        "human_enhancers_ensembl",
        "human_ensembl_regulatory",
        "human_nontata_promoters",
        "human_ocr_ensembl"
    )
    output_path = project_root / "data" / "genomic_benchmark"
    
    for t in tasks:
        download_genomic_benchmark_task(t, output_path)