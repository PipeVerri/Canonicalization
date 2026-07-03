import os
from datasets import load_dataset
from pathlib import Path
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded
import h5py
from scipy.io import loadmat
import numpy as np
import math
from tqdm import tqdm

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
        
def process_deepsea_data(deepsea_dir, chunk_size=10000):
    for f in ("train", "test", "valid"):
        print(f"  Processing {f}...")
        file_dir = deepsea_dir / f"{f}.mat"
        assert file_dir.exists()

        if f == "train":
            with h5py.File(file_dir) as data:
                total = data["trainxdata"].shape[-1]
                n_chunks = math.ceil(total / chunk_size)
                x_all = np.lib.format.open_memmap(deepsea_dir / f"{f}-x.npy", dtype=np.int8, mode="w+", shape=(total, 1000))
                y_all = np.lib.format.open_memmap(deepsea_dir / f"{f}-y.npy", dtype=np.int8, mode="w+", shape=(total, 919))

                for i in tqdm(range(n_chunks), desc=f"  {f}", unit="chunk"):
                    start, end = i * chunk_size, min((i + 1) * chunk_size, total)

                    x_chunk = data["trainxdata"][..., start:end].transpose(2, 0, 1) # (N, 1000, 4) 
                    x_chunk = np.argmax(x_chunk, axis=2) # (N, 1000)
                    # The dataset nucleotides are in the AGCT order, but the data is expected as ACGTN, so the indexes will be changed
                    g_nucleotides = x_chunk == 1
                    c_nucleotides = x_chunk == 2
                    x_chunk[g_nucleotides] = 2
                    x_chunk[c_nucleotides] = 1
                    x_all[start:end] = x_chunk

                    y_chunk = data["traindata"][..., start:end].T
                    y_all[start:end] = y_chunk
        else:
            data = loadmat(file_dir)
            x_raw = data[f"{f}xdata"].reshape(-1, 4, 1000)
            y = data[f"{f}data"].reshape(-1, 919)

            total = x_raw.shape[0]
            n_chunks = math.ceil(total / chunk_size)
            x_parts = []

            for i in tqdm(range(n_chunks), desc=f"  {f}", unit="chunk"):
                start, end = i * chunk_size, min((i + 1) * chunk_size, total)
                x_parts.append(np.argmax(x_raw[start:end], axis=1))

            x = np.concatenate(x_parts, axis=0)
            del x_raw

            np.save(deepsea_dir / f"{f}-x.npy", x)
            np.save(deepsea_dir / f"{f}-y.npy", y)

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

    print(f"Processing deepsea tasks...")
    process_deepsea_data(project_root / "data/deepsea/")

    print(f"Processing NT benchmark downstream tasks...")
    for task in tasks:
        export_to_fasta(task, output_base)
    
    print(f"Processing genomic benchmark downstream tasks...")
    tasks = (
        "dummy_mouse_enhancers_ensembl", 
        "demo_coding_vs_intergenomic_seqs", 
        "demo_human_or_worm",
        "drosophila_enhancers_stark",
        "human_enhancers_cohn",
        "human_enhancers_ensembl",
        "human_ensembl_regulatory",
        "human_nontata_promoters",
        "human_ocr_ensembl"
    )
    output_path = project_root / "data" / "genomic_benchmark"
    
    for t in tasks:
        download_genomic_benchmark_task(t, output_path)