import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os
from intervaltree import IntervalTree
import mappy as mp
import pyranges as pr

# Standard imports as the project is installed with pip -e .
import src.utils as utils
import src.utils.train
from src.utils.canonicalization import (
    read_fasta_into_records_dict,
    parse_sequence_report,
    rename_ncbi_annotations_to_ucsc,
    parse_transcribed_features_from_gff,
    canonicalize_genome,
    save_seqs_dict,
    generate_to_canonicalize_and_should_rc_at_end,
    canonicalize_nt_benchmark_fasta,
    save_canonicalization_control_signal,
    ParsedGenome
)

log = src.utils.train.get_logger(__name__)

# Register OmegaConf resolvers
OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

@hydra.main(config_path="../configs", config_name="config.yaml")
def main(config: DictConfig):    
    # Process config
    config = utils.train.process_config(config)
    c_cfg = config.canonicalization
    
    log.info("Loading training genome...")
    training_genome = read_fasta_into_records_dict(Path(c_cfg.reference_fasta_file))

    training_genome_to_canonicalize = {}
    def parse_hg38_segments(positive_coding_segments, negative_coding_segments, ucsc_id):
        if ucsc_id in training_genome: # Add it to training_genome_to_canonicalize if it's in the training_genome            
            segments_to_canonicalize, should_rc_at_end = generate_to_canonicalize_and_should_rc_at_end(
                positive_coding_segments, negative_coding_segments, c_cfg.orient_all_segments, c_cfg.partial_canonicalization_works
            )
            training_genome_to_canonicalize[ucsc_id] = {"segments": segments_to_canonicalize, "should_rc_at_end": should_rc_at_end}
    hg38 = ParsedGenome(c_cfg, c_cfg.hg38, parse_segments_using=parse_hg38_segments, genome_name="hg38")

    # Apply the canonicalization to the training genome
    transformation = c_cfg.transformation
    log.info(f"Canonicalizating genome with transformation: {transformation}...")
    canonicalized = canonicalize_genome(
        training_genome, 
        training_genome_to_canonicalize,
        transformation,
    )

    log.info("Saving results...")
    out_path = Path(c_cfg.reference_fasta_output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    save_path = out_path / "canonicalized.fasta"
    save_seqs_dict(canonicalized, save_path)
    log.info(f"Saved {transformation} to {save_path}")
    
    # Save the control signal
    log.info("Saving control signal...")
    control_signal_dir = out_path / "control_signals"
    control_signal_dir.mkdir(parents=True, exist_ok=True)
    save_canonicalization_control_signal(training_genome, training_genome_to_canonicalize, control_signal_dir)
    
    # Canonicalize the NT benchmark genomes
    log.info("Canonicalizing NT benchmark downstream tasks...")
    # Clean the parsed NT benchmark dir in case there's the old dataset
    os.system(f"rm -r {str(c_cfg.nt_benchmark_output_dir)}/*")
    
    # Process each dataset
    badly_aligned_tasks = []

    for task in os.listdir(c_cfg.nt_benchmark_dir):
        log.info(f"Processing {task}...")
        train_file_name = f"{task}_train.fasta"
        test_file_name = f"{task}_test.fasta"
        task_dir = Path(c_cfg.nt_benchmark_dir) / task
        output_task_dir = Path(c_cfg.nt_benchmark_output_dir) / task
        output_task_dir.mkdir(parents=True, exist_ok=True)
        
        was_badly_aligned = canonicalize_nt_benchmark_fasta(
            task_dir / train_file_name,
            aligner,
            gencode_transcribed_segments_intervaltree,
            transformation,
            c_cfg.partial_seq_canonicalization,
            c_cfg.orient_promoters,
            output_task_dir,
        )
        if was_badly_aligned:
            badly_aligned_tasks.append(train_file_name)
        
        canonicalize_nt_benchmark_fasta(
            task_dir / test_file_name,
            aligner,
            gencode_transcribed_segments_intervaltree,
            transformation,
            c_cfg.partial_seq_canonicalization,
            c_cfg.orient_promoters,
            output_task_dir,
        )

    if len(badly_aligned_tasks) > 0:
        print("Badly aligned tasks(worse than 50% off):")
        for t in badly_aligned_tasks:
            print(t)

if __name__ == "__main__":
    if "PROJECT_ROOT" not in os.environ:
        os.environ["PROJECT_ROOT"] = str(Path(__file__).parents[1].absolute())
    main()
