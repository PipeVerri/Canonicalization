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
    build_ncbi_to_training_id,
    parse_transcribed_features_from_gff,
    canonicalize_genome,
    save_seqs_dict,
    canonicalize_nt_benchmark_fasta,
    intervaltree_to_tuples
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
    
    log.info("Reading training genome...")
    training_genome = read_fasta_into_records_dict(Path(c_cfg.reference_fasta_file))
    ncbi_to_training_id = build_ncbi_to_training_id(Path(c_cfg.reference_assembly_report), training_genome)
    #sequence_name_to_obo_term = load_ontology(c_cfg.ontology_path)

    log.info("Deciding canonicalization directions...")
    training_genome_canonicalization_segments = {}

    annotations_df = pr.read_gff3(c_cfg.gff_fasta_file).df
    # Drop the annotations that have na on any of the subsets
    nan_columns_to_drop = ["Feature", "Start", "End", "Strand", "Chromosome"]
    if not c_cfg.include_na_transcript_type:
        nan_columns_to_drop.append("transcript_type")
    annotations_df = annotations_df.dropna(subset=nan_columns_to_drop)

    for reference_id, training_genome_id in ncbi_to_training_id.items():
        positive_template_segments, negative_template_segments = parse_transcribed_features_from_gff(
            annotations_df,
            reference_id,
            annotations_level=c_cfg.annotations_level,
            include_cdjv_segments=c_cfg.include_cdjv_segments,
            include_pseudogenes=c_cfg.include_pseudogenes
        )

        # For each overlapping segment, calculate the cost of bringing stuff from - to + vs + to -
        # The iteration and intervaltree works with positive/negative or negative/positive. The choice of putting negative_template_segments is completely arbitrary
        negative_template_intervaltree = IntervalTree.from_tuples(negative_template_segments)
        keep_positive_strand_cost = 0
        keep_negative_strand_cost = 0

        for (start, end) in positive_template_segments:
            overlapping_templates = intervaltree_to_tuples(negative_template_intervaltree.overlap(start, end))
            if c_cfg.partial_canonicalization_works:
                pass # Keeping the positive strand or the negative strand increases the cost the same, because both have the same overlap length 
            else:
                keep_positive_strand_cost += sum(seg_end - seg_start for seg_start, seg_end in overlapping_templates)
                keep_negative_strand_cost += end - start
        
        # Decide which segments to canonicalize
        if keep_positive_strand_cost < keep_negative_strand_cost:
            training_genome_canonicalization_segments[training_genome_id] = positive_template_segments
        else:
            training_genome_canonicalization_segments[training_genome_id] = negative_template_segments

    # Apply the canonicalization to the training genome
    transformation = c_cfg.transformation
    log.info(f"Canonicalizating genome with transformation: {transformation}...")
    canonicalized = canonicalize_genome(
        training_genome, 
        training_genome_canonicalization_segments, 
        transformation,
    )

    log.info("Saving results...")
    out_path = Path(c_cfg.reference_fasta_output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    save_path = out_path / "canonicalized.fasta"
    save_seqs_dict(canonicalized, save_path)
    log.info(f"Saved {transformation} to {save_path}")
    
    # # Canonicalize the NT benchmark genomes
    # log.info("Canonicalizing NT benchmark downstream tasks...")
    # # Load the aligner index
    # log.info("Loading reference genome index...")
    # aligner = mp.Aligner(c_cfg.reference_genome_index)
    # # Convert to_orient into an intervaltree
    # training_genome_canonicalization_intervaltree = {i: IntervalTree.from_tuples(training_genome_canonicalization_segments[i]) for i in training_genome_canonicalization_segments}
    
    # # Process each dataset
    # for task in os.listdir(c_cfg.nt_benchmark_dir):
    #     log.info(f"Processing {task}...")
    #     train_file_name = f"{task}_train.fasta"
    #     test_file_name = f"{task}_test.fasta"
    #     task_dir = Path(c_cfg.nt_benchmark_dir) / task
    #     output_task_dir = Path(c_cfg.nt_benchmark_output_dir) / task
        
    #     canonicalize_nt_benchmark_fasta(
    #         task_dir / train_file_name, 
    #         aligner, 
    #         training_genome_canonicalization_intervaltree, 
    #         transformation, 
    #         output_task_dir / train_file_name,
    #         c_cfg.partial_seq_canonicalization
    #     )
        
    #     canonicalize_nt_benchmark_fasta(
    #         task_dir / test_file_name, 
    #         aligner, 
    #         training_genome_canonicalization_intervaltree, 
    #         transformation, 
    #         output_task_dir / test_file_name,
    #         c_cfg.partial_seq_canonicalization
    #     )

if __name__ == "__main__":
    if "PROJECT_ROOT" not in os.environ:
        os.environ["PROJECT_ROOT"] = str(Path(__file__).parents[1].absolute())
    main()
