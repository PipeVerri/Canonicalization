import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os
from intervaltree import IntervalTree
import mappy as mp

# Standard imports as the project is installed with pip -e .
import src.utils as utils
import src.utils.train
from src.utils.canonicalization import (
    read_fasta_into_records_dict,
    build_ncbi_to_training_id,
    load_ontology,
    gff_records_generator,
    parse_record_into_segments_and_skew,
    canonicalize_genome,
    save_seqs_dict,
    canonicalize_nt_benchmark_fasta
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

    sequence_name_to_obo_term = load_ontology(c_cfg.ontology_path)

    log.info("Deciding canonicalization directions...")
    training_genome_canonicalization_segments = {}    
    
    for record in gff_records_generator(Path(c_cfg.reference_gff_file), ncbi_to_training_id):
        training_genome_id = ncbi_to_training_id[record.id]
        positive_segments, negative_segments, total_skew = parse_record_into_segments_and_skew(
            record, 
            training_genome, 
            training_genome_id,
            sequence_name_to_obo_term,
            orient_pseudogene_exons=c_cfg.orient_pseudogene_exons,
            raise_error_if_encounter=c_cfg.raise_error_if_encounter,
            orient_whole_pseudogene=c_cfg.orient_whole_pseudogene,
            orient_cdjv_segments=c_cfg.orient_cdjv_segments
        )

        if total_skew > 0: 
            training_genome_canonicalization_segments[training_genome_id] = {
                "to_orient": positive_segments,
                "correctly_oriented": IntervalTree.from_tuples(negative_segments)
            }
        else: 
            training_genome_canonicalization_segments[training_genome_id] = {
                "to_orient": negative_segments,
                "correctly_oriented": IntervalTree.from_tuples(positive_segments)
            }

    # Apply the canonicalization to the training genome
    transformation = c_cfg.transformation
    log.info(f"Canonicalizating genome with transformation: {transformation}...")
    results = canonicalize_genome(
        training_genome, 
        training_genome_canonicalization_segments, 
        [transformation],
        c_cfg.overlap_check_whole_segment
    )

    log.info("Saving results...")
    out_path = Path(c_cfg.reference_fasta_output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
        
    seqs = results[transformation]
    save_path = out_path / "canonicalized.fasta"
    save_seqs_dict(seqs, save_path)
    log.info(f"Saved {transformation} to {save_path}")
    
    # Canonicalize the NT benchmark genomes
    log.info("Canonicalizing NT benchmark downstream tasks...")
    # Load the aligner index
    log.info("Loading reference genome index...")
    aligner = mp.Aligner(c_cfg.reference_genome_index)
    # Convert to_orient into an intervaltree
    to_orient_intervaltree = IntervalTree.from_tuples(training_genome_canonicalization_segments["to_orient"])
    
    # Process each dataset
    for task in os.listdir(c_cfg.nt_benchmark_dir):
        log.info(f"Processing {task}...")
        train_file_name = f"{task}_train.fasta"
        test_file_name = f"{task}_test.fasta"
        task_dir = Path(c_cfg.nt_benchmark_dir) / task
        output_task_dir = Path(c_cfg.nt_benchmark_output_dir) / task
        
        canonicalize_nt_benchmark_fasta(
            task_dir / train_file_name, 
            aligner, 
            to_orient_intervaltree, 
            transformation, 
            output_task_dir / train_file_name
        )
        
        canonicalize_nt_benchmark_fasta(
            task_dir / test_file_name, 
            aligner, 
            to_orient_intervaltree, 
            transformation, 
            output_task_dir / test_file_name
        )

if __name__ == "__main__":
    if "PROJECT_ROOT" not in os.environ:
        os.environ["PROJECT_ROOT"] = str(Path(__file__).parents[1].absolute())
    main()
