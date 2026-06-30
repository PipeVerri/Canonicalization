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
    generate_to_canonicalize_and_should_rc_at_end,
    save_records_into_fasta,
    canonicalize_genome,
    save_canonicalization_control_signal,
    ParsedGenome,
    NTBenchmarkCanonicalizer,
    GenomicBenchmarkCanonicalizer,
    BendBenchmarkCanonicalizer,
    DeepseaCanonicalizer,
    CanonicalizedDownstreamTask
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
    training_genome = read_fasta_into_records_dict(Path(c_cfg.pretraining_fasta_file))

    training_genome_to_canonicalize = {}
    def parse_hg38_segments(positive_coding_segments, negative_coding_segments, ucsc_id):
        if ucsc_id in training_genome: # Add it to training_genome_to_canonicalize if it's in the training_genome            
            segments_to_canonicalize, should_rc_at_end = generate_to_canonicalize_and_should_rc_at_end(
                positive_coding_segments, negative_coding_segments, c_cfg.orient_all_segments, c_cfg.partial_canonicalization_works
            )
            training_genome_to_canonicalize[ucsc_id] = {"segments": segments_to_canonicalize, "should_rc_at_end": should_rc_at_end}
    
    hg38 = ParsedGenome("hg38", c_cfg, c_cfg.hg38)

    # Apply the canonicalization to the training genome
    transformation = c_cfg.transformation
    log.info(f"Canonicalizating genome with transformation: {transformation}...")
    canonicalized = canonicalize_genome(
        training_genome, 
        training_genome_to_canonicalize,
        transformation,
    )

    log.info("Saving results...")
    out_path = Path(c_cfg.pretraining_fasta_output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    save_path = out_path / "canonicalized.fasta"
    save_records_into_fasta(canonicalized, save_path)
    log.info(f"Saved {transformation} to {save_path}")
    
    # Save the control signal
    log.info("Saving control signal...")
    control_signal_dir = out_path / "control_signals"
    control_signal_dir.mkdir(parents=True, exist_ok=True)
    save_canonicalization_control_signal(training_genome, training_genome_to_canonicalize, control_signal_dir)
    
    # Canonicalize the downstream tasks
    # The reference genomes are too large to load them all at once, so the tasks are canonicalized partially loading one reference genome at a time
    downstream_tasks: list[CanonicalizedDownstreamTask] = [
        NTBenchmarkCanonicalizer(c_cfg, c_cfg.nt_benchmark),
        GenomicBenchmarkCanonicalizer(c_cfg, c_cfg.genomic_benchmark),
        BendBenchmarkCanonicalizer(c_cfg, c_cfg.bend_benchmark),
        DeepseaCanonicalizer(c_cfg, c_cfg.deepsea)
    ]
    
    log.info("Canonicalizing downstream tasks using hg38...")
    for task in downstream_tasks:
        task.canonicalize_available_tasks(hg38)
    hg38.cleanup()
    
    log.info("Loading mouse reference genome...")
    mouse = ParsedGenome("mouse", c_cfg, c_cfg.mouse)
    log.info("Canonicalizing downstream tasks using mouse...")
    for task in downstream_tasks:
        task.canonicalize_available_tasks(mouse)
    mouse.cleanup()
    
    log.info("Loading drosophila reference genome...")
    drosophila = ParsedGenome("drosophila", c_cfg, c_cfg.drosophila)
    log.info("Canonicalizing downstream tasks using drosophila...")
    for task in downstream_tasks:
        task.canonicalize_available_tasks(drosophila)
    drosophila.cleanup()

    log.info("Loading worm reference genome...")
    worm = ParsedGenome("worm", c_cfg, c_cfg.worm)
    log.info("Canonicalizing downstream tasks using worm...")
    for task in downstream_tasks:
        task.canonicalize_available_tasks(worm)
    worm.cleanup()

if __name__ == "__main__":
    if "PROJECT_ROOT" not in os.environ:
        os.environ["PROJECT_ROOT"] = str(Path(__file__).parents[1].absolute())
    main()
