import atexit
import os
from .utils import read_fasta_into_records_dict, save_records_into_fasta, intervaltree_to_tuples, transform_sequence
from .parsed_genome import ParsedGenome, AlignmentError
import numpy as np
from pathlib import Path
from Bio.Seq import MutableSeq, Seq
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import h5py
import src.utils.train
import shutil
from tqdm.auto import tqdm
import multiprocessing as mp
from multiprocessing.pool import Pool

log = src.utils.train.get_logger(__name__)

def get_alignment_transcribed_segments(alignment, reference_genome: ParsedGenome):
    if alignment.strand == 1:
        segments_to_check = reference_genome.genome_transcribed_segments[alignment.ctg]["positive"]
    elif alignment.strand == -1:
        segments_to_check = reference_genome.genome_transcribed_segments[alignment.ctg]["negative"]
    else:
        raise RuntimeError(f"Unknown strand: {alignment.strand}")

    return intervaltree_to_tuples(segments_to_check.overlap(alignment.r_st, alignment.r_en))

@dataclass(frozen=True)
class DownstreamTask:
    task_name: str
    reference_genome_name: str
    folder_name: Optional[str] = None

class CanonicalizedDownstreamTask:
    tasks_list = None

    def __init__(self, config, task_config, hydra_logger=True):
        self.input_dir = Path(task_config.input_dir)
        assert self.input_dir.exists()
        self.output_dir = Path(task_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tasks lists should be a list of DownstreamTask instances
        assert self.tasks_list is not None
        self.has_downstream_task_been_processed = {t: False for t in self.tasks_list}
        self.unaligned_tasks_counter = {t.task_name: {"unaligned": 0, "aligned_without_annotations": 0, "total_tasks": 0} for t in self.tasks_list}
        self.task_config = task_config
        self.config = config
        self.hydra_logger = hydra_logger

        atexit.register(self._verify_all_tasks_were_canonicalized)
        atexit.register(self._log_unaligned_tasks)
        self._clean_output_dir()

    def _clean_output_dir(self):
        if self.output_dir.exists() and len(os.listdir(self.output_dir)) > 0:
            os.system(f"rm -r {str(self.output_dir)}/*")

    def canonicalize_available_tasks(self, reference_genome):
        for task in self.has_downstream_task_been_processed:
            if not self.has_downstream_task_been_processed[task] and task.reference_genome_name == reference_genome.name:
                self._log(f"  Processing {task.task_name}")
                self.process_task(task, reference_genome)
    
    def process_task(self, task, reference_genome: ParsedGenome) -> None:
        raise RuntimeError("unimplemented")

    def align_and_canonicalize_record(self, seq, task_name, reference_genome: ParsedGenome, update_counter=True):
        alignment, status = reference_genome.find_annotated_alignment(str(seq))
        self.unaligned_tasks_counter[task_name]["total_tasks"] += 1

        if update_counter:
            self._update_counter(task_name, status)

        if alignment is None:
            seq_canon = seq
            control_signal = np.zeros(len(seq))
        else:
            seq_canon, control_signal = self._canonicalize_record(seq, alignment, reference_genome)
        return seq_canon, control_signal, status

    def _update_counter(self, task_name, status):
        self.unaligned_tasks_counter[task_name]["total_tasks"] += 1
        if status == AlignmentError.PrimaryAlignmentNotFound:
            self.unaligned_tasks_counter[task_name]["unaligned"] += 1
        elif status == AlignmentError.AnnotatedAlignmentNotFound:
            self.unaligned_tasks_counter[task_name]["aligned_without_annotations"] += 1
        elif status is not None:
            raise RuntimeError(f"Unknown alignment error: {status}")

    def _canonicalize_record(self, seq, alignment, reference_genome: ParsedGenome):
        transcribed_segments_that_contain_seq = get_alignment_transcribed_segments(alignment, reference_genome)
        seq_canon = MutableSeq(seq)
        control_signal = np.zeros(len(seq))
        oriented_nucleotides_count = 0

        for segment in transcribed_segments_that_contain_seq:
            start, end = reference_genome.align_transcript_position_with_seq(alignment, segment)
            oriented_nucleotides_count += end - start
            if self.config.partial_downstream_canonicalization:
                transform_sequence(seq_canon, start, end, self.config.transformation)
                control_signal[start:end] = 1

        more_than_half_of_seq_oriented = oriented_nucleotides_count > len(seq) / 2
        if not self.config.partial_downstream_canonicalization and more_than_half_of_seq_oriented:
            transform_sequence(seq_canon, 0, len(seq), self.config.transformation)
            control_signal[0:len(seq)] = 1

        return seq_canon, control_signal

    def _verify_all_tasks_were_canonicalized(self):
        unprocessed_tasks = []
        for task in self.has_downstream_task_been_processed:
            if not self.has_downstream_task_been_processed[task]:
                unprocessed_tasks.append(task.task_name)
        
        if len(unprocessed_tasks) > 0:
            raise RuntimeError("The following tasks were not verified: {}".format('\n'.join(unprocessed_tasks)))

    def _log_unaligned_tasks(self):
        for task_name in self.unaligned_tasks_counter:
            if self.unaligned_tasks_counter[task_name]["total_tasks"] == 0:
                continue

            unaligned_percentage = round(self.unaligned_tasks_counter[task_name]["unaligned"] * 100 / self.unaligned_tasks_counter[task_name]["total_tasks"], 2)
            aligned_without_annotations_percentage = round(self.unaligned_tasks_counter[task_name]["aligned_without_annotations"] * 100 / self.unaligned_tasks_counter[task_name]["total_tasks"], 2)
            if unaligned_percentage > 0 or aligned_without_annotations_percentage > 0:
                log.warn(f"Task {task_name} has {unaligned_percentage}% unaligned records({self.unaligned_tasks_counter[task_name]['unaligned']}) and {aligned_without_annotations_percentage}% records({self.unaligned_tasks_counter[task_name]['aligned_without_annotations']}) that couldn't be processed")
    
    def _log(self, message):
        if self.hydra_logger:
            log.info(message)
        else:
            print(message)

class NTBenchmarkCanonicalizer(CanonicalizedDownstreamTask):
    HISTONE_TASKS = ("H3K4me3", "H3K27ac", "H3K27me3", "H3K4me1", "H3K36me3", "H3K9me3", "H3K9ac", "H3K4me2",
                     "H4K20me1", "H2AFZ")
    ENHANCERS_TASKS = ("enhancers", "enhancers_types")
    PROMOTER_TASKS = ("promoter_all", "promoter_tata", "promoter_no_tata")
    SPLICING_TASKS = ("splice_sites_all", "splice_sites_acceptors", "splice_sites_donors")
    tasks_list = [
        DownstreamTask(task_name=t, reference_genome_name="hg38") for t in 
        HISTONE_TASKS + ENHANCERS_TASKS + PROMOTER_TASKS + SPLICING_TASKS
    ]

    def process_task(self, task, reference_genome: ParsedGenome):
        assert reference_genome.name == "hg38"
        self._process_fasta_file(task.task_name, "train", reference_genome)
        self._process_fasta_file(task.task_name, "test", reference_genome)
        self.has_downstream_task_been_processed[task] = True

    def _process_fasta_file(self, task_name, split, reference_genome: ParsedGenome):
        fasta_path = self.input_dir / task_name / f"{task_name}_{split}.fasta"
        output_dir = self.output_dir / task_name
        output_path = output_dir / f"{task_name}_{split}.fasta"
        output_dir.mkdir(parents=True, exist_ok=True)

        records = read_fasta_into_records_dict(fasta_path)
        parsed_records = {}
        control_signals = {}
        for record_id, seq in records.items():
            parsed_id, parsed_seq, control_signal = self._process_record(task_name, record_id, seq, reference_genome)
            parsed_records[parsed_id] = parsed_seq
            control_signals[parsed_id] = control_signal

        save_records_into_fasta(parsed_records, output_path)
        np.savez(str(output_path).replace(".fasta", ".npz"), **control_signals)

    def _process_record(self, task_name, record_id, seq, reference_genome: ParsedGenome):
        if task_name in self.task_config.skip_tasks:
            return record_id, seq, np.zeros(len(seq)) # 0s for the control signal

        # There's no need for canonicalizing the labels because:
        # - On histone tasks, histones interact with double stranded DNA, the label is the same for the RC
        # - Enhancers (seem like) they're mostly bidirectional, and even if they aren't, I can't know if taking a non-enhancer sequence's RC suddenly turns it into an enhancer
        # - The promoters aren't transcribed
        # - The splicing seqs are already on the coding strand
        seq_canon, control_signal, _ = self.align_and_canonicalize_record(seq, task_name, reference_genome)
        return record_id, seq_canon, control_signal

class GenomicBenchmarkCanonicalizer(CanonicalizedDownstreamTask):
    tasks_list = (
        DownstreamTask(task_name="dummy_mouse_enhancers_ensembl", reference_genome_name="mouse"),
        DownstreamTask(task_name="demo_coding_vs_intergenomic_seqs", reference_genome_name="hg38"),
        DownstreamTask(task_name="demo_human_or_worm", reference_genome_name="hg38", folder_name="human"),
        DownstreamTask(task_name="demo_human_or_worm", reference_genome_name="roundworm", folder_name="worm"),
        DownstreamTask(task_name="drosophila_enhancers_stark", reference_genome_name="drosophila"),
        DownstreamTask(task_name="human_enhancers_cohn", reference_genome_name="hg38"),
        DownstreamTask(task_name="human_enhancers_ensembl", reference_genome_name="hg38"),
        DownstreamTask(task_name="human_ensembl_regulatory", reference_genome_name="hg38"),
        DownstreamTask(task_name="human_nontata_promoters", reference_genome_name="hg38"),
        DownstreamTask(task_name="human_ocr_ensembl", reference_genome_name="hg38"),
    )
    _worker_state_dict = {}

    def process_task(self, task, reference_genome: ParsedGenome):
        # Each task has the structure of split/label/seq.txt
        for split in os.listdir(self.input_dir / task.task_name):
            if task.folder_name is not None:
                self._process_seqs_dir(
                    task.task_name,
                    self.input_dir / task.task_name / split / task.folder_name,
                    self.output_dir / task.task_name / split / task.folder_name,
                    reference_genome
                )
            else:
                for label in os.listdir(self.input_dir / task.task_name / split):
                    self._process_seqs_dir(
                        task.task_name,
                        self.input_dir / task.task_name / split / label,
                        self.output_dir / task.task_name / split / label,
                        reference_genome
                    )
        self.has_downstream_task_been_processed[task] = True

    @staticmethod
    def _process_single_file(filename):
        state = GenomicBenchmarkCanonicalizer._worker_state_dict
        c = state["canonicalizer"]

        with open(state["seqs_dir_input"] / filename, "r") as f:
            seq = f.read().strip()

        if state["task_name"] in c.task_config.skip_tasks:
            seq_canon = seq
            control_signal = np.zeros(len(seq))
            status = None
        else:
            seq_canon, control_signal, status = c.align_and_canonicalize_record(seq, state["task_name"], state["reference_genome"], update_counter=False)

        with open(state["seqs_dir_output"] / filename, "w") as f:
            f.write(str(seq_canon))

        return control_signal, status

    @staticmethod
    def _init_worker(task_name, canonicalizer, reference_genome, seqs_dir_input, seqs_dir_output):
        GenomicBenchmarkCanonicalizer._worker_state_dict.update({
            "task_name": task_name,
            "canonicalizer": canonicalizer,
            "reference_genome": reference_genome,
            "seqs_dir_input": seqs_dir_input,
            "seqs_dir_output": seqs_dir_output
        })

    def _process_seqs_dir(self, task_name, seqs_dir_input, seqs_dir_output, reference_genome: ParsedGenome):
        seqs_dir_output.mkdir(parents=True, exist_ok=True)
        control_signals = {}
        input_files = os.listdir(seqs_dir_input)

        with Pool(
            context=mp.get_context("fork"),
            processes=self.config.num_workers,
            initializer=self._init_worker,
            initargs=(task_name, self, reference_genome, seqs_dir_input, seqs_dir_output)
        ) as pool:
            control_signals_iter = pool.imap(self._process_single_file, input_files, chunksize=64)
            for filename, (control_signal, status) in tqdm(zip(input_files, control_signals_iter), total=len(input_files)):
                control_signals[filename] = control_signal
                self._update_counter(task_name, status)
        
        np.savez(str(seqs_dir_output / "control_signals.npz"), **control_signals)

class BendBenchmarkCanonicalizer(CanonicalizedDownstreamTask):
    gene_finding_bed_columns = ("chromosome", "start", "end", "transcript_id", "strand", "flank_left", "flank_right", "length", "split")
    tasks_list = [DownstreamTask(task_name=t, reference_genome_name="hg38") for t in (
        "gene_finding", "cpg_methylation", "variant_effects_disease", "variant_effects_expression"
    )]
    
    def process_task(self, task, reference_genome: ParsedGenome) -> None:
        bed_name = task.task_name + ".bed"
        shutil.copy(self.input_dir / bed_name, self.output_dir / bed_name) # The bed file stays the same for all

        # The only thing that needs to be canonicalized are the gene_finding labels
        if task.task_name != "gene_finding":
            self.has_downstream_task_been_processed[task] = True
            return
        
        df = pd.read_csv(str(self.input_dir / f"{task.task_name}.bed"), sep="\t", header=None, names=self.gene_finding_bed_columns)
        df = df.iloc[1:].reset_index() # The first row has the column names, drop it
        df[["start", "end"]] = df[["start", "end"]].astype(int) # Convert the cols to int
        
        with h5py.File(self.input_dir / bed_name.replace(".bed", ".hdf5"), "r") as f:
            labels = f["labels"]
        
            # Make a mock alignment and get the control signals to know which labels to flip
            Alignment = namedtuple("Alignment", ["r_st", "r_en", "ctg", "strand"])
        
            canonicalized_labels = []
            for idx, label in enumerate(labels):
                df_row = df.iloc[idx]
                if df_row["strand"] == '+':
                    strand = 1
                elif df_row["strand"] == '-':
                    strand = -1
                else:
                    raise RuntimeError(f"Strand {df_row['strand']} not recognized")

                alignment = Alignment(df_row["start"], df_row["end"], df_row["chromosome"], strand)
                mock_seq = Seq("A" * (df_row["end"] - df_row["start"]))
                _, control_signal = self._canonicalize_record(mock_seq, alignment, reference_genome)
            
                # Flip the labels according to the control signal
                # The labels are {Ef, Df, If, Af, Er, Dr, Ir, Ar, NC} (exon, donor, acceptor, intron) and the strand they're on (A.1)
                mask = control_signal.astype(bool)
                mapping = np.array([4, 5, 6, 7, 0, 1, 2, 3]) # Ef -> Er, Df -> Dr...
                canonicalized_label = label.copy().astype(np.int8)
                canonicalized_label[mask] = mapping[canonicalized_label[mask]]
                canonicalized_labels.append(canonicalized_label)
            
        # Save the canonicalized labels
        with h5py.File(self.output_dir / bed_name.replace(".bed", ".hdf5"), "w") as f:
            # The np arrays have variable length, this is how they should be saved
            dt = h5py.vlen_dtype(np.dtype("int8"))
            ds = f.create_dataset("labels", shape=(len(canonicalized_labels,)), dtype=dt)
            for i, arr in enumerate(canonicalized_labels):
                ds[i] = arr

        self.has_downstream_task_been_processed[task] = True

# The SNPs occurs on both strands, so taking the RC should leave the label intact
class DeepseaCanonicalizer(CanonicalizedDownstreamTask):
    tasks_list = (DownstreamTask(task_name="deepsea", reference_genome_name="hg38"),)

    VOCAB = np.array(["A", "C", "G", "T", "N"]) # Taken from the definition of CharTokenizer in genomics.py
    CHAR_TO_IDX = {c: idx for idx, c in enumerate(VOCAB)}
    _worker_state_dict = {} # The permanent stuff for each worker

    def process_task(self, task, reference_genome: ParsedGenome) -> None:
        self._process_split("train", task.task_name, reference_genome)
        self._process_split("test", task.task_name, reference_genome)
        self._process_split("valid", task.task_name, reference_genome)
        self.has_downstream_task_been_processed[task] = True

    @staticmethod
    def _init_worker(task_name, canonicalizer, reference_genome, x, x_canonicalized, control_signals):
        # Is passed through CoW, so when writing it, it gets copied for itself 
        DeepseaCanonicalizer._worker_state_dict.update({
            "task_name": task_name,
            "canonicalizer": canonicalizer,
            "reference_genome": reference_genome,
            "x": x
        })
        x_canonicalized.flush()
        control_signals.flush()

    @staticmethod
    def _canonicalize_sequence(idx):
        state = DeepseaCanonicalizer._worker_state_dict
        task_name = state["task_name"]
        c = state["canonicalizer"]
        reference_genome = state["reference_genome"]
        
        # Canonicalize each sequence, they get automatically saved to disk
        seq = "".join(DeepseaCanonicalizer.VOCAB[state["x"][idx]])
        seq_canon, control_signal, status = c.align_and_canonicalize_record(seq, task_name, reference_genome, update_counter=False)
        seq_canon_arr = np.array([DeepseaCanonicalizer.CHAR_TO_IDX[c] for c in str(seq_canon)])
        return seq_canon_arr, control_signal, status


    def _process_split(self, split, task_name, reference_genome: ParsedGenome):
        # Load using mmap because they're massive
        x = np.load(self.input_dir / f"{split}-x.npy", mmap_mode="r")

        x_canonicalized = np.lib.format.open_memmap(str(self.output_dir / f"{split}-x.npy"), mode="w+", dtype=x.dtype, shape=x.shape)
        control_signals = np.lib.format.open_memmap(str(self.output_dir / f"{split}-control-signal.npy"), mode="w+", dtype=np.int8, shape=x.shape)

        with Pool(
            context=mp.get_context("fork"), # Fork doens't clone memory unless it's written into, uses CoW
            processes=self.config.num_workers, # Using all cores crashes my system
            initializer=DeepseaCanonicalizer._init_worker,
            initargs=(task_name, self, reference_genome, x, x_canonicalized, control_signals),
        ) as pool:
            # Sends 64 tasks to each work to reduce the overhead of data transfer
            results_iter = pool.imap(DeepseaCanonicalizer._canonicalize_sequence, range(x.shape[0]), chunksize=64)
            for idx, (seq_canon, control_signal, status) in tqdm(enumerate(results_iter), total=x.shape[0]):
                x_canonicalized[idx] = seq_canon
                control_signals[idx] = control_signal
                self._update_counter(task_name, status)
        
        # Copy the y arr
        shutil.copy(self.input_dir / f"{split}-y.npy", self.output_dir / f"{split}-y.npy")