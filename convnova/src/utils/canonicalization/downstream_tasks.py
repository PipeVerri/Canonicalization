import atexit
from ctypes import c_char
from curses import raw
import os
from re import split
from unittest import skip

from sympy import true

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
from src.utils.bend_benchmark_h5 import BEND_H5_FILENAME, read_bend_task, write_bend_h5
import shutil
from tqdm.auto import tqdm
import multiprocessing as mp
from multiprocessing.pool import Pool

log = src.utils.train.get_logger(__name__)

def get_alignment_transcribed_segments(alignment, reference_genome: ParsedGenome):
    # If the alignment is on the positive strand, I want to check if it's on top of a NEGATIVE transcript, and vice versa
    if alignment.strand == 1:
        segments_to_check = reference_genome.genome_transcribed_segments[alignment.ctg]["negative"]
    elif alignment.strand == -1:
        segments_to_check = reference_genome.genome_transcribed_segments[alignment.ctg]["positive"]
    else:
        raise RuntimeError(f"Unknown strand: {alignment.strand}")

    return intervaltree_to_tuples(segments_to_check.overlap(alignment.r_st, alignment.r_en))

def get_all_alignment_segments(alignment, reference_genome: ParsedGenome):
    get_strand_features = lambda strand: intervaltree_to_tuples(reference_genome.genome_transcribed_segments[alignment.ctg][strand].overlap(alignment.r_st, alignment.r_en))
    return {
        "positive": get_strand_features("positive"),
        "negative": get_strand_features("negative")
    }


@dataclass(frozen=True)
class DownstreamTask:
    task_name: str
    reference_genome_name: str
    max_length: int
    folder_name: Optional[str] = None

class CanonicalizedDownstreamTask:
    def __init__(self, config, task_config, hydra_logger=True):
        self.tasks_list = None
        self.input_dir = Path(task_config.input_dir)
        assert self.input_dir.exists()
        self.output_dir = Path(task_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.task_config = task_config
        self.config = config
        self.hydra_logger = hydra_logger
        self._create_tasks_list()
        # Tasks lists should be a list of DownstreamTask instances
        assert self.tasks_list is not None
        self.has_downstream_task_been_processed = {t: False for t in self.tasks_list}
        self.unaligned_tasks_counter = {t.task_name: {"unaligned": 0, "aligned_without_annotations": 0, "total_tasks": 0} for t in self.tasks_list}
        self._check()

        atexit.register(self._verify_all_tasks_were_canonicalized)
        atexit.register(self._log_unaligned_tasks)
        self._clean_output_dir()

    def _create_tasks_list(self) -> None:
        raise RuntimeError("unimplemented")

    def _check(self):
        for skipped in self.task_config.skip_tasks:
            if not any(task.task_name == skipped for task in self.tasks_list):
                raise ValueError(f"Skipped task {skipped} not present in task_list")

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

    def align_and_canonicalize_record(self, seq, task: DownstreamTask, reference_genome: ParsedGenome, update_counter=True):
        # Align the retained prefix so query coordinates refer to the same sequence
        # that is transformed, including on reverse-strand alignments.
        seq = seq[:task.max_length]
        alignment, status = reference_genome.find_annotated_alignment(str(seq))

        if update_counter:
            self._update_counter(task.task_name, status)

        if alignment is None:
            seq_canon = seq
            control_signal = raw_control_signal = np.zeros(len(seq))
        else:
            seq_canon, control_signal, raw_control_signal = self._canonicalize_record(seq, alignment, reference_genome, task.max_length)
        return seq_canon, control_signal, raw_control_signal, status

    def _update_counter(self, task_name, status):
        self.unaligned_tasks_counter[task_name]["total_tasks"] += 1
        if status == AlignmentError.PrimaryAlignmentNotFound:
            self.unaligned_tasks_counter[task_name]["unaligned"] += 1
        elif status == AlignmentError.AnnotatedAlignmentNotFound:
            self.unaligned_tasks_counter[task_name]["aligned_without_annotations"] += 1
        elif status is not None:
            raise RuntimeError(f"Unknown alignment error: {status}")

    def _canonicalize_record(self, seq, alignment, reference_genome: ParsedGenome, max_length: int):
        """Canonicalizes a sequence and returns the canonicalized sequence and control signals

        Args:
            seq (_type_): A Biopython Seq object
            alignment (_type_): An Alignment named tuple
            reference_genome (ParsedGenome): The reference genome used for canonicalization

        Returns:
            tuple: canonicalized_seq, control_signal, raw_control_signal
        """

        # The model should always view the transcribed features in the + orientattion. The RC-at-end doesn't change that, everything always ends up in the + orientation
        # If seq is on the positive branch, but it's on top of a negative transcribed feature, the model should see the RC of seq
        # If seq is on the negative branch, but it's on top of a positive transcribed feature, the model should see the RC of seq
        # If seq is on the negative branch and isn't on top of a transcribed feature, even though the model has never seen it, I won't take the RC to test generalization
        
        # Even if seq falls on a transcripted feature, the model may have never seen the + strand of seq because the canonicalization algorithm may had to take compromises and not orient the segment where seq is
        # Still, seq will be passed on it's + transcribed orientation because is that the model learns the transcriptional assimetry, not the specific compromises of my canonicalization algorithm

        # Start by trimming seq
        # If I reverse a downstream task sequence, and then it gets trimmed, the model may not be seeing the part of the sequence
        # that it would have seen originally.
        # That's why, when canonicalizing, I trim the sequences first so that doesn't happen
        seq = seq[:max_length]

        transcribed_segments_that_contain_seq = get_alignment_transcribed_segments(alignment, reference_genome)
        seq_canon = MutableSeq(seq)
        control_signal = np.zeros(len(seq))
        raw_control_signal = np.zeros(len(seq))
        oriented_nucleotides_count = 0

        # Canonicalize and create control signal
        for segment in transcribed_segments_that_contain_seq:
            start, end = reference_genome.align_transcript_position_with_seq(alignment, segment, len(seq))
            if end <= start:
                continue
            oriented_nucleotides_count += end - start
            if self.config.partial_downstream_canonicalization:
                transform_sequence(seq_canon, start, end, self.config.transformation)
                control_signal[start:end] = 1

        # Full or partial orientation
        more_than_half_of_seq_oriented = oriented_nucleotides_count > len(seq) / 2
        if not self.config.partial_downstream_canonicalization and more_than_half_of_seq_oriented:
            transform_sequence(seq_canon, 0, len(seq), self.config.transformation)
            control_signal[0:len(seq)] = 1

        # Code the raw control signal
        all_segments_that_overlap = get_all_alignment_segments(alignment, reference_genome)
        for segment in all_segments_that_overlap["positive"]:
            start, end = reference_genome.align_transcript_position_with_seq(alignment, segment, len(seq))
            if end <= start:
                continue
            raw_control_signal[start:end] = 1
        for segment in all_segments_that_overlap["negative"]:
            start, end = reference_genome.align_transcript_position_with_seq(alignment, segment, len(seq))
            if end <= start:
                continue
            raw_control_signal[start:end] = -1

        # If the alignment is on the - strand, then the control signal for it should be inverted
        if alignment.strand == -1:
            raw_control_signal = raw_control_signal * -1

        return seq_canon, control_signal, raw_control_signal

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

    def _create_tasks_list(self):
        # FASTA task names are plural; the dataset metadata uses singular keys.
        dataset_names = {
            "splice_sites_acceptors": "splice_sites_acceptor",
            "splice_sites_donors": "splice_sites_donor",
        }
        self.tasks_list = [
            DownstreamTask(task_name=t, reference_genome_name="hg38",
                           max_length=self.task_config.dataset[dataset_names.get(t, t)].max_length) for t in
            self.HISTONE_TASKS + self.ENHANCERS_TASKS + self.PROMOTER_TASKS + self.SPLICING_TASKS
        ]

    def process_task(self, task: DownstreamTask, reference_genome: ParsedGenome):
        assert reference_genome.name == "hg38"
        self._process_fasta_file(task, "train", reference_genome)
        self._process_fasta_file(task, "test", reference_genome)
        self.has_downstream_task_been_processed[task] = True

    def _process_fasta_file(self, task: DownstreamTask, split, reference_genome: ParsedGenome):
        fasta_path = self.input_dir / task.task_name / f"{task.task_name}_{split}.fasta"
        output_dir = self.output_dir / task.task_name
        output_path = output_dir / f"{task.task_name}_{split}.fasta"
        output_dir.mkdir(parents=True, exist_ok=True)

        records = read_fasta_into_records_dict(fasta_path)
        parsed_records = {}
        control_signals = {}
        raw_control_signals = {}
        for record_id, seq in records.items():
            parsed_id, parsed_seq, control_signal, raw_control_signal = self._process_record(task, record_id, seq, reference_genome)
            parsed_records[parsed_id] = parsed_seq
            control_signals[parsed_id] = control_signal
            raw_control_signals[parsed_id] = raw_control_signal

        save_records_into_fasta(parsed_records, output_path)
        np.savez(str(output_path).replace(".fasta", ".npz"), **control_signals)
        np.savez(output_dir / f"{task.task_name}_{split}_raw_signals.npz", **raw_control_signals)

    def _process_record(self, task: DownstreamTask, record_id, seq, reference_genome: ParsedGenome):
        # There's no need for canonicalizing the labels because:
        # - On histone tasks, histones interact with double stranded DNA, the label is the same for the RC
        # - Enhancers (seem like) they're mostly bidirectional, and even if they aren't, I can't know if taking a non-enhancer sequence's RC suddenly turns it into an enhancer
        # - The promoters aren't transcribed
        # - The splicing seqs are already on the coding strand

        seq = seq[:task.max_length]
        seq_canon, control_signal, raw_control_signal, _ = self.align_and_canonicalize_record(seq, task, reference_genome)

        # This is done after processing to get the raw_control_signal
        if task.task_name in self.task_config.skip_tasks:
            empty_control_signal = np.zeros(len(seq))
            return record_id, seq, empty_control_signal, raw_control_signal # 0s for the control signal

        return record_id, seq_canon, control_signal, raw_control_signal

class GenomicBenchmarkCanonicalizer(CanonicalizedDownstreamTask):
    _worker_state_dict = {}

    def _create_tasks_list(self) -> None:
        tasks_dict = [
            {"name": "dummy_mouse_enhancers_ensembl", "reference_genome_name": "mouse"},
            {"name": "demo_coding_vs_intergenomic_seqs", "reference_genome_name": "hg38"},
            {"name": "demo_human_or_worm", "reference_genome_name": "hg38", "folder_name": "human"},
            {"name": "demo_human_or_worm", "reference_genome_name": "roundworm", "folder_name": "worm"},
            {"name": "drosophila_enhancers_stark", "reference_genome_name": "drosophila"},
            {"name": "human_enhancers_cohn", "reference_genome_name": "hg38"},
            {"name": "human_enhancers_ensembl", "reference_genome_name": "hg38"},
            {"name": "human_ensembl_regulatory", "reference_genome_name": "hg38"},
            {"name": "human_nontata_promoters", "reference_genome_name": "hg38"},
            {"name": "human_ocr_ensembl", "reference_genome_name": "hg38"},
        ]

        self.tasks_list = [DownstreamTask(
            task_name=t["name"],
            reference_genome_name=t["reference_genome_name"],
            max_length=self.task_config.dataset[t["name"]].max_length,
            folder_name=t.get("folder_name", None)
        ) for t in tasks_dict]

    def process_task(self, task, reference_genome: ParsedGenome):
        # Each task has the structure of split/label/seq.txt
        for split in os.listdir(self.input_dir / task.task_name):
            if task.folder_name is not None:
                self._process_seqs_dir(
                    task,
                    self.input_dir / task.task_name / split / task.folder_name,
                    self.output_dir / task.task_name / split / task.folder_name,
                    reference_genome,
                    skip_task=False
                )
            else:
                for label in os.listdir(self.input_dir / task.task_name / split):
                    self._process_seqs_dir(
                        task,
                        self.input_dir / task.task_name / split / label,
                        self.output_dir / task.task_name / split / label,
                        reference_genome,
                        skip_task=task.task_name == "human_ensembl_regulatory" and label == "promoter"
                    )
        self.has_downstream_task_been_processed[task] = True

    @staticmethod
    def _process_single_file(args):
        filename, skip_task = args

        state = GenomicBenchmarkCanonicalizer._worker_state_dict
        c: GenomicBenchmarkCanonicalizer = state["canonicalizer"]

        with open(state["seqs_dir_input"] / filename, "r") as f:
            seq = f.read().strip()

        task = state["task"]
        seq = seq[:task.max_length]
        seq_canon, control_signal, raw_control_signal, status = c.align_and_canonicalize_record(seq, task, state["reference_genome"], update_counter=False)
        # Do it after to get the raw control signal
        if (task.task_name in c.task_config.skip_tasks) or skip_task:
            status = None
            seq_canon = seq
            control_signal = np.zeros(len(seq))

        with open(state["seqs_dir_output"] / filename, "w") as f:
            f.write(str(seq_canon))

        return control_signal, raw_control_signal, status

    @staticmethod
    def _init_worker(task: DownstreamTask, canonicalizer, reference_genome, seqs_dir_input, seqs_dir_output):
        GenomicBenchmarkCanonicalizer._worker_state_dict.update({
            "task": task,
            "canonicalizer": canonicalizer,
            "reference_genome": reference_genome,
            "seqs_dir_input": seqs_dir_input,
            "seqs_dir_output": seqs_dir_output
        })

    def _process_seqs_dir(self, task: DownstreamTask, seqs_dir_input, seqs_dir_output, reference_genome: ParsedGenome, skip_task):
        seqs_dir_output.mkdir(parents=True, exist_ok=True)
        control_signals = {}
        raw_control_signals = {}
        input_files = os.listdir(seqs_dir_input)

        with Pool(
            context=mp.get_context("fork"),
            processes=self.config.num_workers,
            initializer=self._init_worker,
            initargs=(task, self, reference_genome, seqs_dir_input, seqs_dir_output)
        ) as pool:
            control_signals_iter = pool.imap(self._process_single_file, zip(input_files, [skip_task] * len(input_files)), chunksize=64)
            for filename, (control_signal, raw_control_signal, status) in tqdm(zip(input_files, control_signals_iter), total=len(input_files)):
                control_signals[filename] = control_signal
                raw_control_signals[filename] = raw_control_signal
                self._update_counter(task.task_name, status)
        
        np.savez(str(seqs_dir_output / "control_signals.npz"), **control_signals)
        np.savez(str(seqs_dir_output / "raw_control_signals.npz"), **raw_control_signals)

class BendBenchmarkCanonicalizer(CanonicalizedDownstreamTask):
    gene_finding_bed_columns = ("chromosome", "start", "end", "transcript_id", "strand", "flank_left", "flank_right", "length", "split")

    def _create_tasks_list(self):
        self.tasks_list = [
            DownstreamTask(task_name=t, reference_genome_name="hg38",
                           max_length=self.task_config.dataset[t].max_length)
            for t in ("gene_finding", "cpg_methylation", "variant_effects_expression", "variant_effects_disease")
        ]

    @staticmethod
    def _reverse_label(label, control_signal):
        segments = []
        segment_start = None
        for i in range(len(control_signal)):
            if control_signal[i] == 1 and segment_start is None:
                segment_start = i
            elif control_signal[i] == 0 and segment_start is not None:
                segments.append((segment_start, i))
                segment_start = None
        if segment_start is not None:
            segments.append((segment_start, len(control_signal)))

        for (start, end) in segments:
            label[start:end] = label[start:end][::-1]

        return label

    @staticmethod
    def _complement_label(label, control_signal):
        # Flip the labels according to the control signal
        # The labels are {Ef, Df, If, Af, Er, Dr, Ir, Ar, NC} (exon, donor, acceptor, intron) and the strand they're on (A.1)
        mask = control_signal.astype(bool)
        mapping = np.array([4, 5, 6, 7, 0, 1, 2, 3, 8]) # Ef -> Er, Df -> Dr..., NC stays NC
        label[mask] = mapping[label[mask]]
        return label

    def process_task(self, task, reference_genome: ParsedGenome) -> None:
        if task.task_name in self.task_config.skip_tasks:
            self.has_downstream_task_been_processed[task] = True
            return

        original_df = read_bend_task(self.input_dir / BEND_H5_FILENAME, task.task_name)
        df = original_df.copy()
        Alignment = namedtuple("Alignment", ["r_st", "r_en", "ctg", "strand", "q_st", "q_en"]) # Mock alignment because I already have the genome coords
        # Watch out with doing a = b = []. They share the same pointer
        splits = ("train", "test", "valid")
        control_signals = {split: [] for split in splits}
        raw_control_signals = {split: [] for split in splits}

        for idx, row in df.iterrows():
            # The code for BEND explicitly ignores the strand field of the dataset 
            # (https://github.com/frederikkemarin/BEND/blob/main/conf/embedding/embed.yaml)

            seq = Seq(row["sequence"][:task.max_length])
            alignment = Alignment(row["start"], row["start"] + len(seq), row["chromosome"], 1, 0, len(seq)) # q_en in mappy is exclusive
            canonicalized_seq, control_signal, raw_control_signal = self._canonicalize_record(seq, alignment, reference_genome, task.max_length)
            df.at[idx, "sequence"] = str(canonicalized_seq)
            df.at[idx, "end"] = alignment.r_en
            if "length" in df.columns:
                df.at[idx, "length"] = len(seq)

            # For cpg_methylation the label stays as-is. For gene_finding, the label should be modified
            if task.task_name == "gene_finding":
                if len(row["label"]) != len(row["sequence"]):
                    raise ValueError("Gene-finding sequence and label lengths must match before cropping")
                canonicalized_label = row["label"][:task.max_length].copy().astype(np.int8)
                if self.config.transformation == "reverse":
                    canonicalized_label = self._reverse_label(canonicalized_label, control_signal)
                elif self.config.transformation == "complement":
                    canonicalized_label = self._complement_label(canonicalized_label, control_signal)
                elif self.config.transformation == "rc":
                    canonicalized_label = self._reverse_label(canonicalized_label, control_signal)
                    canonicalized_label = self._complement_label(canonicalized_label, control_signal)
                else:
                    raise ValueError(f"Transformation {self.config.transformation} not recognized")

                df.at[idx, "label"] = canonicalized_label

            control_signals[row["split"]].append(control_signal)
            raw_control_signals[row["split"]].append(raw_control_signal)

        # Store everything
        write_bend_h5(self.output_dir / BEND_H5_FILENAME, {task.task_name: df}, mode="a")
        control_signals_dir = self.output_dir / "control_signals"
        raw_control_signals_dir = self.output_dir / "raw_control_signals"
        control_signals_dir.mkdir(exist_ok=True)
        raw_control_signals_dir.mkdir(exist_ok=True)

        for split in splits:
            np.save(control_signals_dir / f"{task.task_name}_{split}", np.array(control_signals[split], dtype=object))
            np.save(raw_control_signals_dir / f"{task.task_name}_{split}", np.array(raw_control_signals[split], dtype=object))
        self.has_downstream_task_been_processed[task] = True

# The SNPs occurs on both strands, so taking the RC should leave the label intact
class DeepseaCanonicalizer(CanonicalizedDownstreamTask):

    def _create_tasks_list(self):
        self.tasks_list = [DownstreamTask(task_name="deepsea", reference_genome_name="hg38",
                                          max_length=self.task_config.dataset.max_length)]

    VOCAB = np.array(["A", "C", "G", "T", "N"]) # Taken from the definition of CharTokenizer in genomics.py
    CHAR_TO_IDX = {c: idx for idx, c in enumerate(VOCAB)}
    _worker_state_dict = {} # The permanent stuff for each worker

    def _check(self):
        pass # Skip tasks isn't declared

    def process_task(self, task, reference_genome: ParsedGenome) -> None:
        self._process_split("train", task, reference_genome)
        self._process_split("test", task, reference_genome)
        self._process_split("valid", task, reference_genome)
        self.has_downstream_task_been_processed[task] = True

    @staticmethod
    def _init_worker(task: DownstreamTask, canonicalizer, reference_genome, x, x_canonicalized, control_signals):
        # Is passed through CoW, so when writing it, it gets copied for itself 
        DeepseaCanonicalizer._worker_state_dict.update({
            "task": task,
            "canonicalizer": canonicalizer,
            "reference_genome": reference_genome,
            "x": x
        })
        x_canonicalized.flush()
        control_signals.flush()

    @staticmethod
    def _canonicalize_sequence(idx):
        state = DeepseaCanonicalizer._worker_state_dict
        task = state["task"]
        c = state["canonicalizer"]
        reference_genome = state["reference_genome"]
        
        # Canonicalize each sequence, they get automatically saved to disk
        seq = "".join(DeepseaCanonicalizer.VOCAB[state["x"][idx]])
        seq_canon, control_signal, raw_control_signal, status = c.align_and_canonicalize_record(seq, task, reference_genome, update_counter=False)
        seq_canon_arr = np.array([DeepseaCanonicalizer.CHAR_TO_IDX[c] for c in str(seq_canon)])
        return seq_canon_arr, control_signal, raw_control_signal, status


    def _process_split(self, split, task: DownstreamTask, reference_genome: ParsedGenome):
        # Load using mmap because they're massive
        x = np.load(self.input_dir / f"{split}-x.npy", mmap_mode="r")
        output_shape = (x.shape[0], min(x.shape[1], task.max_length))

        x_canonicalized = np.lib.format.open_memmap(str(self.output_dir / f"{split}-x.npy"), mode="w+", dtype=x.dtype, shape=output_shape)
        control_signals = np.lib.format.open_memmap(str(self.output_dir / f"{split}-control-signal.npy"), mode="w+", dtype=np.int8, shape=output_shape)
        raw_control_signals = np.lib.format.open_memmap(str(self.output_dir / f"{split}-raw-control-signal.npy"), mode="w+", dtype=np.int8, shape=output_shape)

        with Pool(
            context=mp.get_context("fork"), # Fork doens't clone memory unless it's written into, uses CoW
            processes=self.config.num_workers, # Using all cores crashes my system
            initializer=DeepseaCanonicalizer._init_worker,
            initargs=(task, self, reference_genome, x, x_canonicalized, control_signals),
        ) as pool:
            # Sends 64 tasks to each work to reduce the overhead of data transfer
            results_iter = pool.imap(DeepseaCanonicalizer._canonicalize_sequence, range(x.shape[0]), chunksize=64)
            for idx, (seq_canon, control_signal, raw_control_signal, status) in tqdm(enumerate(results_iter), total=x.shape[0]):
                x_canonicalized[idx] = seq_canon
                control_signals[idx] = control_signal
                raw_control_signals[idx] = raw_control_signal
                self._update_counter(task.task_name, status)
        
        # Copy the y arr, no modification to the labels
        shutil.copy(self.input_dir / f"{split}-y.npy", self.output_dir / f"{split}-y.npy")
