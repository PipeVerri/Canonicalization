import atexit
import os
from .utils import read_fasta_into_records_dict, save_records_into_fasta, intervaltree_to_tuples, transform_sequence
from .parsed_genome import ParsedGenome, AlignmentError
import numpy as np
from pathlib import Path
from Bio.Seq import MutableSeq
from collections import namedtuple

def get_alignment_transcribed_segments(alignment, reference_genome: ParsedGenome):
    if alignment.strand == 1:
        segments_to_check = reference_genome.genome_transcribed_segments[alignment.ctg]["positive"]
    elif alignment.strand == -1:
        segments_to_check = reference_genome.genome_transcribed_segments[alignment.ctg]["negative"]
    else:
        raise RuntimeError(f"Unknown strand: {alignment.strand}")

    return intervaltree_to_tuples(segments_to_check.overlap(alignment.r_st, alignment.r_en))

def save_control_signals(control_signals, output_path):
    np.savez(output_path, control_signals)

class CanonicalizedDownstreamTask:
    def __init__(self, config, task_config, tasks_list):
        self.input_dir = Path(task_config.input_dir)
        assert self.input_dir.exists()
        self.output_dir = Path(task_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tasks lists should be a list of {task_name: ..., reference_genome_name: ...}
        self.has_downstream_task_been_processed = {t: False for t in tasks_list}
        self.unaligned_tasks_counter = {t["task_name"]: {"unaligned": 0, "aligned_without_annotations": 0, "total_tasks": 0} for t in tasks_list}
        self.task_config = task_config
        self.config = config

        atexit.register(self._verify_all_tasks_were_canonicalized)
        atexit.register(self._log_unaligned_tasks)
        self._clean_output_dir()

    def _clean_output_dir(self):
        if self.output_dir.exists():
            os.system(f"rm -r {str(self.output_dir)}/*")

    def canonicalize_available_tasks(self, reference_genome):
        for task in self.has_downstream_task_been_processed:
            if not self.has_downstream_task_been_processed[task] and task["reference_genome_name"] == reference_genome.name:
                self._process_task(task, reference_genome)
    
    def _process_task(self, task, reference_genome: ParsedGenome):
        raise RuntimeError("unimplemented")

    def _align_and_canonicalize_record(self, seq, task_name, reference_genome: ParsedGenome):
        alignment, status = reference_genome.find_annotated_alignment(str(seq))
        self.unaligned_tasks_counter[task_name]["total_tasks"] += 1

        if alignment is None:
            if status == AlignmentError.PrimaryAlignmentNotFound:
                self.unaligned_tasks_counter[task_name]["unaligned"] += 1
            elif status == AlignmentError.AnnotatedAlignmentNotFound:
                self.unaligned_tasks_counter[task_name]["aligned_without_annotations"] += 1
            else:
                raise RuntimeError(f"Unknown alignment error: {status}")
            return seq, False, np.zeros(len(seq))
        else:
            return self._canonicalize_record(seq, alignment, reference_genome)

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

        return seq_canon, more_than_half_of_seq_oriented, control_signal

    def _verify_all_tasks_were_canonicalized(self):
        unprocessed_tasks = []
        for task in self.has_downstream_task_been_processed:
            if not self.has_downstream_task_been_processed[task]:
                unprocessed_tasks.append(task["task_name"])
        
        if len(unprocessed_tasks) > 0:
            raise RuntimeError("The following tasks were not verified: {}".format('\n'.join(unprocessed_tasks)))

    def _log_unaligned_tasks(self):
        for task_name in self.unaligned_tasks_counter:
            unaligned_percentage = round(self.unaligned_tasks_counter[task_name]["unaligned"] * 100 / self.unaligned_tasks_counter[task_name]["total_tasks"], 2)
            aligned_without_annotations_percentage = round(self.unaligned_tasks_counter[task_name]["aligned_without_annotations"] * 100 / self.unaligned_tasks_counter[task_name]["total_tasks"], 2)
            if unaligned_percentage > 0 or aligned_without_annotations_percentage > 0:
                print(f"Task {task_name} has {unaligned_percentage}% unaligned records({self.unaligned_tasks_counter[task_name]['unaligned']}) and {aligned_without_annotations_percentage}% records({self.unaligned_tasks_counter[task_name]['aligned_without_annotations']}) that couldn't be processed")

class NTBenchmark(CanonicalizedDownstreamTask):
    HISTONE_TASKS = ("H3K4me3", "H3K27ac", "H3K27me3", "H3K4me1", "H3K36me3", "H3K9me3", "H3K9ac", "H3K4me2",
                     "H4K20me1", "H2AFZ")
    ENHANCERS_TASKS = ("enhancers", "enhancers_types")
    PROMOTER_TASKS = ("promoter_all", "promoter_tata", "promoter_no_tata")
    SPLICING_TASKS = ("splice_sites_all", "splice_sites_acceptor", "splice_sites_donor")

    def __init__(self, config, task_config):
        super().__init__(config, task_config, {t: "hg38" for t in self.HISTONE_TASKS + self.ENHANCERS_TASKS + self.PROMOTER_TASKS + self.SPLICING_TASKS})

    def _process_task(self, task, reference_genome: ParsedGenome):
        assert reference_genome.name == "hg38"
        self._process_fasta_file(task["task_name"], "train", reference_genome)
        self._process_fasta_file(task["task_name"], "train", reference_genome)
        self.has_downstream_task_been_processed[task] = True

    def _process_fasta_file(self, task_name, split, reference_genome: ParsedGenome):
        fasta_path = self.input_dir / task_name / f"{task_name}_{split}.fasta"
        output_path = self.output_dir / task_name / f"{task_name}_{split}.fasta"

        records = read_fasta_into_records_dict(fasta_path)
        parsed_records = {}
        control_signals = {}
        for record_id, seq in records.items():
            parsed_id, parsed_seq, control_signal = self._process_record(task_name, record_id, seq, reference_genome)
            parsed_records[parsed_id] = parsed_seq
            control_signals[parsed_id] = control_signal

        save_records_into_fasta(parsed_records, output_path)
        save_control_signals(control_signals, str(output_path).replace(".fasta", ".npz"))

    def _process_record(self, task_name, record_id, seq, reference_genome: ParsedGenome):
        if task_name in self.PROMOTER_TASKS and not self.config.orient_promoters:
            return record_id, seq, np.zeros(len(seq)) # 0s for the control signal

        alignment, status = reference_genome.find_annotated_alignment(str(seq))

        # There's no need for canonicalizing the labels because:
        # - On histone tasks, histones interact with double stranded DNA, the label is the same for the RC
        # - Enhancers (seem like) they're mostly bidirectional, and even if they aren't, I can't know if taking a non-enhancer sequence's RC suddenly turns it into an enhancer
        # - The promoters aren't transcribed
        # - The splicing seqs are already on the coding strand
        seq_canon, _, control_signal = self._canonicalize_record(seq, alignment, reference_genome)
        return record_id, seq_canon, control_signal

class GenomicBenchmark(CanonicalizedDownstreamTask):
    PROMOTER_TASKS = ("human_nontata_promoters")

    def __init__(self, config, task_config):
        super().__init__(config, task_config, (
            {"task_name": "dummy_mouse_enhancers_ensembl", "reference_genome_name": "mouse"},
            {"task_name": "demo_coding_vs_intergenomic_seqs", "reference_genome_name": "hg38"},
            {"task_name": "demo_human_or_worm", "reference_genome_name": "hg38", "folder_name": "human"},
            {"task_name": "demo_human_or_worm", "reference_genome_name": "worm", "folder_name": "worm"},
            {"task_name": "drosophila_enhancers_stark", "reference_genome_name": "drosophila"},
            {"task_name": "human_enhancers_cohn", "reference_genome_name": "hg38"},
            {"task_name": "human_enhancers_ensembl", "reference_genome_name": "hg38"},
            {"task_name": "human_ensembl_regulatory", "reference_genome_name": "hg38"},
            {"task_name": "human_nontata_promoters", "reference_genome_name": "hg38"},
            {"task_name": "human_ocr_ensembl", "reference_genome_name": "hg38"},
        ))

    def _process_task(self, task, reference_genome: ParsedGenome):
        # Each task has the structure of split/label/seq.txt
        for split in os.listdir(self.input_dir / task["task_name"]):
            if task.get("folder_name") is not None:
                self._process_seqs_dir(
                    task["task_name"],
                    self.input_dir / task["task_name"] / split / task["folder_name"],
                    self.output_dir / task["task_name"] / split / task["folder_name"]
                )
            else:
                for label in os.listdir(self.input_dir / task["task_name"] / split):
                    self._process_seqs_dir(
                        task["task_name"],
                        self.input_dir / task["task_name"] / split / label,
                        self.output_dir / task["task_name"] / split / label
                    )
        self.has_downstream_task_been_processed[task] = True

    def _process_seqs_dir(self, task_name, seqs_dir_input, seqs_dir_output):
        for file in os.listdir(seqs_dir_input):
            with open(seqs_dir_input / file, "r") as f:
                seq = f.read().strip()
