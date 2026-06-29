import atexit
import os
from .utils import read_fasta_into_records_dict, save_records_into_fasta, intervaltree_to_tuples, transform_sequence
from .parsed_genome import ParsedGenome, AlignmentError
import numpy as np
from pathlib import Path
from Bio.Seq import MutableSeq, Seq
from collections import namedtuple
import pandas as pd
import h5py

def get_alignment_transcribed_segments(alignment, reference_genome: ParsedGenome):
    if alignment.strand == 1:
        segments_to_check = reference_genome.genome_transcribed_segments[alignment.ctg]["positive"]
    elif alignment.strand == -1:
        segments_to_check = reference_genome.genome_transcribed_segments[alignment.ctg]["negative"]
    else:
        raise RuntimeError(f"Unknown strand: {alignment.strand}")

    return intervaltree_to_tuples(segments_to_check.overlap(alignment.r_st, alignment.r_en))

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
                self.process_task(task, reference_genome)
    
    def process_task(self, task, reference_genome: ParsedGenome) -> None:
        raise RuntimeError("unimplemented")

    def align_and_canonicalize_record(self, seq, task_name, reference_genome: ParsedGenome):
        alignment, status = reference_genome.find_annotated_alignment(str(seq))
        self.unaligned_tasks_counter[task_name]["total_tasks"] += 1

        if alignment is None:
            if status == AlignmentError.PrimaryAlignmentNotFound:
                self.unaligned_tasks_counter[task_name]["unaligned"] += 1
            elif status == AlignmentError.AnnotatedAlignmentNotFound:
                self.unaligned_tasks_counter[task_name]["aligned_without_annotations"] += 1
            else:
                raise RuntimeError(f"Unknown alignment error: {status}")
            return seq, np.zeros(len(seq))
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

        return seq_canon, control_signal

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

    def process_task(self, task, reference_genome: ParsedGenome):
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
        np.savez(str(output_path).replace(".fasta", ".npz"), **control_signals)

    def _process_record(self, task_name, record_id, seq, reference_genome: ParsedGenome):
        if task_name in self.task_config.skip_tasks:
            return record_id, seq, np.zeros(len(seq)) # 0s for the control signal

        # There's no need for canonicalizing the labels because:
        # - On histone tasks, histones interact with double stranded DNA, the label is the same for the RC
        # - Enhancers (seem like) they're mostly bidirectional, and even if they aren't, I can't know if taking a non-enhancer sequence's RC suddenly turns it into an enhancer
        # - The promoters aren't transcribed
        # - The splicing seqs are already on the coding strand
        seq_canon, control_signal = self.align_and_canonicalize_record(seq, task_name, reference_genome)
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

    def process_task(self, task, reference_genome: ParsedGenome):
        # Each task has the structure of split/label/seq.txt
        for split in os.listdir(self.input_dir / task["task_name"]):
            if task.get("folder_name") is not None:
                self._process_seqs_dir(
                    task["task_name"],
                    self.input_dir / task["task_name"] / split / task["folder_name"],
                    self.output_dir / task["task_name"] / split / task["folder_name"],
                    reference_genome
                )
            else:
                for label in os.listdir(self.input_dir / task["task_name"] / split):
                    self._process_seqs_dir(
                        task["task_name"],
                        self.input_dir / task["task_name"] / split / label,
                        self.output_dir / task["task_name"] / split / label,
                        reference_genome
                    )
        self.has_downstream_task_been_processed[task] = True

    def _process_seqs_dir(self, task_name, seqs_dir_input, seqs_dir_output, reference_genome: ParsedGenome):
        control_signals = {}
        for file in os.listdir(seqs_dir_input):
            with open(seqs_dir_input / file, "r") as f:
                seq = f.read().strip()
            
            if task_name in self.task_config.skip_tasks:
                seq_canon = seq
                control_signal = np.zeros(len(seq))
            else:
                seq_canon, control_signal = self.align_and_canonicalize_record(seq, task_name, reference_genome)
            
            with open(seqs_dir_output / file, "w") as f:
                f.write(str(seq_canon))
            control_signals[f] = control_signal
        
        np.savez(str(seqs_dir_output / "control_signals.npz"), **control_signals)

class BendBenchmark(CanonicalizedDownstreamTask):
    gene_finding_bed_columns = ("chromosome", "start", "end", "transcript_id", "strand", "flank_left", "flank_right", "length", "split")
    
    def __init__(self, config, task_config):
        super().__init__(config, task_config, [{"task_name": t, "reference_genome_name": "hg38"} for t in (
            "gene_finding", "cpg_methylation", "variant_effects_disease", "variant_effects_expression"
        )])
    
    def process_task(self, task, reference_genome: ParsedGenome) -> None:
        bed_name = task["task_name"] + ".bed"
        (self.input_dir / bed_name).copy(self.output_dir / bed_name) # The bed file stays the same for all

        # The only thing that needs to be canonicalized are the gene_finding labels
        if task["task_name"] != "gene_finding":
            self.has_downstream_task_been_processed[task] = True
            return
        
        df = pd.read_csv(str(self.input_dir / f"{task['task_name']}.bed"), sep="\t", header=None, names=self.gene_finding_bed_columns)
        df = df.iloc[1:].reset_index() # The first row has the column names, drop it
        df[("start, end")] = df[("start", "end")].astype(int) # Convert the cols to int
        
        with h5py.File(self.input_dir / bed_name.replace(".bed", ".h5"), "r") as f:
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
            canonicalized_label = label.copy()
            canonicalized_label[mask] = mapping[canonicalized_label[mask]]

            canonicalized_labels.append(canonicalized_label)
        
        # Save the canonicalized labels
        canonicalized_labels = np.stack(canonicalized_labels)
        with h5py.File(self.output_dir / bed_name.replace(".bed", ".h5")) as f:
            f.create_dataset("labels", data=canonicalized_labels)
        self.has_downstream_task_been_processed[task] = True

# The SNPs occurs on both strands, so taking the RC should leave the label intact
class ChromatinProfile(CanonicalizedDownstreamTask):
    def __init__(self, config, task_config):
        # Even though the dataset was built using hg19, I'll use the features from hg38 and map them to hg19
        super().__init__(config, task_config, (
            {"task_name": "deepsea", "reference_genome_name": "hg38"}
        ))
    
    def process_task(self, task, reference_genome: ParsedGenome) -> None:
        self._process_split("train", task["task_name"], reference_genome)
        self._process_split("test", task["task_name"], reference_genome)
        self._process_split("valid", task["task_name"], reference_genome)
        self.has_downstream_task_been_processed[task] = True
    
    def _process_split(self, split, task_name, reference_genome: ParsedGenome):
        # Load using mmap because they're massive
        x = np.load(self.input_dir / f"{split}-x.npy", mmap_mode="r")
        y = np.load(self.input_dir / f"{split}-y.npy", mmap_mode="r") # Only load it to copy it to the output folder 

        x_canonicalized = []
        control_signals = []
        vocab = np.array(["A", "C", "G", "T", "N"]) # Taken from the definition of CharTokenizer in genomics.py
        char_to_idx = {c: idx for idx, c in enumerate(vocab)}
        
        # Canonicalize each sequence
        for seq_arr in x:
            seq = "".join(vocab[seq_arr])
            seq_canon, control_signal = self.align_and_canonicalize_record(seq, task_name, reference_genome)
            seq_canon_arr = np.array([char_to_idx[c] for c in str(seq_canon)])
            x_canonicalized.append(seq_canon_arr)
            control_signals.append(control_signal)
        
        # Save them
        x_canonicalized = np.stack(x_canonicalized)
        control_signals = np.stack(control_signals)
        np.save(str(self.output_dir / f"{split}-x.npy"), x_canonicalized)
        np.save(str(self.output_dir / f"{split}-y.npy"), y)
        np.save(str(self.output_dir / f"{split}-control-signal.npy"), control_signals)