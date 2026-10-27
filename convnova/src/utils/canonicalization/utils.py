from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import pronto
from pathlib import Path
import src.utils.train
import io

log = src.utils.train.get_logger(__name__)

#########################
# Genomic files parsing #
#########################

def read_fasta_into_records_dict(fasta_path: Path | str):
    records = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        records[record.id] = record.seq
    return records

def save_records_into_fasta(seqs_dict, save_path):
    records = [
        SeqRecord(seq, id=name, description="")
        for name, seq in seqs_dict.items()
    ]
    SeqIO.write(records, save_path, "fasta")

#########################
# Biopython Seq Parsing #
#########################

def reverse(seq, start, end):
    seq[start:end] = seq[start:end][::-1]

def complement(seq, start, end):
    seq[start:end] = seq[start:end].complement()

def transform_sequence(seq, start, end, transformation):
    if transformation == "reverse":
        reverse(seq, start, end)
    elif transformation == "complement":
        complement(seq, start, end)
    elif transformation == "rc":
        reverse(seq, start, end)
        complement(seq, start, end)
    else:
        raise ValueError(f"{transformation} not recognized")

########
# Misc #
########

def intervaltree_to_tuples(interval_tree):
    return [(interval.begin, interval.end) for interval in interval_tree]

def merge_segment_list(segments_list):
    if not segments_list:
        return []
    merged_segments = [segments_list[0]]

    for current_start, current_end in segments_list[1:]:
        last_start, last_end = merged_segments[-1]
        if current_start <= last_end:
            merged_segments[-1] = (last_start, max(last_end, current_end))
        else:
            merged_segments.append((current_start, current_end))

    return merged_segments