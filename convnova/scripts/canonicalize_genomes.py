from Bio import SeqIO
from Bio.Seq import Seq, MutableSeq
from Bio.SeqRecord import SeqRecord
from pathlib import Path
import json
from BCBio import GFF
import bisect
from intervaltree import IntervalTree
import pronto

##########
# Config #
##########

OVERLAP_CHECK_WHOLE_SEGMENT = True
TRANSCRIPTED_REGIONS = ("mRNA")

#########################
# Genomic files parsing #
#########################

# FASTA parsing

def read_fasta_into_records_dict(fasta_path):
    records = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        records[record.id] = record.seq
    return records

# NCBI assembly parsing

def parse_ncbi_assembly(jsonl_path):
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            # Avoid empty lines
            if line.strip():  
                data.append(json.loads(line))
    return data

def build_ncbi_to_training_id(assembly_path, training_genome):
    ncbi_to_training_id = {}
    ncbi_assembly = parse_ncbi_assembly(assembly_path)
    for assembly in ncbi_assembly:
        if assembly["ucscStyleName"] in training_genome:
            ncbi_to_training_id[assembly["refseqAccession"]] = assembly["ucscStyleName"]
    return ncbi_to_training_id

# GFF

def gff_records_generator(gff_path, ncbi_to_training_id):
    # Limits the GFF records to the ones present in the training genome
    limit_info = {
        "gff_id": list(ncbi_to_training_id.keys())
    }
    
    with open(gff_path, "r") as f:
        for rec in GFF.parse(f, limit_info=limit_info):
            yield rec


#########################
# Biopython Seq Parsing #
#########################

def reverse(seq, start, end):
    seq[start:end] = seq[start:end][::-1]

def complement(seq, start, end):
    seq[start:end] = seq[start:end].complement()

def save_seqs_dict(seqs_dict, save_path):
    records = [
        SeqRecord(seq, id=name, description="")
        for name, seq in seqs_dict.items()
    ]
    SeqIO.write(records, save_path, "fasta")

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
        if current_start <= last_end:  # Overlap or contiguous
            merged_segments[-1] = (last_start, max(last_end, current_end))  # Merge
        else:
            merged_segments.append((current_start, current_end))  # No overlap, add to list
    
    return merged_segments


###############
# OBO Parsing #
###############

sequence_ontology = pronto.Ontology(Path(__file__).parents[1] / "so.clean.obo")

# Maps name (like "transcript") to Term object
sequence_name_to_obo_term = {
    term.name: term
    for term in sequence_ontology.terms()
    if term.name is not None and not term.obsolete
}

def is_feature_type_transcribed(name):
    global sequence_name_to_obo_term
    ancestors = ("transcript","mRNA", "primary_transcript")

    term = sequence_name_to_obo_term.get(name)
    if term is None:
        term = sequence_name_to_obo_term.get(name.replace("_", ""))
        if term is None:
            print(f"Warning: {name} not found in Sequence Ontology")
            return False

    superclases = term.superclasses(with_self=True)

    for a in ancestors:
        a_term = sequence_name_to_obo_term.get(a)
        if a_term in superclases:
            return True

    return False

##############
# Main logic #
##############

def parse_record_into_segments_and_skew(record, training_genome, training_genome_id):
    positive_segments = []
    negative_segments = []
    total_skew = 0

    for feature in record.features:
        for transcribed_feature in explore_recursively_transcribed_features(feature):
            location = transcribed_feature.location
            if location.strand != 1 and location.strand != -1:
                continue

            feature_seq = training_genome[training_genome_id][location.start:location.end]
            
            if location.strand == 1:
                segments_list = positive_segments
                total_skew += feature_seq.count("G") + feature_seq.count("T") - feature_seq.count("C") - feature_seq.count("A")
            elif location.strand == -1:
                segments_list = negative_segments
                total_skew += feature_seq.count("C") + feature_seq.count("A") - feature_seq.count("G") - feature_seq.count("T")

            bisect.insort(segments_list, (location.start, location.end))

    # Merge adjacent segments to avoid having too many small segments that would be inefficient to flip
    positive_segments = merge_segment_list(positive_segments)
    negative_segments = merge_segment_list(negative_segments)

    return positive_segments, negative_segments, total_skew

def explore_recursively_transcribed_features(feature):
    if feature.type == "exon":
        raise ValueError("Shouldn't have encountered exon, should have encountered parent")
    
    if is_feature_type_transcribed(feature.type):
        return [feature]
    else:
        to_return = []
        for sub_feature in getattr(feature, "sub_features", []):
            print("Scanned sub-feature")
            to_return += explore_recursively_transcribed_features(sub_feature)
        return to_return

def should_canonicalize_segment(start, end, correctly_oriented_segments, overlap_check_whole_segment=OVERLAP_CHECK_WHOLE_SEGMENT):
    overlaps = intervaltree_to_tuples(correctly_oriented_segments.overlap(start, end))

    if overlap_check_whole_segment:
        # If a flip touches any part of a correctly oriented segment, you consider the whole segment broken.
        broken_nucleotides_length = sum(
            seg_end - seg_start
            for seg_start, seg_end in overlaps
        )
    else:
        # Only count the effectively overlapped part.
        broken_nucleotides_length = sum(
            min(end, seg_end) - max(start, seg_start)
            for seg_start, seg_end in overlaps
        )
    
    # Only flip the segment if the number of broken nucleotides is less than the number of non-broken nucleotides, to avoid breaking too many features
    return broken_nucleotides_length < (end - start)

def canonicalize_genome(training_genome, training_genome_canonicalization_segments):
    reversed_training_genome = {}
    complemented_training_genome = {}
    rc_training_genome = {}

    for training_genome_id, segments in training_genome_canonicalization_segments.items():
        reverse_seq = MutableSeq(training_genome[training_genome_id])
        complement_seq = MutableSeq(training_genome[training_genome_id])
        rc_seq = MutableSeq(training_genome[training_genome_id])
        
        for (start, end) in segments["to_orient"]:
            if should_canonicalize_segment(start, end, segments["correctly_oriented"]):
                reverse(reverse_seq, start, end)
                complement(complement_seq, start, end)
                # RC
                reverse(rc_seq, start, end)
                complement(rc_seq, start, end)

        reversed_training_genome[training_genome_id] = reverse_seq
        complemented_training_genome[training_genome_id] = complement_seq
        rc_training_genome[training_genome_id] = rc_seq

    return reversed_training_genome, complemented_training_genome, rc_training_genome

if __name__ == "__main__":
    data_dir = Path(__file__).parents[1] / "data/hg38/"
    ncbi_data_dir = data_dir / "ncbi_dataset/data/GCF_000001405.26/"

    print("Reading training genome...")
    training_genome = read_fasta_into_records_dict(data_dir / "hg38.ml.fa")
    ncbi_to_training_id = build_ncbi_to_training_id(ncbi_data_dir / "sequence_report.jsonl", training_genome)

    # Canonicalize the genomes to have them all on the template strand direction
    print("Deciding canonicalization directions...")
    training_genome_canonicalization_segments = {}    
    
    for record in gff_records_generator(ncbi_data_dir / "genomic.gff", ncbi_to_training_id):
        training_genome_id = ncbi_to_training_id[record.id]
        positive_segments, negative_segments, total_skew = parse_record_into_segments_and_skew(record, training_genome, training_genome_id)

        # On the template strand G + T < A + C (reference 1b). Canonicalize the genome so all the features are in the template direction
        if total_skew > 0: # G + T > A + C on the + strand, so the template strands are on the - side
            training_genome_canonicalization_segments[training_genome_id] = {
                "to_orient": positive_segments,
                "correctly_oriented": IntervalTree.from_tuples(negative_segments)
            }
        else: # Template strands are on the + side
            training_genome_canonicalization_segments[training_genome_id] = {
                "to_orient": negative_segments,
                "correctly_oriented": IntervalTree.from_tuples(positive_segments)
            }

    # Apply the canonicalization to the training genome
    print("Canonicalizating genome...")
    reversed_training_genome, complemented_training_genome, rc_training_genome = canonicalize_genome(training_genome, training_genome_canonicalization_segments)

    print("Saving...")
    out_path = data_dir / "../parsed/"
    output_prefix = "overlap_whole_" if OVERLAP_CHECK_WHOLE_SEGMENT else "overlap_partial_"
    
    save_seqs_dict(reversed_training_genome, out_path / (output_prefix + "reversed_canonicalization.fasta"))
    save_seqs_dict(complemented_training_genome, out_path / (output_prefix + "complemented_canonicalization.fasta"))
    save_seqs_dict(rc_training_genome, out_path / (output_prefix + "rc_canonicalization.fasta"))