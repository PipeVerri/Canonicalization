import json
import bisect
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import MutableSeq
from Bio.SeqRecord import SeqRecord
from BCBio import GFF
from intervaltree import IntervalTree
import pronto
import src.utils as utils
import src.utils.train

log = src.utils.train.get_logger(__name__)

#########################
# Genomic files parsing #
#########################

def read_fasta_into_records_dict(fasta_path):
    records = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        records[record.id] = record.seq
    return records

def parse_ncbi_assembly(jsonl_path):
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
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

def gff_records_generator(gff_path, ncbi_to_training_id):
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
        if current_start <= last_end:  
            merged_segments[-1] = (last_start, max(last_end, current_end))  
        else:
            merged_segments.append((current_start, current_end))  
    
    return merged_segments

###############
# OBO Parsing #
###############

def load_ontology(ontology_path):
    log.info(f"Loading Sequence Ontology from {ontology_path}")
    sequence_ontology = pronto.Ontology(ontology_path)

    sequence_name_to_obo_term = {
        term.name: term
        for term in sequence_ontology.terms()
        if term.name is not None and not term.obsolete
    }
    
    return sequence_name_to_obo_term

def is_feature_type_transcribed(name, sequence_name_to_obo_term, orient_whole_pseudogene=False, orient_cdjv_segments=True):
    ancestors = ["transcript","mRNA", "primary_transcript"] 
    if orient_whole_pseudogene:
        ancestors.append("pseudogene")
    if orient_cdjv_segments:
        ancestors.extend(("C_gene_segment", "D_gene_segment", "J_gene_segment", "V_gene_segment"))

    term = sequence_name_to_obo_term.get(name)
    if term is None:
        term = sequence_name_to_obo_term.get(name.replace("_", ""))
        if term is None:
            log.warning(f"{name} not found in Sequence Ontology")
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

def parse_record_into_segments_and_skew(record, training_genome, training_genome_id, 
                                        sequence_name_to_obo_term,
                                        orient_pseudogene_exons=False, 
                                        raise_error_if_encounter=(), 
                                        orient_whole_pseudogene=False, 
                                        orient_cdjv_segments=True):
    positive_segments = []
    negative_segments = []
    total_skew = 0

    for feature in record.features:
        for transcribed_feature in explore_recursively_transcribed_features(
            feature, 
            sequence_name_to_obo_term,
            orient_pseudogene_exons=orient_pseudogene_exons, 
            raise_error_if_encounter=raise_error_if_encounter,
            orient_whole_pseudogene=orient_whole_pseudogene,
            orient_cdjv_segments=orient_cdjv_segments
        ):
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

    positive_segments = merge_segment_list(positive_segments)
    negative_segments = merge_segment_list(negative_segments)

    return positive_segments, negative_segments, total_skew

def explore_recursively_transcribed_features(feature, sequence_name_to_obo_term, is_inside_pseudogene=False, 
                                             orient_pseudogene_exons=False, 
                                             raise_error_if_encounter=(),
                                             orient_whole_pseudogene=False,
                                             orient_cdjv_segments=True):
    if feature.type == "exon":
        if is_inside_pseudogene:
            if orient_pseudogene_exons:
                return [feature]
            else:
                return []
        else:
            raise ValueError("Encountered exon outside pseudogenome")

    if feature.type in raise_error_if_encounter:
        raise ValueError(f"Encountered {feature.type}")

    if feature.type == "pseudogene":
        is_inside_pseudogene = True

    if is_feature_type_transcribed(feature.type, sequence_name_to_obo_term, orient_whole_pseudogene=orient_whole_pseudogene, orient_cdjv_segments=orient_cdjv_segments):
        return [feature]
    else:
        to_return = []
        for sub_feature in feature.sub_features:
            to_return += explore_recursively_transcribed_features(
                sub_feature, 
                sequence_name_to_obo_term,
                is_inside_pseudogene=is_inside_pseudogene,
                orient_pseudogene_exons=orient_pseudogene_exons,
                raise_error_if_encounter=raise_error_if_encounter,
                orient_whole_pseudogene=orient_whole_pseudogene,
                orient_cdjv_segments=orient_cdjv_segments
            )
        return to_return

def should_canonicalize_segment(start, end, correctly_oriented_segments, overlap_check_whole_segment=True):
    overlaps = intervaltree_to_tuples(correctly_oriented_segments.overlap(start, end))

    if overlap_check_whole_segment:
        broken_nucleotides_length = sum(
            seg_end - seg_start
            for seg_start, seg_end in overlaps
        )
    else:
        broken_nucleotides_length = sum(
            min(end, seg_end) - max(start, seg_start)
            for seg_start, seg_end in overlaps
        )
    
    return broken_nucleotides_length < (end - start)

def canonicalize_genome(training_genome, training_genome_canonicalization_segments, transformations, overlap_check_whole_segment):
    results = {t: {} for t in transformations}

    for training_genome_id, segments in training_genome_canonicalization_segments.items():
        mut_seqs = {t: MutableSeq(training_genome[training_genome_id]) for t in transformations}
        
        for (start, end) in segments["to_orient"]:
            if should_canonicalize_segment(start, end, segments["correctly_oriented"], overlap_check_whole_segment):
                if "reverse" in mut_seqs:
                    reverse(mut_seqs["reverse"], start, end)
                if "complement" in mut_seqs:
                    complement(mut_seqs["complement"], start, end)
                if "rc" in mut_seqs:
                    reverse(mut_seqs["rc"], start, end)
                    complement(mut_seqs["rc"], start, end)

        for t in transformations:
            results[t][training_genome_id] = mut_seqs[t]

    return results


#################################
# NT Benchmark canonicalization #
#################################

def canonicalize_nt_benchmark_fasta(fasta_path, aligner, to_orient_intervaltree, transformation, save_path):
    records = read_fasta_into_records_dict(fasta_path)
    
    canonicalized_records = {}
    for id in records:
        # Align the record with the reference genome
        seq = str(records[id])
        best_alignment = next((h for h in aligner.map(seq) if h.is_primary), None)
        if best_alignment is None or best_alignment.mapq < 40:
            raise ValueError(f"Couldn't find good alignment for {id}")
    
        # Check if seq should be canonicalized or not
        alignment_start = best_alignment.r_st
        alignment_end = best_alignment.r_en
        overlapping_segments = intervaltree_to_tuples(to_orient_intervaltree.overlap(alignment_start, alignment_end))
        
        # Canonicalize seq using the segments
        for segment in overlapping_segments:
            # Index the segment relative to the seq's position
            segment_start = max(0, segment[0] - alignment_start)
            segment_end = min(len(seq), segment[1] - alignment_end)
            canonicalized_seq = MutableSeq(records[id])
            
            if transformation == "reverse":
                canonicalized_seq = reverse(canonicalized_seq, segment_start, segment_end)
            elif transformation == "complement":
                canonicalized_seq = complement(canonicalized_seq, segment_start, segment_end)
            elif transformation == "rc":
                canonicalized_seq = reverse(complement(canonicalized_seq, segment_start, segment_end), segment_start, segment_end)
            else:
                raise ValueError(f"{transformation} not recognized")
        
        raise ValueError("Y canonicalization needed")
        
        canonicalized_records[id] = canonicalized_seq
    
    # Save the canonicalized_record
    save_seqs_dict(canonicalized_records, save_path)