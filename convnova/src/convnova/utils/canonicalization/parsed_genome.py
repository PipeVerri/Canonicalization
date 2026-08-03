import json
import src.utils.train
import pyranges as pr
import bisect
import mappy as mp
from intervaltree import IntervalTree
from enum import Enum
import gc
from pathlib import Path
import io
import pronto
from .utils import merge_segment_list

log = src.utils.train.get_logger(__name__)

def parse_sequence_report(jsonl_path):
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def find_primary_hit(hits):
    for h in hits:
        if h.is_primary:
            return h
    return None

def rename_ncbi_annotations_to_ucsc(annotations, sequence_report):
    # Parse NCBI's GFF record to use gencode's chromosome/scaffolds names
    # For assembled chromosomes, the gencode name is the ucsc id, for the rest, is the genbank accession

    to_rename = {}
    for report in sequence_report:
        if report["role"] == "assembled-molecule":
            to_rename[report["refseqAccession"]] = report["ucscStyleName"]
        else:
            to_rename[report["refseqAccession"]] = report["genbankAccession"]

    for name, new_name in to_rename.items():
        annotations["Chromosome"] = annotations["Chromosome"].replace(name, new_name)

def load_and_clean_obo_file(ontology_path):
    output = ""
    with Path(ontology_path).open(encoding="utf-8") as f:
        for line in f:
            # Remove the xrefs with backlash
            if (line.startswith("xref:") and "\\" in line) or (line.startswith("def:")):
                continue
            output += line + "\n"
    return io.BytesIO(output.encode("utf-8"))

def load_ontology(ontology_path):
    sequence_ontology = pronto.Ontology(load_and_clean_obo_file(ontology_path))

    sequence_name_to_obo_term = {
        term.name: term
        for term in sequence_ontology.terms()
        if term.name is not None and not term.obsolete
    }
    
    return sequence_name_to_obo_term

class ParsedGenome:
    """Loads the genome index and parses it's .gff file to take the positive and negative segments"""

    def __init__(self, genome_name, canonicalization_config, genome_config, verbose=True, parse_segments_using=None, hydra_logger=True) -> None:
        self.canonicalization_config = canonicalization_config
        self.genome_config = genome_config
        self.verbose = verbose
        self.hydra_logger = hydra_logger
        self.name = genome_name

        if verbose:
            self._log(f"Parsing {genome_name}")
        # Load the index
        if verbose:
            self._log("  Loading genome index...")
        self.aligner = mp.Aligner(genome_config.index_file)
        # Load the GFF
        if verbose:
            self._log("  Loading GFF file...")
        # Load the SO for parsing NCBI's genomes, it's light so there's no need to check if it should load
        self.ontology = load_ontology(canonicalization_config.sequence_ontology_file)

        annotations_df = pr.read_gff3(genome_config.gff_file).df
        # Rename it to match the GENCODE chromosome names
        if genome_config.rename_gff_using_seq_report:
            seq_report = parse_sequence_report(genome_config.sequence_report_file)
            rename_ncbi_annotations_to_ucsc(annotations_df, seq_report)

        annotations_df = self._clean_annotations_df(annotations_df) # Drop the rows that have nan on the important columns
        # Filter the df to only contain transcribed features
        if genome_config.is_gff_from_ncbi:
            annotations_df = self._filter_ncbi_transcribed_features(annotations_df)
        else:
            annotations_df = self._filter_gencode_transcribed_features(annotations_df)

        self.genome_transcribed_segments = self._parse_annotations_df_into_segments(annotations_df, parse_segments_using)

    def _filter_ncbi_transcribed_features(self, df):
        feature_mask = {f: self._is_ontology_feature_transcribed(f) for f in df["Feature"].unique()}
        df = df[df["Feature"].map(feature_mask) == True]
        return df

    def _is_ontology_feature_transcribed(self, name):
        ancestors = ["transcript","mRNA", "primary_transcript"] 
        if self.canonicalization_config.include_pseudogenes:
            ancestors.append("pseudogene")
        if self.canonicalization_config.include_cdjv_segments:
            ancestors.extend(("C_gene_segment", "D_gene_segment", "J_gene_segment", "V_gene_segment"))

        term = self.ontology.get(name)
        if term is None:
            term = self.ontology.get(name.replace("_", ""))
            if term is None:
                log.warning(f"{name} not found in Sequence Ontology")
                return False

        superclases = term.superclasses(with_self=True)

        for a in ancestors:
            a_term = self.ontology.get(a)
            if a_term in superclases:
                return True

        return False

    def _filter_gencode_transcribed_features(self, df):
        annotations_level = self.canonicalization_config.annotations_level
        assert type(annotations_level) == int and annotations_level <= 3 and annotations_level >= 1

        df = df[df["level"].astype(int) <= annotations_level]  # Filter the annotations for the desired level
        df = df[df["Feature"] == "transcript"]  # Filter all the non-transcript features

        # ~ inverts the index, that's why the na is inverted
        # If the user wants to exclude na transcript_types, it's taken care of when dropping the na columns in the beggining
        if not self.canonicalization_config.include_pseudogenes:
            df = df[~df["transcript_type"].str.contains("pseudogene", na=False)]
        if not self.canonicalization_config.include_cdjv_segments:
            df = df[~df["transcript_type"].str.contains("IG_C", na=False)]
            df = df[~df["transcript_type"].str.contains("IG_D", na=False)]
            df = df[~df["transcript_type"].str.contains("IG_J", na=False)]
            df = df[~df["transcript_type"].str.contains("IG_V", na=False)]
        
        return df

    def _clean_annotations_df(self, annotations_df):
        nan_columns_to_drop = ["Feature", "Start", "End", "Strand", "Chromosome"]
        if not self.canonicalization_config.include_na_transcript_type and not self.genome_config.is_gff_from_ncbi:
            nan_columns_to_drop.append("transcript_type")
        return annotations_df.dropna(subset=nan_columns_to_drop)        

    def _parse_annotations_df_into_segments(self, annotations_df, parse_segments_using):
        # Extract the coding segments from it
        chromosomes = annotations_df["Chromosome"].unique().tolist()
        genome_transcribed_segments = {}

        for idx, ucsc_id in enumerate(
                chromosomes):  # Parse the whole df to know the transcribed segments later for the NT benchmark canonicalization
            if self.verbose:
                self._log(f"  Parsing record {idx + 1} / {len(chromosomes)} from gff...")

            positive_coding_segments, negative_coding_segments = self._parse_transcribed_features_from_gff(annotations_df, ucsc_id)

            if parse_segments_using is not None:
                parse_segments_using(positive_coding_segments, negative_coding_segments, ucsc_id)

            genome_transcribed_segments[ucsc_id] = {
                "positive": IntervalTree.from_tuples(positive_coding_segments),
                "negative": IntervalTree.from_tuples(negative_coding_segments)
            }

        return genome_transcribed_segments

    def _parse_transcribed_features_from_gff(self, df, ucsc_id):
        chromosome_feautres = df[df["Chromosome"] == ucsc_id]  # Filter the annotations for the target chromosome
        assert len(df) > 0  # Check that the chromosome actually exists

        # Build the positive and negative segments list
        positive_segments = []
        negative_segments = []

        for _, row in chromosome_feautres.iterrows():
            # The strand for each transcribed feature is the coding strand(opposite of the template)
            # Pyranges v0 converts the 1-based index of the GFF to 0-based (https://pyranges.readthedocs.io/en/v0/tutorial.html)
            if row["Strand"] == '+':
                bisect.insort_left(positive_segments, (row["Start"], row["End"]))
            elif row["Strand"] == '-':
                bisect.insort_left(negative_segments, (row["Start"], row["End"]))
            else:
                print(row)
                print(type(row))
                raise ValueError(f"Strand value {row['Strand']} not recognized")

        positive_segments = merge_segment_list(positive_segments)
        negative_segments = merge_segment_list(negative_segments)

        # Ignore chrM because it's so small in humans it's reasonable that it only has genes on one strand
        if len(positive_segments) == 0 and ucsc_id != "chrM":
            log.warning(f"positive_segments has length 0 on {ucsc_id}")
        if len(negative_segments) == 0 and ucsc_id != "chrM":
            log.warning(f"negtative segments has length 0 on {ucsc_id}")

        return positive_segments, negative_segments

    def find_annotated_alignment(self, seq_str):
        hits = self.aligner.map(seq_str)
        # Find the primary hit
        primary_hit = find_primary_hit(hits)
        if primary_hit is None:
            return None, AlignmentError.PrimaryAlignmentNotFound
        
        # If the primary hit isn't annotated, then find a hit with the same mapq that is
        if not primary_hit.ctg in self.genome_transcribed_segments:
            annotated_hit = self._find_annotated_hit(hits, primary_hit.mapq)
            if annotated_hit is None:
                return None, AlignmentError.AnnotatedAlignmentNotFound
            else:
                return annotated_hit, None
        else:
            return primary_hit, None
    
    def _find_annotated_hit(self, hits, primary_hit_mapq):
        for hit in hits:
            if hit.mapq >= primary_hit_mapq and hit.ctg in self.genome_transcribed_segments:
                return hit
        return None

    def align_transcript_position_with_seq(self, alignment, segment):
        chromosome_length = len(self.genome_transcribed_segments[alignment.ctg])
        segment_start = max(0, segment[0] - alignment.r_st)
        segment_end = min(chromosome_length, segment[1] - alignment.r_st)
        return segment_start, segment_end

    def _log(self, message):
        if self.hydra_logger:
            log.info(message)
        else:
            print(message)

    def cleanup(self):
        del self.aligner
        del self.genome_transcribed_segments
        gc.collect()

class AlignmentError(Enum):
    PrimaryAlignmentNotFound = 1
    AnnotatedAlignmentNotFound = 2