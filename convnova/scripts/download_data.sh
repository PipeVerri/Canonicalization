#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $SCRIPT_DIR/../.env

mkdir -p $SCRIPT_DIR/../data/hg38
mkdir -p $SCRIPT_DIR/../data/mouse
mkdir -p $SCRIPT_DIR/../data/bend_benchmark

# ConvNova's training hg38 genome
curl https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz > $SCRIPT_DIR/../data/hg38/hg38.ml.fa.gz
curl https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed > $SCRIPT_DIR/../data/hg38/human-sequences.bed

# BEND benchmark
curl https://sid.erda.dk/share_redirect/aNQa0Oz2lY/data/gene_finding/gene_finding.bed > $SCRIPT_DIR/../data/bend_benchmark/gene_finding.bed
curl https://sid.erda.dk/share_redirect/aNQa0Oz2lY/data/gene_finding/gene_finding.hdf5 > $SCRIPT_DIR/../data/bend_benchmark/gene_finding.hdf5
curl https://sid.erda.dk/share_redirect/aNQa0Oz2lY/data/cpg_methylation/cpg_methylation.bed > $SCRIPT_DIR/../data/bend_benchmark/cpg_methylation.bed
curl https://sid.erda.dk/share_redirect/aNQa0Oz2lY/data/variant_effects/variant_effects_disease.bed > $SCRIPT_DIR/../data/bend_benchmark/variant_effects_disease.bed
curl https://sid.erda.dk/share_redirect/aNQa0Oz2lY/data/variant_effects/variant_effects_expression.bed > $SCRIPT_DIR/../data/bend_benchmark/variant_effects_expression.bed


# Download the genomes from GENCODE because they're in UCSC style names by default

# Human annotations and genome
curl https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.primary_assembly.annotation.gff3.gz > $SCRIPT_DIR/../data/hg38/gencode.v50.primary_assembly.annotation.gff3.gz
curl https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/GRCh38.primary_assembly.genome.fa.gz > $SCRIPT_DIR/../data/hg38/GRCh38.primary_assembly.genome.fa.gz
# Mouse annotations
curl https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M39/gencode.vM39.primary_assembly.annotation.gff3.gz > $SCRIPT_DIR/../data/mouse/gencode.vM39.primary_assembly.annotation.gff3.gz
curl https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M39/GRCm39.primary_assembly.genome.fa.gz > $SCRIPT_DIR/../data/mouse/GRCm39.primary_assembly.genome.fa.gz

gunzip $SCRIPT_DIR/../data/hg38/*.gz
gunzip $SCRIPT_DIR/../data/mouse/*.gz

# Download NCBI datasets
datasets download genome accession GCF_000001405.40 --include gff3,seq-report --filename $SCRIPT_DIR/../data/hg38/ncbi.zip
datasets download genome accession GCF_000001215.4 --include gff3,genome,seq-report --filename $SCRIPT_DIR/../data/drosophilia.zip
datasets download genome accession GCF_000001635.27 --include gff3,seq-report --filename $SCRIPT_DIR/../data/mouse/ncbi.zip
datasets download genome accession GCF_000002985.6 --include gff3,genome,seq-report --filename $SCRIPT_DIR/../data/roundworm.zip

unzip $SCRIPT_DIR/../data/hg38/ncbi.zip -d $SCRIPT_DIR/../data/hg38/
unzip $SCRIPT_DIR/../data/drosophilia.zip -d $SCRIPT_DIR/../data/drosophilia
unzip $SCRIPT_DIR/../data/mouse/ncbi.zip -d $SCRIPT_DIR/../data/mouse/
unzip $SCRIPT_DIR/../data/roundworm.zip -d $SCRIPT_DIR/../data/roundworm