#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p $SCRIPT_DIR/../data/hg38/
curl https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz > $SCRIPT_DIR/../data/hg38/hg38.ml.fa.gz
curl https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed > $SCRIPT_DIR/../data/hg38/human-sequences.bed
gunzip $SCRIPT_DIR/../data/hg38/hg38.ml.fa.gz
curl https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50/gencode.v50.annotation.gff3.gz > $SCRIPT_DIR/../data/hg38/gencode.v50.annotation.gff3.gz
gunzip $SCRIPT_DIR/../data/hg38/gencode.v50.annotation.gff3.gz