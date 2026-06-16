#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REFERENCE_GENOME="$SCRIPT_DIR/../data/hg38/GRCh38.primary_assembly.genome.fa"
OUT_PATH="$SCRIPT_DIR/../data/reference_genome_index.mmi"

minimap2 -x sr -d $OUT_PATH $REFERENCE_GENOME