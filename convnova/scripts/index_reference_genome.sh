#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

minimap2 -x sr -d $SCRIPT_DIR/../data/hg38_index.mmi $SCRIPT_DIR/../data/hg38/GRCh38.p14.genome.fa
minimap2 -x sr -d $SCRIPT_DIR/../data/mouse_index.mmi $SCRIPT_DIR/../data/mouse/GRCm39.primary_assembly.genome.fa
minimap2 -x sr -d $SCRIPT_DIR/../data/drosophilia_index.mmi $SCRIPT_DIR/../data/drosophilia/ncbi_dataset/data/GCF_000001215.4/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna
minimap2 -x sr -d $SCRIPT_DIR/../data/roundworm_index.mmi $SCRIPT_DIR/../data/roundworm/ncbi_dataset/data/GCF_000002985.6/GCF_000002985.6_WBcel235_genomic.fna