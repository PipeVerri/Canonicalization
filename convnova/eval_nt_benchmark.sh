#!/bin/bash

# All NT benchmark tasks
TASKS="enhancers,enhancers_types,H3,H3K4me1,H3K4me2,H3K4me3,H3K9ac,H3K14ac,H3K36me3,H3K79me3,H4,H4ac,promoter_all,promoter_no_tata,promoter_tata,splice_sites_acceptors,splice_sites_donors,splice_sites_all"
#TASKS="enhancers"

python train.py -m \
    experiment=canon/nt_benchmark \
    dataset.dataset_name=$TASKS