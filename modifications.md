# Modifications to the runs compared to the original ConvNova paper

Extracted verbatim from `Paper/paper.tex`.

## Evaluation protocol and run setup

> The evaluation protocol for a model consists of 5 runs with 5 different random seeds. Those 3 versions of the model are evaluated using that protocol on all downstream tasks using a fewer-parameter configuration to speed up the evaluation (see section \ref{hyperparameter-search}).
>
> Then, for the tasks where a promising difference appears between the canonicalization model vs the baseline and leakage control models, that same evaluation protocol is used but with the original, larger configuration used in ConvNova. This screening step is only used to decide which tasks deserve the additional compute of a full-scale evaluation.

> The 5 random seeds used were: 1938, 205, 8262, 1480 and 4436.

> It should be noted that a single pretrained model per trial was used for the rest of the downstream tasks. The difference in precision was to speed up the hyperparameter search, and the number of pretraining epochs was also changed from 400 to 200.

## Nucleotide Transformer Benchmark

> It should be noted that the most recent version of the benchmark was used, which was not used in the original ConvNova paper.

> For canonicalization, GRCh38 was used as the source genome and the annotations were obtained from the Genome Reference Consortium, since all the datasets come from human samples. The tasks were evaluated using a pooling downstream head.

## Genomic Benchmark

> Keep in mind that for \texttt{human\_ensembl\_regulatory}, the model performance dropped compared to the original ConvNova paper because a longer sequence length was used, where the whole dataset sequence is passed to the model instead of a trimmed version of it like in the original ConvNova paper (see more details in section \ref{sec:appendix_downstream_tasks}). It's theorized that this happens because of the pooling decoder diluting the signal located at the start of the sequence used for prediction.

> It should be noted that for the Genomic Benchmark, even though the original ConvNova paper reports a 378k parameter model for this task, the configurations provided in the paper and its GitHub repo don't match this parameter count. The authors were contacted with no response, so the configuration that comes closest to 378k parameters was chosen.

> The configuration is similar to the one used in ConvNova, with a slight difference in the sequence length for Genomic Benchmark. While they trained ConvNova on Genomic Benchmark with a sequence length of 1k (to have a direct comparison against transformer-based models), some tasks on Genomic Benchmark have an actual higher sequence length (one of the tasks, for example, has a sequence length of 5k), and it was decided to use those true sequence lengths to have a more accurate representation of the canonicalization algorithm's performance.

Sequence lengths for Genomic Benchmark:

| Task | Sequence length |
|---|---|
| dummy_mouse_enhancers_ensembl | 5000 |
| demo_coding_vs_intergenomic_seqs | 200 |
| demo_human_or_worm | 200 |
| human_enhancers_cohn | 500 |
| human_enhancers_ensembl | 573 |
| human_ensembl_regulatory | 802 |
| human_nontata_promoters | 251 |
| human_ocr_ensembl | 600 |
| drosophila_enhancers_stark | 3250 |

## BEND Benchmark

> The model was evaluated only on the gene finding tasks, where given a DNA sequence each nucleotide must be classified as intron, exon, or neither, and CpG methylation, where given a DNA sequence multilabel classification is performed on whether it is methylated or not in different cell lines.

> The chromatin accessibility and histone modification tasks are redundant compared to the tasks in the rest of the benchmarks, and therefore were not included in the evaluation.

> The zero-shot variant effects tasks were also skipped, since unlike models with an attention mechanism, each output token was determined by a very small local context, and if one wanted to mitigate that through pooling, the SNP token signal becomes too diluted for classification. Evaluating those tasks in that way gives artificially low results.

> For CpG methylation a linear head was used over the entire encoded sequence, and for gene finding each token is passed through a linear layer.

## Chromatin Profile Prediction

> The source genome of DeepSEA is hg19. In this specific case, for canonicalization, hg38 is used as the source genome in order to use the most up-to-date annotations, and the coordinates are resolved directly by aligning the DeepSEA sequences with minimap2 instead of using the GENCODE lift-over.

## Model / run configurations

> The same configuration as the original ConvNova paper was used, which used a configuration similar to HyenaDNA.

Different sequence lengths were used for each model:

| Parameters | 378k | 1.7M | 6.8M | 7.4M |
|---|---|---|---|---|
| Sequence length | 1k | 1k | 1k | 2k |

Downstream task configurations:

| Task | Sequence Length | Hidden dim | Num GCBs | Size | d_inner |
|---|---|---|---|---|---|
| NT Benchmark | 1k | 128 | 5 | 1.7M | 2 |
| Genomic Benchmark | Depends on the task, shown below | 128 | 5 | 378k | 4 |
| BEND Benchmark | 512 for CpG methylation, 16k for gene finding | 196 | 10 | 7.4M | 2 |
| Chromatin Profile Prediction | 1k | 256 | 5 | 6.8M | 2 |

Hyperparameter search configuration (differences across all tasks):

| Hidden dim | Num GCBs | Kernel size | d_inner | Size | Precision | Random seed |
|---|---|---|---|---|---|---|
| 64 | 4 | 5 | 2 | 262k | bf16 | 2222 |

> It should be noted that a single pretrained model per trial was used for the rest of the downstream tasks. The difference in precision was to speed up the hyperparameter search, and the number of pretraining epochs was also changed from 400 to 200.

Pretraining configuration (same as original ConvNova paper):

| Hyperparameter | Value |
|---|---|
| Learning Rate | β1=0.9, β2=0.999 |
| Batch Size | 256 |
| Weight Decay | 0.1 |
| Dropout | 0.0 |
| Optimizer | AdamW |
| Optimizer Momentum | β1=0.9, β2=0.999 |
| Learning Rate Scheduler | Cosine Decay |
| Epochs | 400 |
| Dilation rate | 4 |
| Precision | fp32 |
| Kernel size | 9 |

## Task classification / canonicalization handling modifications

> It is worth mentioning that tasks where the canonicalization was omitted were still evaluated because it was hypothesized that the canonicalized pretraining may change the performance of the model on those tasks.
