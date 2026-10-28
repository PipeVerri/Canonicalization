# ConvNova configuration and source-diff audit

Audit date: 2026-10-27.

This is the complete audit report, including experiment coverage, effective configuration differences, documentation disagreements, relevant source changes, launcher behavior, conditional defects, and unresolved questions. It does not infer that an undocumented change was unwanted or that a source-level defect occurred in a completed run.

## 1. Reference rules and scope

The audit applies the following rules explicitly confirmed by the repository owner:

1. The supplied `../original/` repository is the comparison source of truth.
2. **Exception: original pretraining is treated as 400 epochs, despite the original YAML specifying 2000.** The known 2000-versus-400 discrepancy is not reported as an error in the paper.
3. Every other disagreement with the supplied original configuration is recorded, including statements in `modifications.md` that inaccurately describe the original.
4. Missing larger-model `paper` configurations for BEND and chromatin-profile prediction are intentional. Their absence is not a defect.
5. The scope is current configurations, their dependencies, relevant changed source, and launchers. Saved runs, historical override files, checkpoint contents, W&B runs, and completed-run results are outside scope.
6. The source review is **diff-driven**: changed sections and relevant new active modules were inspected. Unchanged source was read only where necessary to establish inheritance, parameter construction, or the consumer of a changed setting. This is not a line-by-line review of all source code.
7. The audit makes no experiment-configuration or implementation changes.

`AGENTS.md` was read before the investigation. In particular, the audit followed actual `src.*` imports and registries rather than assuming that the duplicate `src/convnova/*` tree executes.

### 1.1 Path notation

Unless prefixed otherwise, paths in this report are relative to this repository, the fork containing this report:

- `convnova/...`: the current fork.
- `../original/convnova/...`: the supplied original repository.
- `modifications.md`: the document whose claims are being checked.

Evidence links use repository-relative paths and source line anchors. They refer to the files inspected during this audit, not a particular published Git commit.

### 1.2 Classification

| Classification | Meaning |
|---|---|
| Documented and consistent | The relevant change is described in `modifications.md`, and the inspected configuration supports it. |
| Documentation disagreement | A numerical value or substantive statement in the document conflicts with the supplied source or effective configuration. |
| Undocumented difference | A real difference was found but is not stated in `modifications.md`. This classification does not establish that the change was accidental. |
| Implementation adaptation | New behavior supports canonicalization, control channels, or a new dataset integration; its existence is expected, but its exact semantics may need documenting. |
| Definite source-level defect | The configuration or source establishes a failure or incorrect transformation on the stated execution path. Full training was not needed to identify the mechanism. |
| Conditional defect or risk | A problem follows if a specific override, input shape, alignment, artifact, or execution path occurs. Its occurrence in actual datasets or completed runs was not checked. |
| Not verifiable from supplied original | The original lacks the relevant task-specific run configuration, or a statement depends on historical execution rather than checked-in source. |

Severity and intent are separate. A likely improvement over an original bug is still a methodological difference that should be disclosed.

## 2. Method, validation, and limitations

The investigation used read-only directory inspection, paired configuration/source diffs, inheritance tracing, source-consumer tracing, and dependency-free arithmetic. Work was divided by experiment family, with shared training and launcher behavior checked separately.

The ordinary `train.py`, HPO runners, canonicalizers, dataset constructors, and data-preparation entrypoints were not invoked for validation. They can write files, create output directories, download data, construct indexes, or replace generated datasets. For example, normal configuration printing writes `config_tree.txt`, and the HPO runner creates and enters a run directory.

The current audit interpreter did not provide Hydra, OmegaConf, PyTorch, PyTorch Lightning, or PyYAML. Consequently:

- Hydra composition was traced statically, not executed end to end.
- Parameter counts were derived analytically from active model construction; they were not obtained by instantiating the full project.
- Scheduler and effective-batch calculations are configuration-derived, not measurements of actual optimizer steps.
- A targeted dependency-free check of the checkpoint state transformation was performed using isolated source logic and fake in-memory objects. It was not a Lightning integration test.
- No full training, dataset download, data regeneration, or project test suite was run.
- Configured `train_len` values were not checked against actual split cardinalities.
- Conditional alignment and sequence-boundary problems were not tested against production data.

The original pretraining schedule is normalized to 400 epochs for comparisons, as instructed. Other original defaults are not silently replaced by presumed paper settings or undocumented historical command-line overrides.

Evidence for why ordinary entrypoints were avoided:

- [Configuration processing and printing](convnova/src/utils/train.py#L56-L130).
- [Run-directory creation and training call](convnova/src/utils/hpo_runner.py#L28-L54).
- [Canonicalizer output cleanup](convnova/src/utils/canonicalization/downstream_tasks.py#L55-L81).

## 3. Complete experiment coverage and inheritance

There are **27 current runnable experiment configurations**, plus the shared `experiment/base.yaml` settings file. They consist of nine pretraining experiments and eighteen downstream experiments.

The normal current composition is:

```text
configs/config.yaml
  selected experiment
    family pipeline
      trainer / loader / dataset / task or inline task
      optimizer / scheduler / callbacks
    experiment/base
    baseline experiment local settings
    paper overrides, where applicable
    canonicalized or raw-control wrapper, where applicable
  caller/command-line overrides
  runtime processing and construction
```

The selected baseline configurations explicitly load the pipeline before `experiment/base`, so the shared model/trainer/optimizer settings take precedence over generic pipeline defaults. Their final `_self_` entries then apply local settings.

Original ConvNova experiments specify their relevant settings directly; they do not inherit the fork's new shared base. The original `experiment/base.yaml` is an illustrative configuration, not the effective baseline for the original ConvNova experiment files.

### 3.1 All nine pretraining experiments

| Experiment ID | Immediate configuration dependencies and local role | Comparison family |
|---|---|---|
| [baseline/pretrain](convnova/configs/experiment/baseline/pretrain.yaml) | `pipeline/bert_hg38`, `experiment/base`, cosine scheduler; enables pretraining and FFNs | Original HG38 ConvNova pretraining, with intended small-model changes |
| [hpo/canonicalized/pretrain](convnova/configs/experiment/hpo/canonicalized/pretrain.yaml) | Baseline pretraining plus canonicalization defaults; replaces FASTA | Small canonicalized |
| [hpo/raw_control_signal/pretrain](convnova/configs/experiment/hpo/raw_control_signal/pretrain.yaml) | Baseline pretraining; adds raw-control data and model channel | Small raw control |
| [paper/pretrain_nt_benchmark](convnova/configs/experiment/paper/pretrain_nt_benchmark.yaml) | Baseline pretraining; restores NT width/kernel, 400 epochs and fp32; changes local batch | Larger NT |
| [paper/canonicalized/pretrain_nt_benchmark](convnova/configs/experiment/paper/canonicalized/pretrain_nt_benchmark.yaml) | Paper NT pretraining plus canonicalization defaults; replaces FASTA | Larger NT canonicalized |
| [paper/raw_control_signal/pretrain_nt_benchmark](convnova/configs/experiment/paper/raw_control_signal/pretrain_nt_benchmark.yaml) | Paper NT pretraining; adds raw-control data and model channel | Larger NT raw control |
| [paper/pretrain_genomic_benchmark](convnova/configs/experiment/paper/pretrain_genomic_benchmark.yaml) | Baseline pretraining; expansion 4, 400 epochs and fp32 | Paper Genomic |
| [paper/canonicalized/pretrain_genomic_benchmark](convnova/configs/experiment/paper/canonicalized/pretrain_genomic_benchmark.yaml) | Paper Genomic pretraining plus canonicalization defaults; replaces FASTA | Paper Genomic canonicalized |
| [paper/raw_control_signal/pretrain_genomic_benchmark](convnova/configs/experiment/paper/raw_control_signal/pretrain_genomic_benchmark.yaml) | Paper Genomic pretraining; adds raw-control data and model channel | Paper Genomic raw control |

The original pretraining comparison is [hg38-pretrain/convnova.yaml](../original/convnova/configs/experiment/hg38-pretrain/convnova.yaml).

### 3.2 All twelve NT and Genomic downstream experiments

| NT experiment | Genomic experiment | Immediate inheritance and role |
|---|---|---|
| [baseline/nt_benchmark](convnova/configs/experiment/baseline/nt_benchmark.yaml) | [baseline/genomic_benchmark](convnova/configs/experiment/baseline/genomic_benchmark.yaml) | Family pipeline, shared base, scheduler and local overrides; small-model baseline |
| [paper/nt_benchmark](convnova/configs/experiment/paper/nt_benchmark.yaml) | [paper/genomic_benchmark](convnova/configs/experiment/paper/genomic_benchmark.yaml) | Corresponding baseline with paper-family architecture and precision |
| [paper/canonicalized/nt_benchmark](convnova/configs/experiment/paper/canonicalized/nt_benchmark.yaml) | [paper/canonicalized/genomic_benchmark](convnova/configs/experiment/paper/canonicalized/genomic_benchmark.yaml) | Corresponding paper configuration plus parsed sequence source |
| [paper/raw_control_signal/nt_benchmark](convnova/configs/experiment/paper/raw_control_signal/nt_benchmark.yaml) | [paper/raw_control_signal/genomic_benchmark](convnova/configs/experiment/paper/raw_control_signal/genomic_benchmark.yaml) | Corresponding paper configuration; original sequences and separate raw signals |
| [hpo/canonicalized/nt_benchmark](convnova/configs/experiment/hpo/canonicalized/nt_benchmark.yaml) | [hpo/canonicalized/genomic_benchmark](convnova/configs/experiment/hpo/canonicalized/genomic_benchmark.yaml) | Corresponding small baseline plus parsed sequence source |
| [hpo/raw_control_signal/nt_benchmark](convnova/configs/experiment/hpo/raw_control_signal/nt_benchmark.yaml) | [hpo/raw_control_signal/genomic_benchmark](convnova/configs/experiment/hpo/raw_control_signal/genomic_benchmark.yaml) | Corresponding small baseline; original sequences and separate raw signals |

The original comparisons are [NT ConvNova](../original/convnova/configs/experiment/nt-benchmark/convnova.yaml) and [Genomic ConvNova](../original/convnova/configs/experiment/genomic-benchmark/convnova.yaml).

There is no separate `hpo/baseline` YAML family. The baseline experiments provide the small-model baseline configurations.

### 3.3 All six BEND and chromatin downstream experiments

| BEND experiment | Chromatin experiment | Role |
|---|---|---|
| [baseline/bend_benchmark](convnova/configs/experiment/baseline/bend_benchmark.yaml) | [baseline/chromatin_profile_prediction](convnova/configs/experiment/baseline/chromatin_profile_prediction.yaml) | Small baseline using untransformed data |
| [hpo/canonicalized/bend_benchmark](convnova/configs/experiment/hpo/canonicalized/bend_benchmark.yaml) | [hpo/canonicalized/chromatin_profile_prediction](convnova/configs/experiment/hpo/canonicalized/chromatin_profile_prediction.yaml) | Small model using pre-generated canonicalized data |
| [hpo/raw_control_signal/bend_benchmark](convnova/configs/experiment/hpo/raw_control_signal/bend_benchmark.yaml) | [hpo/raw_control_signal/chromatin_profile_prediction](convnova/configs/experiment/hpo/raw_control_signal/chromatin_profile_prediction.yaml) | Small model using untransformed data and raw annotation signals |

The original repository has no task-specific ConvNova BEND experiment or complete task-specific ConvNova DeepSEA experiment. Its [README explicitly lists DeepSEA/BEND scripts as unreleased](../original/README.md#L19-L24).

Original generic chromatin infrastructure does exist and was used for component comparison:

- [Chromatin pipeline](../original/convnova/configs/pipeline/chromatin_profile.yaml).
- [Chromatin dataset config](../original/convnova/configs/dataset/chromatin_profile.yaml).
- [Chromatin dataset implementation](../original/convnova/src/dataloaders/datasets/chromatin_profile_dataset.py).
- [Separate `deepsea` dataset config](../original/convnova/configs/dataset/deepsea.yaml), which is not selected by a supplied task-specific ConvNova experiment and lacks a matching registered datamodule in the inspected execution path.

Those files do not establish the actual original-paper BEND/DeepSEA training hyperparameters. The report does not substitute the NT experiment for an absent original BEND or DeepSEA run.

### 3.4 Transitive configuration coverage

The following reusable groups were checked where selected by these experiments:

| Group | Files and disposition |
|---|---|
| Root/shared | `configs/config.yaml`; `configs/experiment/base.yaml` |
| Pretraining pipeline/data | `pipeline/bert_hg38.yaml`; `dataset/bert_hg38.yaml` |
| NT pipeline/data | `pipeline/nucleotide_transformer.yaml`; `dataset/nucleotide_transformer.yaml`; old-version compatibility metadata considered separately |
| Genomic pipeline/data | `pipeline/genomic_benchmark.yaml`; `dataset/genomic_benchmark.yaml` |
| BEND pipeline/data | New `pipeline/bend_benchmark.yaml`; new `dataset/bend_benchmark.yaml` |
| Chromatin pipeline/data | `pipeline/chromatin_profile.yaml`; `dataset/chromatin_profile.yaml` |
| Trainer/loader | Selected `trainer/default.yaml`; `loader/default.yaml` |
| Optimizer | Selected `optimizer/adamw.yaml`; unchanged betas verified |
| Scheduler | Selected `scheduler/cosine_warmup_timm.yaml`; initial pipeline choices followed through experiment overrides |
| Callbacks | Selected `callbacks/base.yaml`; `callbacks/checkpoint.yaml` |
| Tasks/heads | Selected reusable task groups and inline/interpolated task/decoder definitions; actual registry construction followed |
| Canonicalization | New `canonicalization/default.yaml`, plus the consumers of settings used for experiment preparation and control inputs |

The family-local scheduler expressions replace the shared base's `dataset.__train_len` expressions in downstream configurations. An invalid shared placeholder is not reported as effective where a descendant overrides it.

The registry selects 18 current NT tasks, nine Genomic tasks, and two BEND tasks; DeepSEA is launched separately. This is 30 downstream tasks per unfiltered small-model evaluation. NT and Genomic together expose 162 task/config combinations across their twelve downstream configs, before seed replication and task screening.

## 4. Model configuration and exact parameter-count disagreements

### 4.1 Effective architecture by family

| Family | Hidden width | GCB kernel | GCB count | `d_inner` | Precision |
|---|---:|---:|---:|---:|---|
| Original pretraining | 128 | 9 | 5 | 2 | fp32 |
| Original NT downstream | 128 | 9 | 5 | 2 | fp32 |
| Original Genomic downstream | 64 | 5 | 5 | 4 | fp16 |
| Current small pretraining | 64 | 5 | 5 | 2 | bf16 |
| Current small downstream, all four benchmark families | 64 | 5 | 5 | 2 | bf16 |
| Current paper NT pretraining/downstream | 128 | 9 | 5 | 2 | fp32 |
| Current paper Genomic pretraining/downstream | 64 | 5 | 5 | 4 | fp32 |

All rows use one CNN stack. Pretraining enables the per-block FFNs and five-symbol prediction head; downstream configurations disable those FFNs and use benchmark-specific heads. That stage distinction already exists in the original code.

Evidence:

- [Current shared model defaults](convnova/configs/experiment/base.yaml#L7-L22).
- [Paper NT downstream](convnova/configs/experiment/paper/nt_benchmark.yaml#L6-L13).
- [Paper Genomic downstream](convnova/configs/experiment/paper/genomic_benchmark.yaml#L6-L14).
- [Original NT architecture](../original/convnova/configs/experiment/nt-benchmark/convnova.yaml#L6-L21).
- [Original Genomic architecture and precision](../original/convnova/configs/experiment/genomic-benchmark/convnova.yaml#L6-L40).

### 4.2 ARCH-01: five GCBs, not the documented four

`modifications.md:76-80` specifies four GCBs for the reduced configuration. The effective shared configuration specifies `num_conv1d: 5`, inherited by every small-model experiment. No inspected launcher changes that count to four.

The reported approximate 262k size is consistent with **five**, not four, GCBs in this implementation. Four blocks with the other reduced-model settings would yield a 221,248-parameter downstream backbone.

Classification: **documentation disagreement**, affecting the small-model architecture table across all benchmark families.

### 4.3 ARCH-02: paper Genomic width is 64, not 128

`modifications.md:72` gives hidden dimension 128. Both the supplied original Genomic experiment and the current paper Genomic experiment use hidden dimension 64, with five GCBs, kernel 5 and expansion 4.

Classification: **documentation disagreement**. The current paper Genomic architecture agrees with the supplied original architecture; the document's width is wrong.

### 4.4 ARCH-03: the supplied original Genomic config does recover approximately 378k

`modifications.md:25` states that the original paper/repository configuration did not match the reported 378k parameter count, and that a closest configuration was chosen.

For the supplied original repository, the active model plus pooled linear head gives:

- 378,050 parameters for two outputs.
- 378,115 parameters for three outputs.

The supplied original Genomic tuple already matches the current paper Genomic tuple. Therefore, the claim that the supplied original configuration cannot recover approximately 378k is contradicted by this checkout and this counting convention.

This does not establish what repository revision was available when the authors were contacted, nor does it verify or dispute the contact history. It establishes the configuration/count relationship in the two repositories provided for this audit.

### 4.5 Parameter counts, including stage and control-channel differences

Counts include biases and LayerNorm parameters. They are analytical counts of the active implementation.

| Family | Downstream backbone, FFN off | Pretraining FFNs added | Pretraining head added | Full pretraining | Raw-control pretraining |
|---|---:|---:|---:|---:|---:|
| Small: H64/K5/5 blocks/inner2 | 262,592 | 21,440 | 325 | **284,357** | **285,509** |
| Paper NT/original pretraining architecture: H128/K9/5/inner2 | 1,688,448 | 83,840 | 645 | **1,772,933** | **1,775,237** |
| Paper Genomic: H64/K5/5/inner4 | 377,920 | 21,440 | 325 | **399,685** | **400,837** |

| Downstream family | Two-output total | Three-output total | Raw-control increment |
|---|---:|---:|---:|
| Small | 262,722 | 262,787 | +1,152 |
| Paper NT | 1,688,706 | 1,688,835 | +2,304 |
| Paper Genomic | 378,050 | 378,115 | +1,152 |

For the additional small-model downstream heads:

| Task | Outputs | Baseline/canonicalized total | Raw-control total |
|---|---:|---:|---:|
| CpG methylation | 7 | 263,047 | 264,199 |
| Gene finding | 9 | 263,177 | 264,329 |
| Chromatin profile | 919 | 322,327 | 323,479 |

Canonicalization alone adds no parameters. Raw control adds one input channel to each of the two input convolutions.

For width `h`, GCB kernel `k`, block count `n`, and expansion multiplier `r`, the downstream backbone count is:

```text
B = 92h
    + 2n(kh² + h)
    + 4nh
    + (2r + 2r²)h²
    + (5r + 3)h
```

Pretraining adds `n(h² + 3h) + 5(h + 1)`. A downstream C-output linear head adds `C(h + 1)`. The raw-control increment is `18h`.

Important counting details:

- Both stem convolutions have fixed kernel 9, even when the GCB kernel is 5.
- `args.hidden_dim` determines actual width; the extra `d_model` keyword is not the constructor's independent width authority.
- The shared MLP is not independently duplicated for every GCB.
- A rounded 262k label is reasonable for the reduced downstream backbone/binary model, but not for its full pretraining model or the 919-output chromatin model.
- A rounded 1.7M label can describe the NT downstream model; its full pretraining model is approximately 1.773M.
- The 378k Genomic downstream label should not be applied to its approximately 400k pretraining model without clarifying the counting convention.

Evidence: [active model construction](convnova/src/models/ConvNova/convnova.py#L58-L118), [original construction](../original/convnova/src/models/ConvNova/convnova.py#L57-L114), and [linear decoder construction](convnova/src/tasks/decoders.py#L38-L52).

### 4.6 ARCH-04: paper Genomic fp16-to-fp32 change

The supplied original Genomic experiment sets `precision: 16`. The current paper Genomic experiment sets `precision: 32`. The reduced Genomic runs use bf16.

The reduced-model precision change is documented. The larger Genomic fp16-to-fp32 difference is not separately documented and is not covered by an unqualified statement that the original larger configuration was restored.

## 5. Pretraining: all nine effective configurations

### 5.1 PRE-01: batch, device and optimization differences

The effective batch calculation uses:

```text
accumulation = ceil(global_batch_size / (devices × local_batch × nodes))
effective full-batch size = local_batch × devices × nodes × accumulation
```

| Setting | Original, with instructed epoch normalization | All three small variants | All three paper NT variants | All three paper Genomic variants |
|---|---:|---:|---:|---:|
| Epochs | 400 | 200 | 400 | 400 |
| Precision | fp32 | bf16 | fp32 | fp32 |
| Devices × nodes | 6 × 1 | 1 × 1 | 1 × 1 | 1 × 1 |
| Per-device training batch | 128 | 512 | 256 | 512 |
| Accumulation | 1 | 1 | 2 | 1 |
| Effective/global training batch | **768** | **512** | **512** | **512** |
| Per-device evaluation batch | 256 | 1024 | 512 | 1024 |
| Learning rate | 0.001 | 0.001 | 0.001 | 0.001 |
| AdamW betas | 0.9 / 0.999 | Same | Same | Same |
| Weight decay | **0.0** | **0.1** | **0.1** | **0.1** |

Full-batch sizes assume complete microbatches. Partial last batches can differ because effective `drop_last` is false.

Evidence:

- [Original trainer, batch, optimizer and global batch](../original/convnova/configs/experiment/hg38-pretrain/convnova.yaml#L28-L68).
- [Current shared trainer/default batch](convnova/configs/experiment/base.yaml#L24-L51).
- [Current baseline pretraining](convnova/configs/experiment/baseline/pretrain.yaml#L25-L64).
- [Paper NT pretraining](convnova/configs/experiment/paper/pretrain_nt_benchmark.yaml).
- [Paper Genomic pretraining](convnova/configs/experiment/paper/pretrain_genomic_benchmark.yaml).

The small architecture, bf16 and 200 epochs are documented. The effective-batch, device, evaluation-batch and pretraining weight-decay changes are not accurately described as the same original setup.

### 5.2 PRE-02: the documented pretraining batch 256 is not the effective batch

`modifications.md:89` says batch size 256. The supplied original configuration has six devices with local batch 128 and accumulation one: effective batch 768. Every current pretraining family has effective batch 512.

Paper NT does use a **local** batch of 256, but accumulates two microbatches. Calling that an effective batch of 256 would be incorrect.

The document should specify whether a batch-size entry is local/per-device or effective/global. With either convention, the statement that 256 is the unchanged original pretraining batch is not supported by the supplied original configuration.

### 5.3 PRE-03: weight decay changed from zero to 0.1

The original pretraining experiment explicitly sets `optimizer.weight_decay: 0.0`, overriding reusable optimizer defaults. The current baseline pretraining explicitly sets 0.1, inherited by all nine pretraining variants.

This is not a canonicalization-only implementation consequence. It is a separate optimization change and contradicts the table heading claiming the listed 0.1 is the same as original ConvNova.

### 5.4 PRE-04: scheduler formula preserved, resolved horizon changed

Both versions use the estimated length:

```text
ceil(1,000,000,000 / 1024) = 976,563
updates_per_epoch = ceil(estimated_train_len / global_batch_size)
t_initial = updates_per_epoch × epochs
warmup_t = 0.01 × t_initial
```

| Configuration | Estimated updates/epoch | `t_initial` | `warmup_t` |
|---|---:|---:|---:|
| Original normalized to 400 epochs | 1,272 | 508,800 | 5,088 |
| Current small, 200 epochs | 1,908 | 381,600 | 3,816 |
| Current paper NT/Genomic, 400 epochs | 1,908 | 763,200 | 7,632 |

These are resolved configuration estimates, not verified dataset update counts. Even after restoring 400 epochs, the fork does not restore the original scheduler horizon because the effective batch changed.

The scheduler type, step-based operation, one-percent warmup, initial warmup LR `1e-6`, LR floor `1e-4`, AdamW LR `1e-3`, betas and gradient clipping 1.0 are preserved.

Evidence: [original scheduler](../original/convnova/configs/experiment/hg38-pretrain/convnova.yaml#L54-L68), [current scheduler](convnova/configs/experiment/baseline/pretrain.yaml#L51-L64), and [dataset length estimate](convnova/configs/dataset/bert_hg38.yaml).

### 5.5 PRE-05: input paths and genome-source relationship

The original BERT-HG38 dataset config uses null BED/FASTA paths, whose datamodule fallback resolves under `data/bert_hg38/`, subject to `DATA_PATH`. The current group supplies explicit project-root paths under `data/hg38/`.

This changes the default input-directory resolution and bypasses the former `DATA_PATH` fallback for those explicit paths.

Canonicalized variants select the generated `parsed/canonicalized.fasta` while retaining the fork's BED path. Raw-control variants retain the untransformed training FASTA and add chromosome-keyed control arrays.

The canonicalization preparation config reads `GRCh38.primary_assembly.genome.fa`, whereas baseline training refers to `hg38.ml.fa`. Those are different configured source files. The audit did not establish their base-by-base, contig-name, masking, or coordinate equivalence. It would be incorrect to assume file identity solely because both are hg38/GRCh38-related. Canonicalization is intended; any additional input-file difference needs separate provenance verification if strict paired-input equivalence is required.

Evidence:

- [Current BERT-HG38 dataset config](convnova/configs/dataset/bert_hg38.yaml).
- [Original fallback paths](../original/convnova/src/dataloaders/genomics.py#L810-L814).
- [Original data-root fallback](../original/convnova/src/dataloaders/base.py#L23-L27).
- [Canonicalization input/output paths](convnova/configs/canonicalization/default.yaml#L17-L20).

### 5.6 PRE-06: removal of double BERT masking

The original five-symbol branch calls the in-place `bert_mask` function twice on the same `data` tensor. The first result is assigned and then discarded; a second call supplies the returned masked input and selection mask. The fork calls it once.

Consequences:

1. Original first-pass corruption can remain at positions outside the second, returned selection mask.
2. With independent 15% selections, the expected union of selected positions is 27.75%, versus 15% for one pass. This is a **selection-union rate**, not a claim that 27.75% of bases necessarily change value: the replacement policy includes unchanged selections and random replacements can coincide with the original value.
3. Both corruption distribution and random-number consumption change.
4. The original second call clones its auxiliary label tensor after first-pass corruption. However, the configured training loss uses the separately cloned outer `target` and the returned MLM mask, not that auxiliary label tensor. The audit therefore does **not** claim that the configured ground-truth targets are automatically first-pass-corrupted.

This likely fixes an original implementation defect, but it is still an undocumented change to the pretraining objective's input corruption.

Evidence:

- [Original two calls and outer target](../original/convnova/src/dataloaders/datasets/hg38_dataset.py#L383-L390).
- [Current single call](convnova/src/dataloaders/datasets/hg38_dataset.py#L432-L441).
- [Configured loss consumption](convnova/src/tasks/metrics.py#L291-L296).

### 5.7 PRE-07: explicit padding exclusion from MLM and model computation

The fork preserves tokenizer attention masks before mapping the tokenizer's larger vocabulary into A/C/G/T/N IDs 0-4. It supplies the valid-token mask to BERT masking, returns it as batch extras, and passes it through to the model.

The original five-symbol branch passes PAD ID 5 to a masking function after remapping all values to IDs 0-4. Converted padding therefore is not excluded using that nonexistent ID. The fork uses the explicit mask to exclude it.

This changes which positions can contribute to MLM selection and changes model features around padding. It applies to baseline and canonicalized/control experiments alike; it is not limited to the new input channel.

Evidence: [current eligibility mask](convnova/src/dataloaders/datasets/hg38_dataset.py#L278-L284), [tokenizer attention-mask retention](convnova/src/dataloaders/datasets/hg38_dataset.py#L403-L441), and [original masking call](../original/convnova/src/dataloaders/datasets/hg38_dataset.py#L383-L390).

The standard replacement policy is preserved: 15% selection, with 80% masked/N, 10% random A/C/G/T, and 10% unchanged among selected positions. Random replacement excludes the N ID.

### 5.8 PRE-08: EOS-as-N is retained, not a new sixth class(todo)

Both original and fork enable EOS for pretraining. The unchanged character tokenizer appends SEP/EOS; after the five-symbol conversion it becomes N/ID 4. For an ordinary full-length 1024-token example, this normally means 1023 DNA tokens and an EOS-as-N token, rather than 1024 DNA tokens plus an extra token.

The EOS position is attention-valid and can be selected for MLM. Real N tokens are also eligible. `replace_N_token` remains false and is bypassed by the early-return five-symbol branch even if enabled there.

This is retained behavior, included because it determines control alignment and the interpretation of the nominal 1024 sequence length. It is not classified as a new fork defect.

Evidence: [unchanged tokenizer special-token construction](convnova/src/dataloaders/datasets/hg38_char_tokenizer.py#L86-L94) and [five-symbol conversion](convnova/src/dataloaders/datasets/hg38_dataset.py#L429-L441).

### 5.9 PRE-09: new raw-control handling and conditional alignment defects

The fork loads chromosome-keyed raw-control arrays, shares the resulting dictionary among dataset splits, slices corresponding genomic intervals, clones/converts values to float, zeros controls at all MLM-selected positions, and stacks sequence/control as two input planes.

Current defaults disable shift and reverse-complement augmentation. New guards reject unsupported control/augmentation combinations.

Conditional problems:

1. **Controls are not tokenized, truncated and padded with the DNA sequence.** In a full-length example with EOS, the final control can still describe the removed last genomic base while the corresponding input token is EOS-as-N. Short boundary intervals can produce a control vector shorter than the padded token/mask vector, leading to indexing or stacking failure. The checked-in defaults do not prove that the BED data triggers the short-boundary failure.
2. Chromosome-key availability, array lengths and coordinate correspondence are not validated when controls are loaded.
3. In the alternate tokenizer-enabled random-mask path, the returned masked tuple contains `None`. Ordinary default collation does not support that element. This path is dormant for the nine default pretraining experiments, which select the five-symbol BERT branch.

Evidence: [control slicing and augmentation handling](convnova/src/dataloaders/datasets/hg38_dataset.py#L83-L135), [control masking/stacking](convnova/src/dataloaders/datasets/hg38_dataset.py#L314-L320), [tokenization versus control handling](convnova/src/dataloaders/datasets/hg38_dataset.py#L389-L465), [alternate random-mask tuple](convnova/src/dataloaders/datasets/hg38_dataset.py#L244-L254), and [datamodule control loading](convnova/src/dataloaders/genomics.py#L1086-L1118).

### 5.10 Pretraining settings preserved or not effectively changed

- Train/validation/test nominal length remains 1024.
- Character tokenizer, five-symbol conversion, EOS enabled, no RC augmentation, no fixed-length validation, no interval padding, no N replacement.
- Alphabet size 5, dropout 0, dilation base 4, one stack, no final convolution.
- Pretraining mode on, representation mode off, FFNs on.
- Effective pretraining worker count remains 12. The reusable dataset-group worker default changed from 4 to 12, but the original experiment already explicitly selected 12.
- Evaluation batch remains defined as twice the local training batch, although the resolved values changed.
- The selected pretraining pipeline, loader group, trainer group, AdamW group, cosine scheduler group and checkpoint callback are unchanged.
- Generic loader YAML includes batch 50 and `drop_last: true`, but the BERT datamodule's own loader methods use datamodule attributes and effectively default to `drop_last: false` in both versions. Those generic YAML fields are not the effective pretraining values.
- New generic HG38 persistent-worker, prefetch, caching and custom-collation changes do not automatically apply to BERT pretraining. `BertHG38` has its own loader implementation.

Evidence: [original BERT loader](../original/convnova/src/dataloaders/genomics.py#L905-L946), [current BERT loader](convnova/src/dataloaders/genomics.py#L1145-L1185), and [generic HG38 loader changes](convnova/src/dataloaders/genomics.py#L203-L254).

## 6. Nucleotide Transformer downstream audit

All findings in this section apply to all six NT descendants unless limited to the small or paper family.

### 6.1 NT-01: experiment-level differences

| Setting | Original effective value | Current effective value | Classification |
|---|---|---|---|
| Architecture | H128/K9/5 blocks/inner2 | Small H64/K5/5/inner2; paper restores original tuple | Reduced model documented; documented four-block count wrong |
| Precision | fp32 | Small bf16; paper fp32 | Reduced precision documented |
| Devices | 2 | 1 | Undocumented execution change |
| Per-device training/evaluation batch | 128 | 256 | Undocumented local execution change |
| Effective/global batch | 256 | 256 | Preserved |
| Accumulation | 1 | 1 | Preserved |
| Default seed | 48 | 2222 before five-seed wrapper overrides | Search seed documented; standalone behavior differs |
| Default task | `H3K36me3` | `enhancers` | Undocumented standalone default change |
| Initial checkpoint | Null unless supplied externally | Fixed baseline seed-2222 path unless overridden | New default/provenance constraint |
| Padding | Left | Right | Undocumented scientific behavior |
| Returned padding mask | False | True | Undocumented scientific behavior |
| Data location | Dataset fallback, environment-aware | Explicit project-root NT directory | Portable path change; fallback behavior changed |
| Workers/pinning | Effective 1 / false | 12 / true | Undocumented performance/RNG-lifecycle change |
| Metric for promoters/splicing | Macro-F1 | MCC | Additional benchmark protocol difference |

Downstream epochs remain 20. AdamW LR 0.001, weight decay 0.1, betas, clipping 1.0, dilation 4, dropout 0, one stack, no downstream FFNs/final convolution, no EOS and no RC augmentation are preserved.

Evidence: [original experiment](../original/convnova/configs/experiment/nt-benchmark/convnova.yaml#L6-L105), [current baseline](convnova/configs/experiment/baseline/nt_benchmark.yaml#L15-L55), and [current dataset defaults](convnova/configs/dataset/nucleotide_transformer.yaml#L1-L18).

### 6.2 NT-02: revised dataset version is supported, with substantive task/metric changes

The preparation code names `nucleotide_transformer_downstream_tasks_revised`, supporting the document's newer-version statement.

In the table below, N is **configured `train_len` used for scheduler calculation**, not an independently verified actual sample count.

| Task mapping | Original N | Current N | Original length | Current length | Metric/classes |
|---|---:|---:|---:|---:|---|
| H3K36me3 | 31,392 | 30,000 | 500 | 1000 | MCC / 2 unchanged |
| H3K4me1 | 28,509 | 30,000 | 500 | 1000 | MCC / 2 unchanged |
| H3K4me2 | 27,614 | 30,000 | 500 | 1000 | MCC / 2 unchanged |
| H3K4me3 | 33,119 | 17,468 | 500 | 1000 | MCC / 2 unchanged |
| H3K9ac | 25,003 | 23,274 | 500 | 1000 | MCC / 2 unchanged |
| promoter_all | 53,276 | 30,000 | 300 | 300 | **Macro-F1 → MCC**, 2 |
| promoter_tata | 5,517 | 30,000 | 300 | 300 | **Macro-F1 → MCC**, 2 |
| promoter_non_tata → promoter_no_tata | 47,759 | 5,062 | 300 | 300 | **Macro-F1 → MCC**, 2 |
| splice_sites_all | 27,000 | 30,000 | 600 | 600 | **Macro-F1 → MCC**, 3 |
| splice_sites_donor → splice_sites_donors | 19,775 | 30,000 | 600 | 600 | **Macro-F1 → MCC**, 2 |
| splice_sites_acceptor → splice_sites_acceptors | 19,961 | 30,000 | 600 | 600 | **Macro-F1 → MCC**, 2 |
| enhancer → enhancers | 14,968 | 30,000 | 200 | 400 | MCC / 2 unchanged |
| enhancer_types → enhancers_types | 14,968 | 30,000 | 200 | 400 | MCC / 3 unchanged |
| Added H2AFZ | Absent | 30,000 | N/A | 1000 | MCC / 2 |
| Added H3K27ac | Absent | 30,000 | N/A | 1000 | MCC / 2 |
| Added H3K27me3 | Absent | 30,000 | N/A | 1000 | MCC / 2 |
| Added H3K9me3 | Absent | 27,438 | N/A | 1000 | MCC / 2 |
| Added H4K20me1 | Absent | 30,000 | N/A | 1000 | MCC / 2 |
| Removed H3 | 13,468 | Absent | 500 | N/A | Previously MCC / 2 |
| Removed H3K14ac | 29,743 | Absent | 500 | N/A | Previously MCC / 2 |
| Removed H3K79me3 | 25,953 | Absent | 500 | N/A | Previously MCC / 2 |
| Removed H4 | 13,140 | Absent | 500 | N/A | Previously MCC / 2 |
| Removed H4ac | 30,685 | Absent | 500 | N/A | Previously MCC / 2 |

The current registry and current metadata agree on eighteen tasks. The old-version registry/config are compatibility alternatives, not the selected current NT benchmark.

Changing promoter/splicing metrics changes both the reported quantity and best-checkpoint selection, because the family pipeline monitors `val/${dataset.metric}`. A blanket newer-benchmark statement does not enumerate this additional protocol change.

The large promoter count differences, in particular `promoter_tata: 30000` and `promoter_no_tata: 5062`, should not be assumed correct cardinalities without checking the actual revised dataset. This report records them as source settings rather than asserting a dataset-count error.

Evidence: [original task metadata](../original/convnova/configs/dataset/nucleotide_transformer.yaml#L15-L105), [current task metadata](convnova/configs/dataset/nucleotide_transformer.yaml#L21-L127), [task registry](convnova/src/hpo/benchmarks.py#L8-L48), and [revised-data preparation](convnova/scripts/prepare_downstream_data.py#L20-L21).

### 6.3 NT-03: downstream length is not uniformly 1k

The effective current lengths are:

- Histone-related tasks: 1000.
- Enhancers: 400.
- Promoters: 300.
- Splice-site tasks: 600.

The `modifications.md:71` NT downstream row saying 1k is therefore not a uniform description of the configured downstream tasks. If it means a maximum or nominal pretraining length, the table needs to say so.

### 6.4 NT-04: schedule durations change with task metadata

The effective formula remains `20 × ceil(N / 256)`, with one-percent warmup. It changes numerically because N changes.

| Task | Original `t_initial` | Current `t_initial` |
|---|---:|---:|
| H3K36me3 | 2460 | 2360 |
| H3K4me3 | 2600 | 1380 |
| promoter_tata | 440 | 2360 |
| promoter_non_tata/no_tata | 3740 | 400 |
| enhancer/enhancers | 1180 | 2360 |

The values are scheduler metadata consequences; actual split sizes were not measured.

Evidence: [original NT scheduler/global batch](../original/convnova/configs/experiment/nt-benchmark/convnova.yaml#L77-L98) and [current scheduler/global batch](convnova/configs/experiment/baseline/nt_benchmark.yaml#L39-L47).

### 6.5 NT-05: pooling existed in the original

The original and current NT pipelines both select identity encoding and a pooled `SequenceDecoder`. The NT pipeline file itself is unchanged. Therefore, the statement that a pooling head is used is correct, but pooling is not a new fork-only decoder family.

Actual behavior changes because the fork supplies padding masks and changes the masked-pooling implementation; these differences are covered in Section 10.

Evidence: [original NT pipeline](../original/convnova/configs/pipeline/nucleotide_transformer.yaml#L11-L26) and [current NT pipeline](convnova/configs/pipeline/nucleotide_transformer.yaml#L11-L26).

## 7. Genomic Benchmark downstream audit

All findings apply to all six Genomic descendants unless otherwise specified.

### 7.1 GB-01: experiment-level differences

| Setting | Original effective value | Current effective value | Classification |
|---|---|---|---|
| Architecture | H64/K5/5 blocks/inner4 | Small inner2; paper restores inner4 with H64 | Reduced model documented; paper width in document wrong |
| Precision | fp16 | Small bf16; paper fp32 | Small change documented; larger fp16→fp32 not separately documented |
| Devices | 4 | 1 | Undocumented execution change |
| Per-device training/evaluation batch | 32 | 64 | Undocumented |
| Global/effective training batch | 128 | 256 | **Undocumented optimization change** |
| Accumulation | 1 | 4 | **Undocumented optimization change** |
| Default seed | 43 | 2222 before seed-wrapper overrides | Different standalone/search default |
| Default task | Dummy mouse enhancers | Human regulatory | Different standalone default |
| Output count | Hardcoded 2 | Task-interpolated, including 3 for regulatory | Additional effective configuration correction/change |
| Length | Hardcoded 2500 | Task-specific table below | Change documented, original value misdescribed |
| Scheduler count | Hardcoded `train_len: 1210` | Task-interpolated counts | Additional scheduler change |
| Padding/mask | Left, no returned mask by default | Right, mask returned | Undocumented model/input difference |
| Data path | Private absolute NAS directory | Project-root directory | Portability change |
| Checkpoint | Null unless supplied externally | Fixed baseline seed-2222 path unless overridden | New default/provenance constraint |

Ten epochs, AdamW LR 0.001, weight decay 0.1, betas, clipping, dilation, dropout, one stack, no downstream FFNs/final convolution, no EOS and no RC augmentation are preserved.

The current comment saying the original used two GPUs with batch 128 each is incorrect for this family. The supplied original Genomic experiment uses **four GPUs with local batch 32**, effective batch 128.

Evidence: [original Genomic experiment](../original/convnova/configs/experiment/genomic-benchmark/convnova.yaml#L34-L93) and [current baseline](convnova/configs/experiment/baseline/genomic_benchmark.yaml#L25-L55).

### 7.2 GB-02: exact length/count/output/schedule comparison

The original dataset-group default is not the effective experiment value. The original experiment overrides length to 2500, output count to 2, and `train_len` to 1210. Merely changing its task name does not update these three fields.

| Task | Original effective cutoff | Current cutoff | Current configured N | Current outputs | Current `t_initial` |
|---|---:|---:|---:|---:|---:|
| dummy_mouse_enhancers_ensembl | 2500 | 5000 | 1,210 | 2 | 50 |
| demo_coding_vs_intergenomic_seqs | 2500 | 200 | 100,000 | 2 | 3910 |
| demo_human_or_worm | 2500 | 200 | 100,000 | 2 | 3910 |
| human_enhancers_cohn | 2500 | 500 | 27,791 | 2 | 1090 |
| human_enhancers_ensembl | 2500 | 573 | 154,842 | 2 | 6050 |
| human_ensembl_regulatory | 2500 | 802 | 289,061 | 3 | 11300 |
| human_nontata_promoters | 2500 | 251 | 36,131 | 2 | 1420 |
| human_ocr_ensembl | 2500 | 600 | 174,756 | 2 | 6830 |
| drosophila_enhancers_stark | No original task entry | 3250 | 6,914 | 2 | 280 |

All nine fork length values match the nine-row table in `modifications.md`.

For the original's default fixed count/global batch, `t_initial = 10 × ceil(1210/128) = 100`, and warmup is one step. The fork uses `10 × ceil(N/256)`, with warmup one percent of that value. Regulatory becomes 11300/113, not 100/1.

For the first eight tasks, the reusable metadata counts are not newly invented by the fork; the additional effective change is allowing them to govern the experiment instead of the original experiment's hardcoded 1210. The original comments explicitly instruct users to adjust settings manually, so these are template/default comparisons, not claims that the original authors actually used the dummy-mouse values for every task.

Drosophila is a new ninth configured task relative to the supplied original task inventory. It is represented in the document's length table and in the current launcher registry.

Evidence: [original hardcoded fields](../original/convnova/configs/experiment/genomic-benchmark/convnova.yaml#L57-L77), [original reusable task metadata](../original/convnova/configs/dataset/genomic_benchmark.yaml), [current metadata](convnova/configs/dataset/genomic_benchmark.yaml#L20-L55), and [current scheduler](convnova/configs/experiment/baseline/genomic_benchmark.yaml#L39-L47).

### 7.3 GB-03: claims about original 1k trimming and longer regulatory inputs are unsupported

`modifications.md:23,27` describes original Genomic training as using 1k and explains the regulatory drop as a result of the fork using a longer, full sequence.

The supplied original experiment sets cutoff **2500**. The fork's regulatory cutoff is **802**, which is shorter, not longer, than that original experiment cutoff. The original reusable dataset group's 1024 default is overridden by the experiment.

Consequently:

1. Original 1k training is not the supplied experiment setting.
2. The longer-than-original explanation is not supported by the supplied configs.
3. The specific pooling-dilution causal explanation cannot be established from configuration inspection.
4. Historical manual overrides could change the comparison, but those are outside scope and were not assumed.

The fork does adopt its stated task-specific lengths. That supported claim should be separated from the unsupported characterization of the original and the proposed performance explanation.

### 7.4 GB-04: input-file filtering and whitespace handling

The fork's dataset loader now selects only `.txt` sequence files, avoiding generated NPZ control files. The canonicalization processor strips surrounding whitespace before rewriting sequences.

These are relevant changed-source adaptations to the new directory contents. No unconditional incorrect sequence output was established from these changes.

Evidence: [current file filtering](convnova/src/dataloaders/datasets/genomic_bench_dataset.py#L195-L203), [original loading](../original/convnova/src/dataloaders/datasets/genomic_bench_dataset.py#L177-L182), and [canonicalized sequence handling](convnova/src/utils/canonicalization/downstream_tasks.py#L288-L300).

## 8. BEND downstream audit

The three BEND configurations are new integrations. The original has no equivalent experiment/pipeline/runtime from which the full original-paper run can be reconstructed. Comparisons against document claims are still possible.

### 8.1 BEND-01: selected tasks agree with the stated exclusions

The launcher registry selects only `cpg_methylation` and `gene_finding`. Chromatin accessibility, histone modification, and zero-shot variant-effect tasks are not selected by the BEND benchmark launcher.

This supports the task-selection statements in `modifications.md:45-49`. It does not independently establish the scientific explanation for excluding those tasks.

Evidence: [registered BEND tasks](convnova/src/hpo/benchmarks.py#L62-L71).

### 8.2 BEND-02: actual task settings and heads

| Setting | CpG methylation | Gene finding |
|---|---|---|
| Nominal length | 512 | 16,384 |
| Outputs | 7 multilabel outputs | 9 token classes |
| Local/evaluation batch | 256 | 8 |
| Accumulation at global 256 | 1 | 32 |
| Head | Masked mean pooling then linear projection | Per-token linear projection, wrapped as `.logits` |
| Task/loss | Base task / BCE | LM task / cross entropy, ignore `-100` |
| Metric | Batch-computed macro AUROC, then logged across batches | Accumulated epoch-level MCC |
| Padding | Right, explicit attention mask | Right, explicit attention mask and ignored padded labels |

The nominal lengths and token-linear gene head agree with the document.

The document's gene description, “intron, exon, or neither,” simplifies a **nine-class**, strand-specific task that also distinguishes splice-related classes. It is not an accurate statement of the configured class count.

The CpG statement “linear head ... over the entire encoded sequence” needs clarification. The actual head is a linear layer applied **after masked mean pooling**, not a flattened sequence-wide linear map with position-specific weights. If the wording only meant that the entire sequence contributes through pooling, it is imprecise rather than a proven intended-head mismatch.

Evidence: [task metadata](convnova/configs/dataset/bend_benchmark.yaml#L32-L62), [BEND head](convnova/src/dataloaders/datasets/bend_benchmark_dataset.py#L21-L36), [pooling head](convnova/src/tasks/decoders.py#L85-L131), [decoder attribute injection](convnova/src/tasks/decoders.py#L309-L325), and [LM flattening](convnova/src/tasks/tasks.py#L180-L203).

### 8.3 BEND-03: unresolved `dataset_dir` interpolation

The new dataset config contains paths such as `${..dataset_dir}/cpg_methylation.bed` and `${..dataset_dir}/gene_finding.hdf5`, including unused variant-effect task entries. No selected BEND defaults define `dataset_dir`. The comment promising a sensible default is not accompanied by that default.

Although the H5-backed dataset constructor does not need those nested metadata paths, normal config processing recursively traverses values and resolved config printing also resolves them. The missing interpolation target can therefore fail before dataset loading.

Affected: all three BEND configs unless a caller supplies the missing key or removes the stale metadata references.

Classification: **definite source-level configuration defect under normal interpolation traversal**. Hydra/OmegaConf execution was not available for an end-to-end reproduction.

Evidence:

- [Missing default and nested paths](convnova/configs/dataset/bend_benchmark.yaml#L1-L85).
- [Normal filtering call](convnova/src/utils/train.py#L67-L69).
- [Recursive resolved-value traversal](convnova/src/utils/config.py#L111-L124).
- [Resolved config printing](convnova/src/utils/train.py#L119-L122).

### 8.4 BEND-04: sequence length used as scheduler sample count

`dataset.train_len` is set to the selected task's `max_length`, not its number of training examples. The baseline schedule consumes that value as a sample count.

| Task | Consumed `train_len` | `t_initial` | `warmup_t` |
|---|---:|---:|---:|
| CpG methylation | 512 | 20 | 0.2 |
| Gene finding | 16,384 | 640 | 6.4 |

The actual dataset length comes from the loaded H5 records. The schedule's current count is unrelated to that split size.

Affected: all three BEND configurations, for both tasks. This is a definite semantic mismatch between the count field and its scheduler consumer; actual split cardinalities were not measured.

Evidence: [incorrect count expression](convnova/configs/dataset/bend_benchmark.yaml#L21), [scheduler consumer](convnova/configs/experiment/baseline/bend_benchmark.yaml#L27-L35), and [actual dataset length](convnova/src/dataloaders/datasets/bend_benchmark_dataset.py#L91-L92).

### 8.5 BEND-05: baseline requires parsed control directory even with controls disabled

The BEND constructor converts `control_signals_dir` to a path and asserts that it exists unconditionally. The default points to the parsed raw-control directory, even for baseline/canonicalized experiments whose control-input flag is false.

Therefore, a baseline H5 dataset alone is insufficient: baseline unexpectedly depends on canonicalization/control artifacts. The raw-control variant legitimately requires those files; the unconditional baseline dependency is separate.

Evidence: [unconditional assertion](convnova/src/dataloaders/genomics.py#L299-L307) and [default control path/flag](convnova/configs/dataset/bend_benchmark.yaml#L27-L28).

### 8.6 BEND-06: splits exist despite the “no test set” comment

The baseline excludes the test loader with a comment saying the benchmark has no test set. The implemented H5 datamodule constructs train, validation and test splits. The default evaluation route therefore omits an existing separate test split.

The false comment is definite. Whether validation-only selection/reporting is the intended protocol requires the owner's decision. No completed reported result is inferred.

Evidence: [test-loader exclusion](convnova/configs/experiment/baseline/bend_benchmark.yaml#L34-L43), [split construction](convnova/src/dataloaders/genomics.py#L356-L374), and [evaluation route](convnova/train.py#L657-L681).

### 8.7 BEND-07: encoding, masks, labels and H5/control identity

New integration behavior:

- Character tokens are mapped to A/C/G/T/N IDs 0-4.
- Tokenizer validity masks survive remapping to N.
- Sequence labels are truncated with the sequence and padded using `-100`.
- Controls are trimmed/right-padded to the runtime sequence length.
- H5 conversion preserves within-split row order structurally.
- Control lists are generated in matching traversal order, but runtime pairing is positional rather than checked by record ID.

Freshly co-generated files are structurally aligned. Mixing stale control arrays with a different H5 file can silently mispair records, because identity/count correspondence is not verified. This is a conditional artifact-consistency risk, not evidence of actual mispairing.

Evidence: [token/label/control handling](convnova/src/dataloaders/datasets/bend_benchmark_dataset.py#L97-L136), [H5 write order](convnova/src/utils/bend_benchmark_h5.py#L34-L45), and [H5 read order](convnova/src/utils/bend_benchmark_h5.py#L69-L87).

The BEND dataset's left-padding guard, like the NT/Genomic guards, rejects left padding even when controls are disabled. The default right-padding configs avoid that failure.

## 9. Chromatin-profile/DeepSEA downstream audit

### 9.1 DEEP-01: differences from original generic chromatin infrastructure

These comparisons are against available generic scaffolding, not a recovered original-paper ConvNova DeepSEA experiment.

| Property | Original scaffold | Current effective behavior |
|---|---|---|
| Input | Private hg19 FASTA plus coordinate/target CSVs | Preprocessed MAT-derived NumPy arrays |
| Nominal length | 1024 | 1000 |
| Flanks | Extends a 1000-base coordinate window by 12 on each side | Uses the entire stored sequence array; no reference-window reconstruction |
| Runtime reference | hg19 config/FASTA, with optional coordinate conversion | No runtime reference field; offline canonicalizer aligns to hg38 |
| Padding | Enabled, left | No runtime padding/mask for fixed arrays |
| Encoding | FASTA string through tokenizer | Already encoded arrays returned directly |
| Dataset name field | `dataset_name: chromatin_profile` | Removed |
| Declared train count | No corresponding field | `train_len: max_length`, incorrectly used as sample count |
| Workers/pinning | Constructor defaults 1 / false | Dataset config 0 / true |
| Loss | Multilabel BCE | BCE preserved |
| Metrics | AUROC, precision, recall, F1 | Only lowercase function `auroc` |
| Monitor | `val/accuracy` in original scaffold | `val/auroc` |
| Training metrics | Original generic behavior | Function metrics skipped during training |
| Scheduler | Generic plateau | Experiment selects cosine-warmup-timm |
| Outputs | 919 | 919 preserved |
| Encoder/decoder | Identity / pooled linear | Identity / pooled linear preserved |

The 1024-to-1000 and flank changes are not explicitly enumerated in the document's approximate “1k” label. They are real input differences from the supplied scaffold.

The original scaffold's monitor/metric combination itself is not internally sufficient to establish a working original-paper run: it names `val/accuracy` while the reusable multilabel task lists other metrics. The fork's different monitor may be a correction, but it remains a difference.

Evidence: [original dataset config](../original/convnova/configs/dataset/chromatin_profile.yaml), [original pipeline](../original/convnova/configs/pipeline/chromatin_profile.yaml), [original task](../original/convnova/configs/task/multilabel_classification.yaml), [original flank extension](../original/convnova/src/dataloaders/datasets/chromatin_profile_dataset.py#L174-L177), [current dataset config](convnova/configs/dataset/chromatin_profile.yaml), [current pipeline](convnova/configs/pipeline/chromatin_profile.yaml), and [replacement runtime dataset](convnova/src/dataloaders/datasets/chromatin_profile_dataset.py#L18-L57).

### 9.2 DEEP-02: zero-worker default is incompatible with the changed loader

The current chromatin dataset config sets `num_workers: 0`. Its datamodule stores that value and uses the inherited HG38 loader. The fork added unconditional `prefetch_factor=4` to that loader.

PyTorch's DataLoader does not accept a non-None prefetch factor with zero workers. Consequently, the checked-in chromatin default fails at loader construction on the project's PyTorch API.

Affected: baseline, canonicalized and raw-control chromatin configurations. The problem also affects NT/Genomic/BEND overrides that set their dataset worker count to zero.

Changing `loader.num_workers` alone does not repair this path: the inherited method reads `self.num_workers` set by the dataset/datamodule constructor, not that generic loader keyword.

Evidence: [zero-worker default](convnova/configs/dataset/chromatin_profile.yaml#L13), [constructor storage](convnova/src/dataloaders/genomics.py#L604-L621), [unconditional prefetch](convnova/src/dataloaders/genomics.py#L240-L253), and [original loader without that argument](../original/convnova/src/dataloaders/genomics.py#L201-L211).

### 9.3 DEEP-03: sequence length used as sample count

The chromatin config sets `train_len: ${.max_length}`, i.e. 1000. The baseline schedule consumes it as a training-example count.

At global batch 256 and ten epochs:

```text
t_initial = ceil(1000/256) × 10 = 40
warmup_t = 0.4
```

The actual dataset length comes from the stored arrays, not `max_length`. All three chromatin variants inherit this incorrect schedule metadata.

Evidence: [count definition](convnova/configs/dataset/chromatin_profile.yaml#L3-L4), [schedule consumer](convnova/configs/experiment/baseline/chromatin_profile_prediction.yaml#L26-L34), and [actual dataset length](convnova/src/dataloaders/datasets/chromatin_profile_dataset.py#L46-L47).

### 9.4 DEEP-04: unused `dataset_dir` and inert memmap setting

The baseline experiment sets `dataset.dataset_dir`, but the replacement datamodule consumes `dataset_path`. The unused key is swallowed through extra kwargs. Defaults happen to refer to the same unparsed directory, but overriding `dataset_dir` does not relocate the input.

The canonicalized configuration correctly overrides `dataset_path`, while still displaying the inherited unused `dataset_dir` value. This can make logged settings misleading.

The raw-control wrapper sets `load_control_signal_using_memmap: false`, but the chromatin loader does not consume that switch and always uses mmap for controls. BEND's downstream loader also ignores the switch, although its ordinary load happens to agree with false. NT/Genomic do not implement that downstream switch either.

Evidence: [unused baseline key](convnova/configs/experiment/baseline/chromatin_profile_prediction.yaml#L19-L24), [actual path consumer](convnova/src/dataloaders/genomics.py#L604-L617), [chromatin control loading](convnova/src/dataloaders/genomics.py#L674-L685), and [BEND control loading](convnova/src/dataloaders/genomics.py#L351-L354).

### 9.5 DEEP-05: validation/test routing and standalone evaluator

The chromatin datamodule loads train, validation and test arrays. The baseline excludes the test loader using the same incorrect “no test set” explanation found in BEND. Default post-fit evaluation is validation of the selected checkpoint, not an evaluation of the separately stored test split.

The separate `scripts/eval_deepsea.py`:

1. Disables the pretraining-only `load_backbone` hook before loading a fine-tuned checkpoint, correctly recognizing that it would otherwise discard the trained decoder.
2. Accumulates full-dataset predictions and labels for AUROC, avoiding the training path's batch-AUROC estimator.
3. Nevertheless sets `val_loader = model.train_dataloader()`. It therefore evaluates the **training split**, despite the variable name suggesting validation.
4. Prints both an equal mean over three task-group AUROCs and overall 919-output macro AUROC; these are different aggregation conventions and should not be interchanged.

The training-split selection is a definite source-level evaluator problem if the script is intended as held-out evaluation. No claim is made that published results came from this script.

Evidence: [all split loading](convnova/src/dataloaders/genomics.py#L667-L708), [test exclusion](convnova/configs/experiment/baseline/chromatin_profile_prediction.yaml#L33-L42), and [standalone evaluator](convnova/scripts/eval_deepsea.py#L49-L112), particularly line 73.

### 9.6 DEEP-06: source genome versus annotation-alignment reference

The document correctly distinguishes hg19-derived DeepSEA sequences from the hg38 reference used for annotation alignment. The canonicalizer converts the stored sequences back to strings, aligns them using the configured hg38 Mappy/minimap2 index, and uses GENCODE v50 annotations.

It does not replace input bases with hg38 sequence. “hg38 as source genome” should be phrased carefully: here it is the alignment/annotation coordinate reference.

The original optional hg38 conversion is a coordinate-lifter path. Calling it specifically a “GENCODE lift-over” implementation is not established by the supplied code.

Evidence: [reference config](convnova/configs/canonicalization/default.yaml#L40-L49), [aligner setup](convnova/src/utils/canonicalization/parsed_genome.py#L79-L97), and [DeepSEA processing](convnova/src/utils/canonicalization/downstream_tasks.py#L449-L459).

## 10. Shared changed model, task, metric and training behavior

### 10.1 SRC-01: explicit padding masking throughout the model

The fork adds a padding-mask argument and zeros:

- Padded one-hot DNA inputs in both branches.
- Padded control-channel values.
- Stem features after the first convolutions.
- Normalized intermediate features before convolution.
- Residual features between blocks.
- Final representation features.

Previously, remapped padding could be processed as real N input and contribute through convolution biases, normalization and pooling. The fork also changes masked pooling from an indexed cumulative-prefix calculation to a weighted average over valid positions.

This is a material representation/head-computation difference. It affects baseline reproduction too, not just control-channel experiments. It may be an improvement, but it is not documented in `modifications.md`.

The masked-pooling branch now handles left/right/noncontiguous masks by direct weighting, with a clamped denominator for all-padding cases. The no-mask branch is unchanged.

Evidence: [model mask handling](convnova/src/models/ConvNova/convnova.py#L123-L203), [current pooling](convnova/src/tasks/decoders.py#L85-L100), and [original pooling](../original/convnova/src/tasks/decoders.py#L91-L108).

### 10.2 SRC-02: masked task now forwards extras/state to encoder and model

`MaskedMultiClass.forward` previously invoked the encoder/model without the extra padding arguments. The fork forwards extras through the encoder, forwards resulting kwargs to the model, and includes state.

This is necessary for the new NT mask behavior to actually reach the backbone. It is an active source difference, not merely a new unused YAML flag.

Evidence: [changed task forward](convnova/src/tasks/tasks.py#L256-L263) compared with the corresponding original `MaskedMultiClass` implementation.

### 10.3 SRC-03: MCC changes from batch averaging to epoch-level accumulation

The original function computes MCC independently per batch, and the training logger averages logged scalar metrics over the epoch.

The fork removes `mcc` from ordinary function metrics, creates a multiclass `MatthewsCorrCoef` accumulator with ignored label `-100`, updates it each batch, and logs the accumulated epoch metric. Independent metric collections are selected/created by dataloader prefix.

Affected: current NT tasks and BEND gene finding. This changes the numerical definition of the reported epoch score and selected best checkpoint relative to the original batch-MCC average. A mean of batch MCCs is not generally equal to MCC computed from the aggregate confusion matrix.

Classification: undocumented evaluation-method change, plausibly correcting original aggregation.

Evidence: [metric conversion/construction](convnova/src/tasks/tasks.py#L43-L80), [prefix-specific metrics](convnova/train.py#L337-L350), and [MCC update/logging](convnova/train.py#L372-L385).

The standalone MCC function also now filters `ignore_index=-100`; ordinary task routing redirects the configured `mcc` name to the accumulated metric.
|
### 10.4 SRC-04: new AUROC is still a mean of batch-local AUROCs

The new lowercase `auroc` function:

1. Applies sigmoid and converts predictions/labels to NumPy.
2. Computes per-output ROC AUC within the current batch.
3. Skips outputs that have only one label value in that batch.
4. Returns the mean of the remaining output AUCs.
5. Returns `None` if no output is defined; the changed logger omits that batch's metric.

Lightning then aggregates the returned batch values. This is not full-dataset macro AUROC. It discards cross-batch rankings and permits a different subset of outputs to contribute in each batch. Sparse 919-output DeepSEA and seven-output CpG tasks are affected.

Changing batch size can change the reported estimator even with identical predictions. Returning `None` avoids NaN contamination but does not make the estimator equivalent to full-dataset AUROC.

Classification: new integration/evaluation behavior requiring explicit specification; potentially misleading if interpreted as standard whole-split macro AUROC.

Evidence: [new AUROC function](convnova/src/tasks/metrics.py#L164-L181), [None filtering and epoch logging](convnova/train.py#L364-L396), and [standalone full-dataset alternative](convnova/scripts/eval_deepsea.py#L86-L112).

### 10.5 SRC-05: training-metric suppression has narrower scope than its name

The chromatin pipeline sets `train.compute_train_torchmetrics: false`. The changed training function uses it to skip ordinary `self.metrics(...)` during training. Loss remains logged.

Despite the name, the separately implemented MCC update is outside that condition. If the same flag is set for an MCC task, it does not suppress the new MCC accumulator update.

This is a configuration-semantics trap, not a default chromatin MCC bug, since chromatin uses AUROC rather than MCC.

Evidence: [chromatin flag](convnova/configs/pipeline/chromatin_profile.yaml#L10-L19) and [separate metric branches](convnova/train.py#L362-L385).

### 10.6 SRC-06: post-fit best-checkpoint reload can mix backbone/head states

The fork newly evaluates `ckpt_path="best"` after fitting: `trainer.test(...)` if enabled, otherwise `trainer.validate(...)`. The original did not do this default post-fit validation and did not request a best checkpoint in its optional test call.

All current downstream baselines configure the pretraining-transfer `load_backbone` state hook, inherited by their descendants. `SequenceLightningModule.load_state_dict` applies that hook on **every** load; it has no initialization-only guard.

The hook:

1. Strips the incoming `model.` prefix.
2. Iterates only the backbone's current state keys.
3. Returns only reconstructed `model.*` entries.
4. Omits the top-level trained downstream `decoder.*` entries.

The downstream decoder is registered separately as `self.decoder`, outside `self.model`. Consequently, restoring a full best fine-tuned checkpoint through this hook does not restore the complete best model.

For the documented ordinary PyTorch Lightning 1.8.6 strategy path, the strategy calls `load_state_dict(checkpoint["state_dict"])` without an explicit strict argument, so this override's `strict=False` default applies. The expected result is **best-checkpoint backbone plus the existing last/current trained decoder**. It is not a newly randomized decoder in this post-fit, existing-model case.

If another version/strategy explicitly passes `strict=True`, the omitted decoder keys instead cause a missing-key error. `train.pretrained_model_strict_load` controls the explicit initial `load_from_checkpoint` call; it does not by itself determine strictness of a later Trainer reload.

The state transformation and non-strict/strict consequences were checked in isolation. Full Lightning restoration was not executed in the current environment.

Evidence:

- [New post-fit reload](convnova/train.py#L792-L801).
- [Unconditionally applied state hook](convnova/train.py#L221-L235).
- [Separate backbone/decoder registration](convnova/train.py#L188-L211).
- [Hook implementation](convnova/src/models/sequence/long_conv_lm.py#L1386-L1429).
- [Pinned environment](convnova/environment.yml#L11-L14).
- [Lightning 1.8.6 restoration](https://github.com/Lightning-AI/lightning/blob/1.8.6/src/pytorch_lightning/trainer/connectors/checkpoint_connector.py#L258-L271).
- [Lightning 1.8.6 strategy call](https://github.com/Lightning-AI/lightning/blob/1.8.6/src/pytorch_lightning/strategies/strategy.py#L361-L363).

The standalone DeepSEA evaluator already disables this hook when loading a fully fine-tuned checkpoint. That local fix does not fix the default training function's best-checkpoint reload.

### 10.7 SRC-07: inherited baseline checkpoint paths require family-specific overrides

Every direct downstream configuration inherits a fixed baseline seed-2222 pretraining checkpoint path unless a caller replaces it. This includes paper, canonicalized and raw-control wrappers.

The normal full-evaluation and paper-evaluation wrappers do inject their run's pretraining checkpoint, supporting the intended paired protocol. Direct config invocation is not equivalent to using those wrappers.

Conditional architecture failures include:

- Paper NT changes width 64→128 relative to default baseline pretraining. If the inherited file contains default small-baseline weights, stem shapes such as `(64,5,9)` do not match `(128,5,9)`.
- Paper Genomic changes expansion 2→4 relative to small-baseline pretraining.
- Raw-control models require a sixth stem input channel, unlike baseline pretraining's five.

Non-strict loading does not suppress incompatible tensor shapes. The hook's “shape MATCH” print statement does not perform a shape check.

Actual checkpoint contents were not inspected, so these are conditional path/config mismatches rather than assertions about a particular file. Matching family-specific pretraining checkpoints supplied by the intended wrappers avoid the default-path mismatch.

Evidence: [fixed NT path](convnova/configs/experiment/baseline/nt_benchmark.yaml#L50-L55), [fixed Genomic path](convnova/configs/experiment/baseline/genomic_benchmark.yaml#L50-L55), [full-evaluation injection](convnova/src/hpo/full_eval.py#L19-L26), [paper injection](convnova/src/hpo/original_paper_eval.py#L22-L33), and [unvalidated passed-through shapes](convnova/src/models/sequence/long_conv_lm.py#L1391-L1416).

### 10.8 SRC-08: loader caching, workers, prefetch and file handles

The generic HG38 loader used by NT, Genomic, BEND and chromatin gains:

- Cached train and validation loaders.
- Persistent workers whenever workers are positive.
- Prefetch factor 4 rather than the previous DataLoader default.
- Custom recursive/stacking collation.
- Worker initialization that reopens worker-local FASTA handles where applicable.

These can affect memory use, startup, worker RNG lifetime and behavior when a datamodule is reused. The default-zero-worker failure is described in DEEP-02.

Additional conditional issues:

- Genomic `setup` does not clear the newly cached loaders, whereas NT explicitly clears them. Reusing a Genomic datamodule after changing task/data can retain a stale loader. Normal per-task subprocess launchers do not demonstrate that failure.
- NT/Genomic left-padding guards reject left padding whenever padding is enabled, even when control signals are disabled. This prevents restoring the original padding through a simple override in baseline experiments. Default right-padding configs do not fail.
- NT FASTA selection now raises if multiple matching FASTAs exist rather than silently selecting the last one. This is validation hardening, not an established output error.

Evidence: [generic loader](convnova/src/dataloaders/genomics.py#L203-L254), [new helper region](convnova/src/dataloaders/genomics.py#L35-L58), [Genomic guard](convnova/src/dataloaders/genomics.py#L433-L434), [NT guard](convnova/src/dataloaders/genomics.py#L540-L541), and [NT FASTA checks](convnova/src/dataloaders/datasets/nucleotide_transformer_dataset.py#L86-L110).

### 10.9 SRC-09: model controls and strand switching

The active model gains `use_control_signal` and `switch_strands`, both default false at the Python-constructor level. Raw-control experiments add one input channel to both stems.

Packed sequence/control detection uses `seq.shape[1] == 2`, rather than an explicit structured-input type or the configuration flag. This is a fragile interface for nonstandard short or incorrectly packed inputs; it does not establish a default failure for the configured sequence lengths.

For raw controls, the complementary branch negates the signal. For the other control mode it uses `1-signal`. The model's second branch complements bases without reversing the sequence, so the control branch is likewise not reversed there. `switch_strands` swaps the branch inputs and changes no parameter count.

The original convolution/gating equations, shared MLP and dilation helper are otherwise preserved. With five blocks and dilation base 4, the implementation's existing dilation schedule is `[1,1,1,4,16]`; the fork did not introduce a new `[1,4,16,64,256]` schedule.

Evidence: [new constructor/channel settings](convnova/src/models/ConvNova/convnova.py#L58-L82), [input unpacking/complement/control routing](convnova/src/models/ConvNova/convnova.py#L123-L166), and the unchanged dilation helper in the same paired source diff.

### 10.10 SRC-10: runtime/logging changes

- `torch.backends.cudnn.benchmark` is now set from config, and the shared experiment base enables it. Seeds alone do not establish identical algorithm selection across environments. This is a reproducibility setting difference, not proof of nondeterministic completed results.
- W&B project changes from `dna` to `Canonicalization`.
- Logger initialization generates a fresh W&B ID in its new-run path, overriding the previously supplied ID there; resume/identity semantics therefore differ.
- W&B settings no longer explicitly request `start_method="fork"`.
- The post-fit return value now contains explicit evaluation results, enabling the new JSON/HPO scoring path.
- A `process_config_and_run` wrapper was added. Its comment says it finishes W&B, but no `wandb.finish()` call appears there. Normal benchmark execution uses subprocesses, so this is not proof of cross-task run reuse on that path.
- The Flash Attention RMSNorm import was moved inside the existing optional-import guard. This removes an unconditional optional-dependency import; it is an import-robustness adaptation, not an architecture change.
- An unused `TRANSPORT_ERROR` import was added to the active ConvNova model. No parameter or configured forward-path effect was identified.

Evidence: [cuDNN setting and wrapper](convnova/train.py#L768-L810), [logger setup](convnova/train.py#L118-L123), [W&B settings](convnova/train.py#L700-L710), [root project/output config](convnova/configs/config.yaml#L61-L78), and [guarded optional imports](convnova/src/models/sequence/long_conv_lm.py#L31-L40).

## 11. Canonicalized and raw-control preparation/configuration differences

These components are new relative to the supplied original. The review followed their config interfaces, sequence/label/control contracts and changed experiment behavior; it was not a line-by-line proof of the entire canonicalization algorithm.

### 11.1 CANON-01: exact variant inputs and control loading

| Variant | Sequence input | Extra model channel | Signal source |
|---|---|---|---|
| Baseline | Unparsed/raw data | None | None |
| Canonicalized NT/Genomic | Parsed task directories | None | Generated control files are not consumed by these configs |
| Raw-control NT/Genomic | Original task directories | Raw annotation channel | Parsed task directories, stored separately from sequences |
| Canonicalized BEND | Parsed H5 | None | No model control input |
| Raw-control BEND | Original H5 | Raw annotation channel | Parsed raw-control arrays |
| Canonicalized chromatin | Parsed arrays | None | No model control input |
| Raw-control chromatin | Original arrays | Raw annotation channel | Parsed raw-control arrays |

NT controls are per-FASTA `{stem}_raw_signals.npz` files keyed by sequence ID. Genomic controls are per-class `raw_control_signals.npz` files keyed by `.txt` filename. BEND/chromatin controls are matched positionally.

Canonicalization YAML supplies offline-preparation metadata. Loading it during a training composition does not canonicalize inputs online or prove that existing parsed files were produced with the currently logged settings.

Paper raw-control Genomic additionally includes canonicalization defaults; paper raw-control NT does not. Both use their explicit data/control paths. This asymmetry was recorded but no default input-selection failure follows from it alone.

Evidence: [NT control loading](convnova/src/dataloaders/datasets/nucleotide_transformer_dataset.py#L75-L98), [NT attachment](convnova/src/dataloaders/datasets/nucleotide_transformer_dataset.py#L145-L155), [Genomic controls/file keys](convnova/src/dataloaders/datasets/genomic_bench_dataset.py#L175-L203), and [Genomic attachment](convnova/src/dataloaders/datasets/genomic_bench_dataset.py#L234-L240).

### 11.2 CANON-02: reference/annotation defaults and document attribution

Relevant defaults include annotation level 2; CDJV inclusion false; pseudogenes included; NA transcript types excluded; RC transformation; 16 preprocessing workers; overlap behavior `both_break`; segment order retained; `full_break_config.orient_all_segments: true`; and non-partial downstream canonicalization.

Reference routing is:

| Data | Alignment/annotation reference |
|---|---|
| All current NT tasks | Human hg38/GRCh38-related reference |
| Human Genomic tasks | Human reference |
| Dummy mouse enhancers | Mouse reference |
| Drosophila enhancers | Drosophila reference |
| Human-versus-worm | Class-specific human/worm reference |
| BEND preparation | hg38 coordinates/reference extraction |
| DeepSEA | Stored hg19-derived sequences aligned to hg38 for annotations |

Human annotations are configured/downloaded as **GENCODE v50**, not annotations obtained from the Genome Reference Consortium as stated in `modifications.md:19`. Mouse uses GENCODE M39; fly/worm use NCBI GFF files.

Evidence: [complete canonicalization defaults](convnova/configs/canonicalization/default.yaml), [reference routing/exclusions](convnova/src/utils/canonicalization/downstream_tasks.py#L191-L333), and [GENCODE download targets](convnova/Makefile#L74-L95).

The human sequence-report path contains `sequene_report.jsonl`; its corresponding rename option is false, so the typo is not established as an active default failure. It remains a dormant path issue if that optional path is enabled.

### 11.3 CANON-03: omitted transformations still retain evaluation coverage

Configured exclusions include all NT promoter and splice-site tasks and the entire Genomic `human_nontata_promoters` task. The Genomic processor additionally excludes the promoter class inside `human_ensembl_regulatory` in code.

Excluded tasks/classes retain original sequence, while raw annotation signals can still be generated. The benchmark registries still evaluate those tasks. This supports `modifications.md:102`.

The class-specific regulatory exclusion is not enumerated in the modifications document and is easy to miss if only YAML skip lists are reviewed.

Evidence: [YAML exclusions](convnova/configs/canonicalization/default.yaml#L25-L38), [NT processing](convnova/src/utils/canonicalization/downstream_tasks.py#L191-L242), and [Genomic class-specific policy](convnova/src/utils/canonicalization/downstream_tasks.py#L244-L333).

### 11.4 CANON-04: downstream strand selection contradicts the local stated orientation contract

The local comments say that a sequence aligned to the positive branch over a negative-strand transcribed feature should be reverse-complemented, and vice versa, so the model sees transcribed features in the stated positive orientation.

The selector actually takes **positive annotations for positive alignments** and **negative annotations for negative alignments**, then transforms those selected overlaps. Parsed annotation signs are explicitly described as coding-strand signs.

For example, for a positive-strand BEND sequence wholly inside a positive transcript, the default full-sequence rule transforms it. A positive-strand sequence wholly inside only a negative transcript is not selected by that helper. This is the opposite of the two explanatory comments.

This is a definite code-versus-comment/contract disagreement, affecting downstream canonicalization generally, not just BEND. The audit does not decide whether the comments or the implemented strand convention reflect the owner's intended scientific definition. That decision requires clarification.

Evidence: [same-strand selector](convnova/src/utils/canonicalization/downstream_tasks.py#L28-L36), [stated contract and transformation](convnova/src/utils/canonicalization/downstream_tasks.py#L127-L153), and [coding-strand interpretation](convnova/src/utils/canonicalization/parsed_genome.py#L184-L190).

### 11.5 CANON-05: raw-control sign is reference-relative, not explicitly query-relative

Raw controls write `+1` for reference-positive annotations and `-1` for reference-negative annotations. Coordinate projection reverses positions for reverse-strand alignments, but does not invert the signs. The model later negates raw controls for its complemented stream.

If the intended control invariant is transcription direction relative to the input/query sequence, reverse-strand alignment signs need reconciliation. If the intended invariant is reference-strand identity, the current sign convention may be deliberate. The source alone does not establish which was intended.

Negative annotations are written after positive annotations, so negative wins in overlaps; the representation cannot distinguish “negative only” from “both positive and negative” at those positions.

Classification: explicit new signal semantics plus an unresolved invariant question, not an asserted unconditional bug.

Evidence: [raw signal assignment](convnova/src/utils/canonicalization/downstream_tasks.py#L155-L162), [position-only reverse projection](convnova/src/utils/canonicalization/parsed_genome.py#L230-L243), and [model sign handling](convnova/src/models/ConvNova/convnova.py#L152-L160).

### 11.6 CANON-06: alignment projection assumes no clipping or indels

Annotation coordinates are projected to sequence indices using reference start/end and strand, ignoring query alignment start/end and CIGAR operations.

If a plus-strand alignment begins at query position 20, the first reference-aligned base is mapped as though it were query position zero. Insertions/deletions likewise invalidate a simple constant-offset mapping. Alignment selection imposes no explicit full-query coverage/identity/MAPQ minimum that would make these assumptions guaranteed.

Affected records could receive incorrect partial transformation boundaries or controls. Default full-sequence decisions can also be affected by the counted overlap. This is a conditional source-level alignment risk; actual occurrence in task data was not measured.

Evidence: [alignment selection](convnova/src/utils/canonicalization/parsed_genome.py#L207-L228) and [projection](convnova/src/utils/canonicalization/parsed_genome.py#L230-L243).

### 11.7 CANON-07: full versus partial transformation and labels

The default is non-partial downstream transformation. A selected overlap must cover **strictly more than half** the sequence for a full-sequence transformation. Partial mode transforms selected segments instead.

BEND preparation extracts plus-reference sequence from hg38 coordinates without reversing according to BED strand; its canonicalizer therefore constructs a plus-strand alignment. That is internally consistent with its own extraction policy, but equivalence to external BEND conventions is not established by the original ConvNova repository.

For gene finding, transformed label regions are reversed and strand-related classes are swapped using `[4,5,6,7,0,1,2,3,8]`. Sequence/label operations are structurally matched for valid nine-class inputs. The local class-order comment is itself inconsistent in its intron/acceptor wording.

CpG multilabel vectors remain unchanged. DeepSEA targets are copied unchanged. That is the intended sequence-level multilabel handling for the configured RC transformation. HPO reverse-only/complement-only variants require their own label-semantic justification; RC invariance alone does not establish theirs.

Evidence: [full/partial rule](convnova/src/utils/canonicalization/downstream_tasks.py#L139-L153), [gene-label transformation and forced strand](convnova/src/utils/canonicalization/downstream_tasks.py#L342-L402), [DeepSEA label copy](convnova/src/utils/canonicalization/downstream_tasks.py#L484-L485), and [BEND preparation](convnova/scripts/prepare_downstream_data.py#L78-L100).

### 11.8 CANON-08: canonicalize-before-truncate can change the represented BEND interval

Full reversal/RC occurs before runtime truncation to 16,384. If an input gene-finding record is longer than that cutoff, baseline/raw retain its original prefix, whereas a full-RC canonicalized record retains the reversed original suffix.

Labels remain aligned with their corresponding transformed sequence, but the compared variants no longer necessarily predict the same original nucleotides. This is conditional on overlength records; actual lengths were not inspected.

Evidence: [preparation transformation](convnova/src/utils/canonicalization/downstream_tasks.py#L342-L402) and [runtime truncation](convnova/src/dataloaders/datasets/bend_benchmark_dataset.py#L97-L128).

### 11.9 CANON-09: preprocessing input assumptions

The BEND preparation path unconditionally drops the first BED row and pairs gene labels with BED rows positionally. It assumes a header and matching order rather than validating both contracts.

The DeepSEA conversion assumes particular train versus validation/test axis layouts and converts AGCT one-hot indices to ACGT. An all-zero one-hot position becomes A under argmax rather than N.

These are explicit new input contracts. The audit did not inspect upstream files to establish that any actual row is lost, axis is wrong, or zero-vector base occurs.

Evidence: [BEND preparation](convnova/scripts/prepare_downstream_data.py#L67-L90) and [DeepSEA preparation](convnova/scripts/prepare_downstream_data.py#L114-L156).

### 11.10 CANON-10: NT alignment statistics double-count the denominator

`align_and_canonicalize_record` increments `total_tasks`, then normally calls `_update_counter`, which increments it again. NT uses that ordinary path. Failure numerators are incremented once.

Therefore, the displayed unaligned/unannotated percentages on that path use a doubled denominator and are halved. This is a preprocessing-statistics defect, not a halving of dataset examples or scheduler counts.

Evidence: [counter updates](convnova/src/utils/canonicalization/downstream_tasks.py#L92-L113), [NT caller](convnova/src/utils/canonicalization/downstream_tasks.py#L235), and [percentage calculation](convnova/src/utils/canonicalization/downstream_tasks.py#L175-L183).

## 12. Launchers, HPO and evaluation protocol

The active benchmark subprocess target is `src.utils.hpo_runner`, not the duplicate `src.hpo.hpo_runner`. Both filenames exist; only the selected import/command path was treated as authoritative.

### 12.1 RUN-01: supported five-seed and shared-checkpoint protocol

The five-seed launcher lists exactly:

```text
1938, 205, 8262, 1480, 4436
```

For small evaluation, it launches pretraining once per seed/run and routes one checkpoint to NT, Genomic, BEND and DeepSEA. For paper NT/Genomic evaluation, it launches the corresponding benchmark-specific pretraining experiment and can restrict downstream evaluation to selected tasks.

This supports the documented protocol as an available code path. It does not prove it was used for completed runs.

HPO search seed 2222 is distinct from the five final-evaluation seeds and is not itself a contradiction.

Evidence: [five-seed launcher](convnova/hpo/random_seeds.py#L5-L45), [small shared-checkpoint routing](convnova/src/hpo/full_eval.py#L19-L54), and [selected paper task routing](convnova/src/hpo/original_paper_eval.py#L19-L33).

### 12.2 RUN-02: caller overrides can supersede generated protocol values

Override ordering is significant:

- The seed launcher constructs `[generated_seed, *caller_overrides]`.
- Benchmark task iteration constructs `[generated_task, *caller_overrides]`.
- Full evaluation constructs `[generated_checkpoint, *caller_overrides]`, then appends task-family-specific overrides.
- A user-supplied `hydra.run.dir` follows the generated pretraining run directory.

Where Hydra accepts repeated ordinary overrides, the later caller setting can supersede the generated seed/task/checkpoint/directory. The launcher does not enforce uniqueness or reject protocol-changing overrides.

This is flexibility rather than proof of an incorrect run. It means the five-seed/shared-checkpoint claim depends on the actual call arguments, not solely on the wrapper's loop.

Task order in full evaluation is built using set subtraction, so execution order is not retained from the declared registry. Per-process seeding limits some order effects, but source order should not be assumed deterministic.

Evidence: [seed precedence](convnova/hpo/random_seeds.py#L25-L44), [task precedence](convnova/src/hpo/benchmarks.py#L95-L103), and [checkpoint/task overrides](convnova/src/hpo/full_eval.py#L12-L54).

### 12.3 RUN-03: Optuna launcher typo blocks normal first-trial construction

`hpo/phase1.py` invokes `trial.suggest_categorial(...)`. The Optuna method is `suggest_categorical`.

Every ordinary call to `build_overrides_dict` reaches that expression and raises `AttributeError` before canonicalization/training. The wrapper does not supply an exception policy that turns it into a functioning trial.

The script declares a twenty-trial TPE study, five startup trials, multivariate/group sampling, and seed 2222 for training, but the checked-in launcher is blocked as written.

Evidence: [typo](convnova/hpo/phase1.py#L47-L48), [sampler/study settings](convnova/hpo/phase1.py#L59-L72), and [objective wrapper](convnova/src/hpo/optimize_study.py#L55-L60).

### 12.4 RUN-04: additional HPO Hydra-key errors are masked by the typo

Even after correcting the typo, the search dictionary emits ordinary overrides for:

- `canonicalization.should_rc_at_end`.
- `model.switch_strands`.

Neither key is declared in the selected YAML tree. The override flattener emits `key=value` without `+` or `++`, so standard Hydra composition rejects the additions. A Python constructor default for `switch_strands` is not a declaration in Hydra's config schema/tree.

In the `full_break`/`orient_all_segments=true` branch, the search additionally emits `canonicalization.full_break_config.keep_segments_order`. The real declared/consumed key is `canonicalization.keep_segments_order`.

Adding `+` to the wrong nested key would make it accepted but still ineffective: the preparation entrypoint reads the top-level key.

Evidence: [search dictionary](convnova/hpo/phase1.py#L15-L48), [flattener](convnova/src/hpo/optuna_trial.py#L98-L113), [declared key](convnova/configs/canonicalization/default.yaml#L12-L15), and [consumer](convnova/scripts/canonicalize_genomes.py#L45-L50).

### 12.5 RUN-05: proposed end-RC hyperparameter has no active consumer

No active canonicalization consumer reads the proposed `canonicalization.should_rc_at_end` config key. Related name matches in the older `src/convnova/` implementation do not make it active in the launcher's `src.utils.canonicalization` path.

The active model does consume `switch_strands`, so that option would become meaningful after correct config declaration/override handling. Merely accepting `should_rc_at_end` as an extra config key would not implement an end-RC search dimension.

This is an active-versus-duplicate implementation mismatch in the search interface, not a claim that the entire active algorithm has no internally derived orientation decision.

Evidence: [proposed search keys](convnova/hpo/phase1.py#L38-L48), [active preparation consumer](convnova/scripts/canonicalize_genomes.py#L42-L64), and [active strand-switch consumer](convnova/src/models/ConvNova/convnova.py#L163-L166).

### 12.6 RUN-06: canonicalization subprocess failure is unchecked

The Optuna objective invokes canonicalization with `subprocess.run` but does not check its return code before launching pretraining/downstream evaluation. The training subprocess wrapper does check return codes.

Because canonicalization writes shared output paths, a failed preprocessing attempt can leave old or partial files. If those are still usable and training proceeds, a trial can consume data that does not correspond to its requested config. This is conditional stale-output reuse, not evidence that a completed trial actually did so.

Evidence: [unchecked preprocessing](convnova/src/hpo/optuna_trial.py#L134-L152), [checked training subprocess](convnova/src/hpo/benchmarks.py#L90-L93), and [shared output location](convnova/scripts/canonicalize_genomes.py#L67-L82).

### 12.7 RUN-07: `output_json=None` becomes a literal output filename

`run_all_tasks` defaults `output_json` to None but passes `str(output_json)`, producing the truthy string `"None"`. The subprocess builder therefore passes `--output-json None`. The runner resolves it as a real path and writes results after successful execution.

The ordinary five-seed small-evaluation path omits an explicit JSON path and exercises this behavior. It can create a shared file literally named `convnova/None`. The Optuna objective supplies an explicit JSON path and does not have this particular default issue.

This is unintended output routing, not inherently a training failure.

Evidence: [string conversion call sites](convnova/src/hpo/full_eval.py#L6-L54), [truthy output argument handling](convnova/src/hpo/benchmarks.py#L83-L85), and [runner path/write handling](convnova/src/utils/hpo_runner.py#L60-L77).

### 12.8 RUN-08: convenience caller is missing a required argument

`hpo/run_all_tasks_caller.py` supplies three positional arguments to `run_all_tasks` and does not supply required `experiments_folder`. The call raises `TypeError` before running its intended jobs.

Its BEND override also uses the obsolete `dataset.dataset_dir`-style route rather than changing the selected `h5_path`. The missing function argument is the immediate blocker; correcting it alone would not establish that all data overrides achieve their stated purpose.

Evidence: [caller](convnova/hpo/run_all_tasks_caller.py#L12-L28) and [required signature](convnova/src/hpo/full_eval.py#L6).

### 12.9 RUN-09: repair script can associate trials/checkpoints incorrectly

`hpo/fix_hpo.py` collects completed Optuna trials and separately collects W&B-derived checkpoint paths, then joins them with positional `zip`, without a trial-ID match or order/count validation. Unequal lists are silently truncated; `[12:]` skips positional pairs, not trial IDs.

Actual mispairing was not checked because W&B/saved-run inspection is outside scope. The association is nevertheless not guaranteed by the source.

The script also maps every trial parameter into the `canonicalization.*` namespace. That misroutes `switch_strands`, which belongs to `model`, and repeats the wrong nested `keep_segments_order` mapping. Its subprocess helper ignores failures. Despite the “both BEND tasks” comment, the gene-finding rerun is commented out and only CpG rerun is enabled.

Evidence: [positional checkpoint association](convnova/hpo/fix_hpo.py#L15-L37), [namespace mapping](convnova/hpo/fix_hpo.py#L43-L49), and [rerun calls](convnova/hpo/fix_hpo.py#L62-L85).

### 12.10 RUN-10: failed-trial checkpoint reuse is implemented

On resuming a failed trial, the study copies completed-task records into the next trial's key, enqueues the failed parameters, skips completed tasks, and uses the previous trial-number checkpoint directory.

This is an explicit exception to a simplistic interpretation of “a newly pretrained model for every numeric trial ID.” It can still represent one pretrained model for a resumed logical trial, but that distinction should be stated if numerical trial IDs are interpreted as independent training replicates.

No historical reuse correctness or actual checkpoint provenance was checked.

Evidence: [restart bookkeeping](convnova/src/hpo/optimize_study.py#L22-L53), [failed checkpoint selection](convnova/src/hpo/optuna_trial.py#L134-L151), and [checkpoint-directory choice](convnova/src/hpo/full_eval.py#L9-L21).

### 12.11 RUN-11: legacy benchmark shell script is not the current five-seed protocol

All four `benchmarks.sh` commands reference absent `experiment/canonicalized/` configuration names. The fourth also uses the obsolete `chromatin_profile` experiment name instead of `chromatin_profile_prediction`.

The script's seed is 4567, outside the documented five seeds. That is a valid arbitrary seed, but the script does not implement the stated five-run protocol. Its W&B group labels raw control while its obsolete experiment names label canonicalized data, making intent unclear even before the config-selection failure.

Evidence: [legacy script](convnova/benchmarks.sh#L1-L4).

### 12.12 RUN-12: larger-model BEND/DeepSEA options are accepted but not supplied

`run_on_select_tasks` accepts BEND and chromatin benchmark names even though the current paper family intentionally lacks those experiment files. Selecting them constructs absent pretraining/downstream IDs.

That is an unsupported launcher branch, not a request to add the intentionally absent larger configurations. Its chromatin branch also passes pretraining overrides rather than the constructed checkpoint-bearing downstream overrides.

For supplied paper NT/Genomic experiments, the intended selected-task checkpoint route exists.

Evidence: [accepted names and constructed paths](convnova/src/hpo/original_paper_eval.py#L8-L35).

### 12.13 RUN-13: HPO objective aggregation is task-group weighted

The trial scorer normalizes task metrics against baseline means/standard deviations, averages within declared task groups, and gives each group equal weight in the final mean. It is not a simple equal mean across all thirty tasks.

Its groups are histone, promoter, enhancers, splicing, general genomic content, gene finding, CpG methylation, and chromatin profile. Missing expected task scores or zero baseline standard deviations are not handled in this computation.

This is a new optimization protocol, with no original-repository counterpart. `modifications.md` does not specify this objective aggregation. Input baseline metric values and historical outputs were not validated.

Evidence: [per-task z-score](convnova/src/hpo/optuna_trial.py#L15-L27), [task groups](convnova/src/hpo/optuna_trial.py#L29-L70), and [grouped objective](convnova/src/hpo/optuna_trial.py#L72-L96).

### 12.14 RUN-14: output paths, W&B-null regression and labeling

The default output directory changed from microsecond-stamped `outputs/date/time` to second-stamped `outputs/job_type/group/date/time`.

Consequences:

1. Group/job type now determine filesystem placement.
2. `wandb=null` or removal of W&B breaks the default directory interpolation because it dereferences `wandb.job_type` and `wandb.group`. The original directory did not depend on W&B. An explicit W&B-independent `hydra.run.dir` avoids that specific problem; disabling W&B mode while retaining its keys also preserves interpolation.
3. Second-resolution naming reduces uniqueness for simultaneous same-group/job-type launches. A collision is conditional, not established in existing outputs.
4. Canonicalized and paper raw-control configs can inherit W&B group `baseline`; HPO raw-control wrappers explicitly change it. Labels alone do not establish the actual model/data variant.

Evidence: [root output config and documented disabling option](convnova/configs/config.yaml#L59-L78), [original output config](../original/convnova/configs/config.yaml#L75-L78), [shared group](convnova/configs/experiment/base.yaml#L2-L5), and [runner directory resolution](convnova/src/utils/hpo_runner.py#L42-L50).

### 12.15 RUN-15: runner path normalization edge cases

The new runner rewrites relative `train.pretrained_model_path` overrides relative to the previous working directory before entering the run directory. This is useful for ordinary relative checkpoint paths.

The parser uses unrestricted `split("=")`, indexes the second item, strips quote characters only for the absolute-path check, and does not special-case an empty or `null` checkpoint value. Therefore:

- An empty checkpoint override can index an empty string.
- `train.pretrained_model_path=null` can be rewritten into a filesystem path named `null` rather than retaining the intended null value.
- Paths containing `=` or quote characters require care and are not robustly normalized by this code.

These are conditional override-path defects, not failures of the normal absolute paths generated by the main wrappers.

Evidence: [checkpoint override rewriting](convnova/src/utils/hpo_runner.py#L19-L26).

## 13. Claim-by-claim reconciliation of `modifications.md`

This section checks the document itself. Detailed evidence and effective values are provided in the preceding sections; the rows below retain every substantive claim rather than replacing the report with an overall verdict.

| Document location | Claim | Audit result |
|---|---|---|
| Lines 7-9 | Five-seed evaluation of three variants on small models, followed by selected-task larger evaluation | Supported as launcher capability. Actual execution is outside scope. Small model is five GCBs, not the later stated four. Larger NT/Genomic coverage is intentional. |
| Line 11 | Seeds 1938, 205, 8262, 1480, 4436 | Matches the five-seed launcher exactly; later caller overrides can supersede generated seeds. |
| Lines 13 and 82 | One pretrained model per trial reused for downstream tasks | Supported by main full-evaluation routing. Direct configs inherit a fixed baseline path unless overridden; failed logical trials can reuse a previous numeric trial's checkpoint. |
| Lines 13 and 82 | Pretraining changes from 400 to 200 epochs | Correct under the explicitly instructed 400-epoch reference. No error is assigned for original YAML 2000. |
| Line 17 | Revised/newer NT version | Supported by the preparation dataset identifier. Task membership, lengths, configured counts and promoter/splicing metrics also change. |
| Line 19 | Human GRCh38-related reference for NT | Supported by reference routing/configuration. |
| Line 19 | Annotations obtained from GRC | Incorrect attribution for the selected source: GENCODE v50 is configured/downloaded. |
| Line 19 | Pooling downstream head for NT | Correct, but pooling already exists in the original pipeline. Mask-aware computation changes separately. |
| Line 23 | Regulatory performance drop caused by passing a longer sequence than original | Not supported by supplied config cutoffs: original 2500, current 802. Performance causation is not established by source inspection. |
| Line 25 | Supplied original Genomic config does not match approximately 378k | Contradicted by the supplied original architecture: 378,050/378,115 with two/three outputs. Historical repository revisions/contact history are not checked. |
| Line 27 | Original Genomic trained at 1k; fork adopts true lengths | Fork task-specific table matches config. Supplied original effective experiment cutoff is 2500, not 1k. |
| Lines 33-41 | Nine Genomic sequence lengths | All nine match current task metadata. Whether each is a true maximum/full sequence length was not established from dataset files. |
| Line 45 | Only gene finding and CpG BEND tasks evaluated | Matches registered BEND tasks. |
| Line 45 | Gene finding classifies intron/exon/neither | Incomplete description of the configured nine-class, strand-specific/splice-related label space. |
| Lines 47-49 | Accessibility/histone and zero-shot variant tasks excluded | Matches launcher task selection. Scientific rationale is not independently verified here. |
| Line 51 | CpG linear head over entire encoded sequence | Actual head is masked-mean pooling then linear projection. Wording requires clarification if it intends a flattened sequence-wide head. |
| Line 51 | Token-wise linear gene-finding head | Matches implementation. |
| Line 55 | DeepSEA sequences originate from hg19; annotations aligned using hg38/minimap2 | Broadly matches implementation. hg38 supplies annotation alignment coordinates, not replacement input bases. “GENCODE lift-over” attribution is not established by original code. |
| Line 59 | Same configuration as original ConvNova | Overbroad: even paper variants retain batch/precision/data/masking/metric/runtime differences detailed in this report. |
| Lines 63-65 | Size-specific pretraining/sequence-length table | Approximately 1k is consistent with 1024 for supplied NT/Genomic pretraining. Supplied task-specific 6.8M/7.4M BEND/DeepSEA configurations are absent in both the relevant original release and the intentionally limited current paper family; those rows cannot be verified as runnable configurations here. |
| Line 71 | NT 1k / hidden128 / five GCBs / 1.7M / inner2 | Paper architecture and rounded downstream size agree. Effective downstream lengths are not uniformly 1k. |
| Line 72 | Genomic hidden128 / five GCBs / 378k / inner4 | Hidden width is 64 in both supplied original and current paper config. Five blocks, inner4 and approximately 378k downstream count agree. |
| Line 73 | BEND 512/16k and larger H196/10/7.4M/inner2 | Current task lengths agree. Current supplied BEND experiments are intentionally small H64/5; larger configuration is not present, so its exact runnable setting/count is not verified. |
| Line 74 | Chromatin 1k and larger H256/5/6.8M/inner2 | Current array length 1000 agrees approximately with 1k. Current experiments are intentionally small; larger task-specific configuration is not supplied. |
| Line 80 | HPO H64/four GCBs/K5/inner2/262k/bf16/seed2222 | Width/kernel/expansion/precision/seed agree. Actual block count is five. Approximately 262k is a downstream-backbone/binary-head count, not universal total or pretraining count. |
| Line 88 | Learning Rate = beta1/beta2 | Incorrect table field: these are optimizer betas. Actual configured learning rate is 0.001. |
| Line 89 | Pretraining batch256, same as original | Incorrect as effective batch: original 768, fork 512. Paper NT local batch256 accumulates twice. |
| Line 90 | Original pretraining weight decay0.1 | Incorrect for supplied original experiment, which explicitly sets0.0. Fork uses0.1. |
| Line 91 | Dropout0.0 | Verified unchanged. |
| Line 92 | AdamW | Verified unchanged. |
| Line 93 | Betas0.9/0.999 | Verified unchanged. |
| Line 94 | Cosine decay | Selected scheduler family agrees; resolved horizons differ because counts/batches/epochs differ. |
| Line 95 | Original pretraining400 epochs | Accepted by explicit user instruction. Current paper variants restore400; small variants use200. |
| Line 96 | Dilation rate4 | Configured base4 agrees. Actual existing helper schedule is not a simple power-of-four increment from the first block. |
| Line 97 | fp32 | Correct for original pretraining and current paper pretraining; small runs use documented bf16. Original Genomic downstream fp16 is a separate difference. |
| Line 98 | Kernel9 | Correct for original pretraining/NT and current paper NT GCBs; small and paper Genomic GCBs use5. Both stems retain9. |
| Line 102 | Tasks omitted from canonicalization still evaluated | Supported by exclusions plus launcher registries, including an additional code-level regulatory promoter-class exclusion. |

The document says its contents were extracted verbatim from `Paper/paper.tex`. This audit checks the claims against code/configuration; it did not independently verify that the extraction remains verbatim or review unrelated prose in the paper.

## 14. Config-tree and source differences with no established effective run change

The following differences are retained so they are not confused with omitted experiment findings:

1. **Experiment reorganization.** Current ConvNova experiments are reorganized into baseline/HPO/paper families. Original Hyena, LegNet, Basenji, transformer, Mamba and ConvNext experiment YAMLs are absent from the current experiment tree. They are not ancestors of the 27 current ConvNova experiments and were not treated as alternative original baselines.
2. **Shared base replacement.** Original `experiment/base.yaml` was illustrative and referenced unrelated groups. The fork replaces it with the active common ConvNova settings. Comparisons were made to the actual original ConvNova experiments, not to that illustrative base.
3. **Old NT compatibility config/registry.** These exist but are not selected by the current NT family. Their existence does not override current revised metadata.
4. **Generic HG38 paths.** `dataset/hg38.yaml` gains explicit project-root paths. The nine audited pretraining experiments select `bert_hg38`, so this generic-group change is not independently counted as their effective path change; their own selected group changes are recorded in PRE-05.
5. **Optimizer Adam file.** `optimizer/adam.yaml` differs only by its final newline. The selected optimizer is AdamW, whose betas are unchanged. No effective optimizer-momentum change follows.
6. **Genomic pipeline.** Its paired diff changes a comment, not the pooled-head selection. Effective behavior differences come from data, masks, architecture and shared runtime.
7. **NT pipeline.** It is unchanged. New NT behavior is not attributed to a nonexistent pipeline-head replacement.
8. **Optional import guard.** The RMSNorm import change is real but does not change the configured ConvNova architecture.
9. **Duplicate package tree.** The fork contains a substantial new `src/convnova/*` tree, including older implementations. The active training registry and launcher import `src.*`; duplicate files were not independently counted as active experiment changes merely because they appear as additions in a recursive diff.
10. **Duplicate HPO runner.** `src/hpo/hpo_runner.py` exists, but the actual benchmark command selects `src.utils.hpo_runner`.
11. **Bytecode and generated artifacts.** Existing `__pycache__` files appear in an ordinary filesystem diff. They were not treated as authored source changes, and saved run/output artifacts were outside scope.
12. **Stale comments.** Baseline defaults comments refer to `/experiment/hpo/hpo_base`, but actual defaults select `/experiment/base`. No hidden `hpo_base` dependency was inferred.
13. **Shared placeholders overridden by descendants.** The shared base's pretraining-specific scheduler count is overridden by family-local downstream expressions. The active BEND/DeepSEA count errors are the family expressions documented above, not the overridden placeholder.

## 15. Open questions requiring the owner's intent or external provenance

These questions are intentionally left open rather than answered through assumptions:

1. **Original-bug corrections:** Were single-pass MLM corruption, explicit pad exclusion, repeated backbone pad masking and epoch-level MCC intentional methodological corrections? If so, should they be disclosed as changes shared by all three model variants?
2. **Batch/regularization:** Were pretraining effective batch512 and weight decay0.1 intentional departures from supplied original batch768/weight-decay0? Was Genomic downstream effective batch256 intentionally doubled from128?
3. **Strand contract:** Is the intended canonical orientation coding/query-positive, as the downstream comments state, or does the same-strand transformation implement the intended opposite convention? Which should be corrected: code or description?
4. **Raw-signal invariant:** Should signs be relative to the reference genome or to the input sequence after reverse-strand alignment? Should overlaps preserve both-strand information instead of negative overwriting positive?
5. **CpG wording:** Does “linear head over the entire encoded sequence” mean pooled mean followed by linear, or a position-specific flattened linear layer?
6. **Evaluation split:** Is validation-only reporting intentional for BEND/DeepSEA despite separate test arrays/H5 splits? Which split should the standalone DeepSEA evaluator use?
7. **Parameter-count convention:** Should tables report downstream backbone, full downstream model, or full pretraining model? Should control-channel increments and the 919-output head be shown?
8. **Declared NT counts:** Do the revised dataset cardinalities match the configured promoter/histone counts? This requires dataset metadata, outside this config-only audit.
9. **Training-genome equivalence:** Are baseline `hg38.ml.fa` and the primary-assembly FASTA used to generate canonicalized data equivalent in coordinates, contigs and relevant base/masking content?
10. **Historical original overrides:** Is there evidence for original1k Genomic runs or another original378k configuration revision? Without that evidence, the supplied original repo remains authoritative as instructed.
11. **Legacy scripts:** Are `benchmarks.sh`, `run_all_tasks_caller.py`, and `fix_hpo.py` archival, or should they be maintained as supported execution paths?
12. **Search dimensions:** Is an explicit end-RC hyperparameter still intended, and should `keep_segments_order` be searched at its active top-level key? The checked-in HPO interface currently does not implement those intentions successfully.

No answer to these questions was assumed when classifying an undocumented difference as intentional or unwanted.

## 16. Final coverage and verification record

| Audit item | Disposition |
|---|---|
| `AGENTS.md` read | Complete |
| `modifications.md` checked claim by claim | Complete, with original400-epoch exception applied |
| All nine pretraining experiment YAMLs | Covered in Sections3-5 |
| All six NT downstream experiment YAMLs | Covered in Sections3,4,6,10-12 |
| All six Genomic downstream experiment YAMLs | Covered in Sections3,4,7,10-12 |
| All three BEND downstream experiment YAMLs | Covered in Sections3,4,8,10-12 |
| All three chromatin downstream experiment YAMLs | Covered in Sections3,4,9-12 |
| Selected inherited configuration groups | Traced statically, with effective-versus-overridden values distinguished |
| Relevant active `src/` differences | Diff-driven review complete for the agreed scope |
| Changed training entrypoint and current launchers | Reviewed, including override and checkpoint routing |
| Exact model parameter counts | Analytical, stage/head/control conventions stated |
| Batch/scheduler arithmetic | Dependency-free calculation; actual dataset step counts not measured |
| Best-checkpoint state transformation | Isolated logic verification; no full Lightning integration run |
| Hydra/OmegaConf full composition | Not executed: dependencies unavailable in audit interpreter |
| PyTorch model instantiation/project tests | Not executed: dependencies unavailable; no training launched |
| Historical saved runs/checkpoint contents/W&B | Excluded by requested scope |
| Actual input-file cardinality/base/alignment checks | Not performed; conditional risks explicitly marked |
| Intentional absence of large BEND/DeepSEA configs | Accepted; not classified as a defect |
| Experiment/source fixes | None made by this audit |

The report distinguishes source-established disagreements from possible historical behavior. Correct intended canonicalization comparisons can coexist with substantial undocumented deviations from original ConvNova. The configuration and source evidence above identifies those deviations without asserting that any particular completed run used the defaults or encountered every listed conditional defect.
