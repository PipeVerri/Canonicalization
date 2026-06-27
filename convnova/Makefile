DATA := data

include .env
#export

.PHONY: all pretrain bend deepsea gencode gencode-human gencode-mouse ncbi index \
        index-hg38 index-mouse index-drosophila index-roundworm

all: pretrain bend deepsea gencode ncbi index

# ─── ConvNova / hg38 training data ──────────────────────────────────────────

pretrain: \
	$(DATA)/hg38/hg38.ml.fa \
	$(DATA)/hg38/human-sequences.bed

$(DATA)/hg38/hg38.ml.fa: | $(DATA)/hg38
	curl -L https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz \
		-o $@.gz
	gunzip -c $@.gz > $@

$(DATA)/hg38/human-sequences.bed: | $(DATA)/hg38
	curl -L https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed \
		-o $@

# ─── BEND benchmark ─────────────────────────────────────────────────────────

BEND_URL := https://sid.erda.dk/share_redirect/aNQa0Oz2lY/data
DEEPSEA_URL := https://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz

bend: \
	$(DATA)/bend_benchmark/gene_finding.bed \
	$(DATA)/bend_benchmark/gene_finding.hdf5 \
	$(DATA)/bend_benchmark/cpg_methylation.bed \
	$(DATA)/bend_benchmark/variant_effects_disease.bed \
	$(DATA)/bend_benchmark/variant_effects_expression.bed

$(DATA)/bend_benchmark/gene_finding.bed: | $(DATA)/bend_benchmark
	curl -L $(BEND_URL)/gene_finding/gene_finding.bed -o $@

$(DATA)/bend_benchmark/gene_finding.hdf5: | $(DATA)/bend_benchmark
	curl -L $(BEND_URL)/gene_finding/gene_finding.hdf5 -o $@

$(DATA)/bend_benchmark/cpg_methylation.bed: | $(DATA)/bend_benchmark
	curl -L $(BEND_URL)/cpg_methylation/cpg_methylation.bed -o $@

$(DATA)/bend_benchmark/variant_effects_disease.bed: | $(DATA)/bend_benchmark
	curl -L $(BEND_URL)/variant_effects/variant_effects_disease.bed -o $@

$(DATA)/bend_benchmark/variant_effects_expression.bed: | $(DATA)/bend_benchmark
	curl -L $(BEND_URL)/variant_effects/variant_effects_expression.bed -o $@

# ─── DeepSEA data download ─────────────────────────────────────────────────

deepsea: $(DATA)/deepsea/train.mat

$(DATA)/deepsea/train.mat: | $(DATA)/deepsea
	curl -L $(DEEPSEA_URL) -o $(DATA)/deepsea/deepsea_train_bundle.v0.9.tar.gz
	tar -xzf $(DATA)/deepsea/deepsea_train_bundle.v0.9.tar.gz -C $(DATA)/deepsea

# ─── GENCODE annotations + genomes ──────────────────────────────────────────

GENCODE_H := https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_50
GENCODE_M := https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M39

gencode: gencode-human gencode-mouse

gencode-human: \
	$(DATA)/hg38/gencode.v50.primary_assembly.annotation.gff3 \
	$(DATA)/hg38/GRCh38.primary_assembly.genome.fa

gencode-mouse: \
	$(DATA)/mouse/gencode.vM39.primary_assembly.annotation.gff3 \
	$(DATA)/mouse/GRCm39.primary_assembly.genome.fa

$(DATA)/hg38/gencode.v50.primary_assembly.annotation.gff3: | $(DATA)/hg38
	curl -L $(GENCODE_H)/gencode.v50.primary_assembly.annotation.gff3.gz \
		-o $@.gz
	gunzip -c $@.gz > $@

$(DATA)/hg38/GRCh38.primary_assembly.genome.fa: | $(DATA)/hg38
	curl -L $(GENCODE_H)/GRCh38.primary_assembly.genome.fa.gz \
		-o $@.gz
	gunzip -c $@.gz > $@

$(DATA)/mouse/gencode.vM39.primary_assembly.annotation.gff3: | $(DATA)/mouse
	curl -L $(GENCODE_M)/gencode.vM39.primary_assembly.annotation.gff3.gz \
		-o $@.gz
	gunzip -c $@.gz > $@

$(DATA)/mouse/GRCm39.primary_assembly.genome.fa: | $(DATA)/mouse
	curl -L $(GENCODE_M)/GRCm39.primary_assembly.genome.fa.gz \
		-o $@.gz
	gunzip -c $@.gz > $@

# ─── NCBI datasets ──────────────────────────────────────────────────────────

ncbi: \
	$(DATA)/hg38/ncbi_dataset/data/GCF_000001405.40/genomic.gff \
	$(DATA)/drosophila/ncbi_dataset/data/GCF_000001215.4/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna \
	$(DATA)/mouse/ncbi_dataset/data/GCF_000001635.27/genomic.gff \
	$(DATA)/roundworm/ncbi_dataset/data/GCF_000002985.6/GCF_000002985.6_WBcel235_genomic.fna

$(DATA)/hg38/ncbi_dataset/data/GCF_000001405.40/genomic.gff: | $(DATA)/hg38
	datasets download genome accession GCF_000001405.40 \
		--include gff3,seq-report \
		--filename $(DATA)/hg38/ncbi.zip
	unzip -n $(DATA)/hg38/ncbi.zip -d $(DATA)/hg38/

$(DATA)/drosophila/ncbi_dataset/data/GCF_000001215.4/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna: | $(DATA)/drosophila
	datasets download genome accession GCF_000001215.4 \
		--include gff3,genome,seq-report \
		--filename $(DATA)/drosophila.zip
	unzip -n $(DATA)/drosophila.zip -d $(DATA)/drosophila/

$(DATA)/mouse/ncbi_dataset/data/GCF_000001635.27/genomic.gff: | $(DATA)/mouse
	datasets download genome accession GCF_000001635.27 \
		--include gff3,seq-report \
		--filename $(DATA)/mouse/ncbi.zip
	unzip -n $(DATA)/mouse/ncbi.zip -d $(DATA)/mouse/

$(DATA)/roundworm/ncbi_dataset/data/GCF_000002985.6/GCF_000002985.6_WBcel235_genomic.fna: | $(DATA)/roundworm
	datasets download genome accession GCF_000002985.6 \
		--include gff3,genome,seq-report \
		--filename $(DATA)/roundworm.zip
	unzip -n $(DATA)/roundworm.zip -d $(DATA)/roundworm/

# ─── Indexing ───────────────────────────────────────────────────────────────

index: index-hg38 index-mouse index-drosophila index-roundworm

index-hg38: $(DATA)/hg38_index.mmi

$(DATA)/hg38_index.mmi: $(DATA)/hg38/GRCh38.primary_assembly.genome.fa
	minimap2 -x sr -d $@ $<

index-mouse: $(DATA)/mouse_index.mmi

$(DATA)/mouse_index.mmi: $(DATA)/mouse/GRCm39.primary_assembly.genome.fa
	minimap2 -x sr -d $@ $<

index-drosophila: $(DATA)/drosophila_index.mmi

$(DATA)/drosophila_index.mmi: $(DATA)/drosophila/ncbi_dataset/data/GCF_000001215.4/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna
	minimap2 -x sr -d $@ $<

index-roundworm: $(DATA)/roundworm_index.mmi

$(DATA)/roundworm_index.mmi: $(DATA)/roundworm/ncbi_dataset/data/GCF_000002985.6/GCF_000002985.6_WBcel235_genomic.fna
	minimap2 -x sr -d $@ $<

# ─── Directory creation ──────────────────────────────────────────────────────

$(DATA)/hg38 $(DATA)/mouse $(DATA)/bend_benchmark $(DATA)/drosophila $(DATA)/roundworm $(DATA)/deepsea:
	mkdir -p $@