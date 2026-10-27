def get_bend_dataset_columns(dataset_name):
    if dataset_name == "gene_finding":
        bed_columns = ("chromosome", "start", "end", "transcript_id", "strand", "flank_left", "flank_right", "length", "split")
        to_convert_to_int = ("start", "end", "length")
    elif dataset_name == "cpg_methylation":
        bed_columns = ("chromosome", "start", "end", "strand", "label", "split")
        to_convert_to_int = ("start", "end")
    else:
        raise ValueError(f"Dataset name {dataset_name} not recognized")
    return bed_columns, to_convert_to_int