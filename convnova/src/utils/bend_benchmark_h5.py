from pathlib import Path

import h5py
import numpy as np
import pandas as pd

BEND_H5_FILENAME = "bend_benchmark.h5"
BEND_SPLITS = ("train", "valid", "test")


def _read_split_group(group):
    data = {}
    for name, dataset in group.items():
        if h5py.check_string_dtype(dataset.dtype) is not None:
            data[name] = dataset.asstr()[:]
        elif h5py.check_vlen_dtype(dataset.dtype) is not None:
            values = dataset[:]
            arrays = np.empty(len(values), dtype=object)
            for index, value in enumerate(values):
                arrays[index] = np.asarray(value)
            data[name] = arrays
        else:
            data[name] = dataset[:]
    return data


def read_bend_split(h5_path, task_name, split):
    """Read one task/split into NumPy arrays without retaining an open H5 handle."""
    split = "valid" if split == "val" else split
    with h5py.File(Path(h5_path), "r") as h5_file:
        return _read_split_group(h5_file[task_name][split])


def read_bend_task(h5_path, task_name):
    """Read all available splits of a task into a DataFrame for canonicalization."""
    frames = []
    with h5py.File(Path(h5_path), "r") as h5_file:
        task_group = h5_file[task_name]
        for split in BEND_SPLITS:
            if split not in task_group:
                continue
            frame = pd.DataFrame(_read_split_group(task_group[split]))
            frame["split"] = split
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _write_split_group(parent, frame):
    for column in frame.columns:
        values = frame[column]
        if column == "label":
            labels = parent.create_dataset(
                "label",
                shape=(len(frame),),
                dtype=h5py.vlen_dtype(np.dtype("int16")),
            )
            for index, value in enumerate(values):
                labels[index] = np.asarray(value, dtype=np.int16).reshape(-1)
        elif pd.api.types.is_numeric_dtype(values.dtype):
            parent.create_dataset(column, data=values.to_numpy())
        else:
            parent.create_dataset(
                column,
                data=values.astype(str).to_numpy(dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )


def write_bend_h5(h5_path, task_frames, mode="a"):
    """Write task DataFrames, replacing whole task groups but preserving other tasks."""
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, mode) as h5_file:
        for task_name, frame in task_frames.items():
            required = {"sequence", "label", "split"}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{task_name} is missing required columns: {sorted(missing)}")
            if task_name in h5_file:
                del h5_file[task_name]
            task_group = h5_file.create_group(task_name)
            for split in BEND_SPLITS:
                split_frame = frame.loc[frame["split"] == split].drop(columns="split")
                if split_frame.empty:
                    continue
                split_group = task_group.create_group(split)
                _write_split_group(split_group, split_frame)
