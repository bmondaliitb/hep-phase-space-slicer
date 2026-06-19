from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FeatureData:
    path: Path
    values: np.ndarray
    names: list[str]
    paths: list[Path] | None = None


def load_feature_data(path: str | Path) -> FeatureData:
    """Load feature data from .npy or .npz into a 2D float array."""
    source = Path(path).expanduser().resolve()
    loaded = np.load(source, allow_pickle=True)

    if isinstance(loaded, np.lib.npyio.NpzFile):
        values, names = _load_npz(loaded)
    else:
        values, names = _coerce_array(loaded)

    if values.ndim != 2:
        raise ValueError(f"Expected a 2D feature table, got shape {values.shape}.")
    if values.shape[1] < 2:
        raise ValueError("At least two feature columns are required.")

    return FeatureData(
        path=source,
        values=np.asarray(values, dtype=np.float64),
        names=names,
        paths=[source],
    )


def load_feature_files(paths: list[str | Path]) -> FeatureData:
    """Load and concatenate one or more feature files row-wise."""
    if not paths:
        raise ValueError("At least one feature file is required.")

    datasets = [load_feature_data(path) for path in paths]
    column_count = datasets[0].values.shape[1]
    for dataset in datasets:
        if dataset.values.shape[1] != column_count:
            raise ValueError(
                "All files must have the same number of columns to be loaded together. "
                f"{dataset.path} has {dataset.values.shape[1]}, expected {column_count}."
            )

    values = np.vstack([dataset.values for dataset in datasets])
    names = list(datasets[0].names)
    source = datasets[0].path if len(datasets) == 1 else datasets[0].path.parent / "multiple_files"
    return FeatureData(
        path=source,
        values=np.asarray(values, dtype=np.float64),
        names=names,
        paths=[dataset.path for dataset in datasets],
    )


def _load_npz(loaded: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[str]]:
    arrays = {name: loaded[name] for name in loaded.files}

    if "values" in arrays:
        values, names = _coerce_array(arrays["values"])
        if "names" in arrays:
            names = [str(item) for item in arrays["names"].tolist()]
        return values, names

    candidates = [
        (name, arr)
        for name, arr in arrays.items()
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and np.issubdtype(arr.dtype, np.number)
    ]
    if not candidates:
        raise ValueError("No numeric 2D array was found in the .npz file.")

    _, array = max(candidates, key=lambda item: item[1].shape[0] * item[1].shape[1])
    values, names = _coerce_array(array)
    if "names" in arrays and len(arrays["names"]) == values.shape[1]:
        names = [str(item) for item in arrays["names"].tolist()]
    return values, names


def _coerce_array(array: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if array.dtype.names:
        numeric_names = [
            name for name in array.dtype.names if np.issubdtype(array.dtype[name], np.number)
        ]
        if not numeric_names:
            raise ValueError("Structured array does not contain numeric fields.")
        values = np.column_stack([array[name] for name in numeric_names])
        return values, numeric_names

    if array.dtype == object and array.shape == ():
        item = array.item()
        if isinstance(item, dict):
            return _coerce_mapping(item)

    values = np.asarray(array)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"Expected numeric data, got dtype {values.dtype}.")
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D numeric array, got shape {values.shape}.")
    names = [f"feature_{index:02d}" for index in range(values.shape[1])]
    return values, names


def _coerce_mapping(mapping: dict) -> tuple[np.ndarray, list[str]]:
    if "values" in mapping:
        values = np.asarray(mapping["values"])
        names = mapping.get("names")
        if names is None:
            names = [f"feature_{index:02d}" for index in range(values.shape[1])]
        return values, [str(name) for name in names]

    columns = {
        str(name): np.asarray(value)
        for name, value in mapping.items()
        if np.asarray(value).ndim == 1 and np.issubdtype(np.asarray(value).dtype, np.number)
    }
    if not columns:
        raise ValueError("Object array dictionary does not contain numeric columns.")
    length = len(next(iter(columns.values())))
    if any(len(column) != length for column in columns.values()):
        raise ValueError("Dictionary columns have inconsistent lengths.")
    names = list(columns)
    values = np.column_stack([columns[name] for name in names])
    return values, names
