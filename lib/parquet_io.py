"""
parquet_io.py — Read and write Parquet batch files for the AIM4ML pipeline.

Each Parquet batch is a table with one row per molecule.  The molecule
itself is stored as an SDF V2000 text block in the `mol_block` column;
all other columns are metadata extracted from the input SDF tags.

Public functions:
    write_batch(path, rows)   → write a list of dicts to a Parquet file
    read_batch(path)            → read a Parquet file → list of dicts
"""

import os

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from .schema import PARQUET_COLUMNS, PARQUET_REQUIRED, PARQUET_OPTIONAL

# Column order for the Parquet file.
_COL_ORDER = list(PARQUET_COLUMNS.keys())


def _make_table(rows):
    """Convert a list of dicts into a PyArrow Table with a dynamic schema.

    Base columns come from PARQUET_COLUMNS.  Any additional keys present
    in the rows are included with type inferred from the first non-None value.
    """
    if not rows:
        arrays = {col: pa.array([], type=pa.from_numpy_dtype(dtype)
                                if dtype != "string" else pa.string())
                  for col, dtype in PARQUET_COLUMNS.items()}
        return pa.table(arrays)

    # Discover extra columns (not in base schema)
    known_cols = set(PARQUET_COLUMNS.keys())
    extra_cols = []
    for row in rows:
        for key in row:
            if key not in known_cols and key not in extra_cols:
                extra_cols.append(key)

    # Full column order: base first, then extras
    full_order = list(PARQUET_COLUMNS.keys()) + extra_cols

    # Build column arrays
    data = {}
    for col in full_order:
        values = [row.get(col) for row in rows]
        if col in PARQUET_COLUMNS:
            dtype = PARQUET_COLUMNS[col]
            required = col in PARQUET_REQUIRED
        else:
            # Infer dtype: if any value is a string, treat as string.
            # Otherwise use first non-None value.
            has_string = any(isinstance(v, str) for v in values if v is not None)
            if has_string:
                dtype, required = "string", False
            else:
                first = next((v for v in values if v is not None), None)
                if isinstance(first, bool):
                    dtype, required = "bool", False
                elif isinstance(first, int):
                    dtype, required = "int32", False
                elif isinstance(first, float):
                    dtype, required = "float64", False
                else:
                    dtype, required = "string", False

        if dtype == "string":
            data[col] = pa.array([str(v) if v is not None else None
                                  for v in values], type=pa.string())
        elif dtype == "bool":
            data[col] = pa.array([bool(v) if v is not None else None
                                  for v in values], type=pa.bool_())
        elif dtype.startswith("float"):
            data[col] = pa.array([float(v) if v is not None else None
                                  for v in values], type=pa.float64())
        elif dtype.startswith("int"):
            data[col] = pa.array([int(v) if v is not None else None
                                  for v in values], type=pa.int32())
        else:
            data[col] = pa.array([str(v) if v is not None else None
                                  for v in values], type=pa.string())

    return pa.table(data)


def write_batch(path, rows):
    """Write a batch of molecule rows to a Parquet file.

    Parameters
    ----------
    path : str
        Path to the output .parquet file.
    rows : list of dict
        Each dict has keys matching PARQUET_COLUMNS.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    table = _make_table(rows)
    pq.write_table(table, path, compression="zstd")


def read_batch(path):
    """Read a Parquet batch file and return a list of dict rows.

    Parameters
    ----------
    path : str
        Path to the .parquet file.

    Returns
    -------
    list of dict
        Each dict has keys matching PARQUET_COLUMNS.
    """
    table = pq.read_table(path)
    df = table.to_pandas()
    # Ensure correct column order and fill missing optional columns
    for col in PARQUET_OPTIONAL:
        if col not in df.columns:
            df[col] = None
    # Base columns first, then any extra columns
    base_cols = list(PARQUET_COLUMNS.keys())
    extra_cols = [c for c in df.columns if c not in base_cols]
    return df[base_cols + extra_cols].to_dict(orient="records")
