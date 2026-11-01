# SPDX-FileCopyrightText: 2026 Yonsei University Health System
# SPDX-License-Identifier: Apache-2.0
"""The data table: reading it, the column map, which rows are doses and observations, and
row conditions (``where DVID == 1``)."""

from __future__ import annotations

import ast
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import nearest

# NONMEM's standard columns (the column map renames the data's own names to these)
STANDARD = ("ID", "TIME", "DV", "AMT", "EVID", "MDV", "CMT", "DVID", "RATE", "II", "ADDL", "SS")


def read_table(source, base: Path | None = None) -> tuple[pd.DataFrame, str, str | None]:
    """A table from a CSV file (comma, tab or space separated; ``.`` is a missing value, as in
    NONMEM) or a pandas DataFrame (copied), with the SHA-256 of what was read (the file's
    bytes, or the table's contents) and the file's full path (None for a DataFrame). Spaces
    around column names and values are dropped; rows are counted from 1 after the header."""
    if isinstance(source, pd.DataFrame):
        df = source.copy().reset_index(drop=True)
        df.columns = [str(c).strip() for c in df.columns]
        h = hashlib.sha256(",".join(map(str, df.columns)).encode())
        h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
        return df, h.hexdigest(), None
    if not isinstance(source, str | Path):
        raise TypeError(
            f"data: give a file name or a pandas DataFrame, not {type(source).__name__}."
        )
    path = Path(source)
    if base is not None and not path.is_absolute():
        path = base / path
    if not path.exists():
        raise ValueError(f"data: no file '{path}'.")
    raw = path.read_bytes()
    df = pd.read_csv(
        io.BytesIO(raw), sep=None, engine="python", na_values=["."], skipinitialspace=True
    )
    df.columns = [str(c).strip() for c in df.columns]
    return df, hashlib.sha256(raw).hexdigest(), str(path.resolve())


def apply_map(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename the data's own columns to the standard names (``ID="Subject"``)."""
    for std, col in mapping.items():
        if std not in STANDARD:
            raise ValueError(
                f"data: '{std}' is not a standard column.{nearest(std, STANDARD)} Standard "
                f"columns: {', '.join(STANDARD)}; other columns (covariates) are kept as named."
            )
        if col not in df.columns:
            raise ValueError(
                f"data: no column '{col}' (mapped to {std}).{nearest(col, map(str, df.columns))} "
                f"The columns are: {', '.join(map(str, df.columns))}."
            )
        if std in df.columns and std != col:
            raise ValueError(
                f"data: {std}={col}, but the data already have a column {std}; map one of them."
            )
    return df.rename(columns={col: std for std, col in mapping.items()})


def observation_rows(df: pd.DataFrame) -> np.ndarray:
    """Rows holding a measurement: EVID 0 when there is an EVID column; otherwise rows with a
    DV value that are not marked MDV = 1 and carry no dose amount."""
    if "EVID" in df.columns:
        return (df["EVID"] == 0).to_numpy()
    if "DV" not in df.columns:
        return np.zeros(len(df), bool)
    rows = df["DV"].notna()
    if "MDV" in df.columns:
        rows &= df["MDV"].fillna(0) == 0
    if "AMT" in df.columns:
        rows &= df["AMT"].fillna(0) == 0
    return rows.to_numpy()


def dose_rows(df: pd.DataFrame) -> np.ndarray:
    """Rows giving a dose: EVID 1 or 4 when there is an EVID column; otherwise a non-zero AMT."""
    if "EVID" in df.columns:
        return df["EVID"].isin([1, 4]).to_numpy()
    if "AMT" in df.columns:
        return (df["AMT"].fillna(0) != 0).to_numpy()
    return np.zeros(len(df), bool)


# ----------------------------------------------------------------------------- conditions
_CMP = {
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
}


def check_condition(text: str) -> str:
    """A row condition (``DVID == 1``, ``CMT == 2 and TIME > 0``), checked for its form."""
    text = text.strip()
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError:
        raise ValueError(
            f"cannot read the condition '{text}'; write it as DVID == 1, or join two with "
            "'and' / 'or'."
        ) from None
    _walk(tree, None, text)
    return text


def condition_columns(text: str) -> set[str]:
    return {n.id for n in ast.walk(ast.parse(text, mode="eval")) if isinstance(n, ast.Name)}


def evaluate_condition(text: str, df: pd.DataFrame) -> np.ndarray:
    """Which rows of ``df`` meet the condition."""
    return np.asarray(_walk(ast.parse(text, mode="eval").body, df, text), bool)


def _walk(node, df, text):
    if isinstance(node, ast.BoolOp):
        parts = [_walk(v, df, text) for v in node.values]
        if df is None:
            return None
        f = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        out = parts[0]
        for p in parts[1:]:
            out = f(out, p)
        return out
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        v = _walk(node.operand, df, text)
        return None if df is None else np.logical_not(v)
    if isinstance(node, ast.Compare):
        left, out = _value(node.left, df, text), None
        for op, right in zip(node.ops, node.comparators, strict=True):
            if type(op) not in _CMP:
                raise ValueError(f"in '{text}': compare with ==, !=, <, <=, > or >=.")
            r = _value(right, df, text)
            if df is not None:
                try:
                    res = _CMP[type(op)](left, r)
                except TypeError:
                    raise ValueError(
                        f"in '{text}': a column holds text, or text and numbers; compare it "
                        'with a quoted text (DV == "BLQ").'
                    ) from None
                out = res if out is None else out & res
            left = r
        return out
    raise ValueError(
        f"cannot read the condition '{text}'; write it as DVID == 1, or join two with 'and' / 'or'."
    )


def _value(node, df, text):
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float | str):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
    ):
        return -node.operand.value
    if isinstance(node, ast.Name):
        if df is None:
            return None
        if node.id not in df.columns:
            values = {str(v) for c in ("DVID", "CMT") if c in df.columns for v in df[c].unique()}
            hint = (
                f' If {node.id} is a value, quote it: "{node.id}".'
                if node.id in values
                else nearest(node.id, map(str, df.columns))
            )
            raise ValueError(f"in '{text}': the data have no column '{node.id}'.{hint}")
        return df[node.id].to_numpy()
    raise ValueError(f"in '{text}': compare a column with a number or a quoted text.")


def value_text(v) -> str:
    """A data value as it is written in a condition: 1, 2.5, "bact"."""
    if isinstance(v, str):
        return f'"{v}"'
    v = float(v)
    return str(int(v)) if v.is_integer() else repr(v)
