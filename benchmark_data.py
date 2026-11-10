from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_csv_file(
    path: Path,
    *,
    target_column: str | None = None,
    has_header: bool | None = None,
    include_target: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load and preprocess a local CSV dataset."""
    content = path.read_text(encoding="utf-8-sig")
    return _parse_csv_content(
        content,
        target_column=target_column,
        has_header=has_header,
        include_target=include_target,
    )


def _parse_csv_content(
    content: str,
    *,
    target_column: str | None,
    has_header: bool | None,
    include_target: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        dialect = csv.Sniffer().sniff(content[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [
        [value.strip() for value in row]
        for row in csv.reader(io.StringIO(content), dialect=dialect)
        if any(value.strip() for value in row)
    ]
    if not rows or len(rows[0]) < 2:
        raise ValueError("CSV file is not a supported tabular dataset")

    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows):
        raise ValueError("Dataset rows must all contain the same number of columns")
    if not include_target and target_column is not None:
        raise ValueError("target_column cannot be used when include_target is False")
    label_index = (
        _resolve_target_column(target_column, rows[0], column_count)
        if include_target
        else None
    )

    if has_header is None:
        has_header = any(
            index != label_index
            and not _is_missing(value)
            and _as_float(value) is None
            for index, value in enumerate(rows[0])
        )

    column_names = rows[0] if has_header else [
        f"column_{index}" for index in range(column_count)
    ]
    if has_header:
        rows = rows[1:]

    if not rows:
        raise ValueError("CSV does not contain any data rows")
    if len(rows) < 2:
        raise ValueError("CSV must contain at least two data rows")
    labels = None
    if label_index is not None:
        labels = [row[label_index].strip() for row in rows]
        if any(not label or _is_missing(label) for label in labels):
            raise ValueError("Target column contains missing labels")
        if len(set(labels)) < 2:
            raise ValueError("Target column must contain at least two distinct labels")

    feature_indices = [
        index
        for index in range(column_count)
        if index != label_index and not _is_identifier_column(column_names[index])
    ]
    if not feature_indices:
        raise ValueError("No feature columns remain after excluding the target and IDs")

    raw_features = np.asarray(
        [[row[index].strip() for index in feature_indices] for row in rows],
        dtype=object,
    )
    missing_values = np.asarray(
        [[_is_missing(value) for value in row] for row in raw_features],
        dtype=bool,
    )
    raw_features[missing_values] = np.nan

    numeric_columns = []
    categorical_columns = []
    for feature_index in range(raw_features.shape[1]):
        values = raw_features[:, feature_index]
        observed = values[
            ~np.asarray([_is_missing(value) for value in values], dtype=bool)
        ]
        converted = [_as_float(value) for value in observed]
        if observed.size and all(value is not None for value in converted):
            if not np.isfinite(np.asarray(converted, dtype=np.float64)).all():
                raise ValueError(
                    f"Feature column {column_names[feature_indices[feature_index]]!r} "
                    "contains non-finite numeric values"
                )
            numeric_columns.append(feature_index)
            raw_features[:, feature_index] = [
                np.nan if _is_missing(value) else float(value)
                for value in values
            ]
        else:
            if observed.size == 0:
                raise ValueError(
                    f"Feature column {column_names[feature_indices[feature_index]]!r} "
                    "contains no non-missing values"
                )
            categorical_columns.append(feature_index)

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median")),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                categorical_columns,
            )
        )

    features = ColumnTransformer(transformers, remainder="drop").fit_transform(
        raw_features
    )
    return (
        MinMaxScaler().fit_transform(features),
        np.asarray(labels) if labels is not None else None,
    )


_MISSING_VALUES = {"", "?", "na", "n/a", "null", "none", "nan"}
_IDENTIFIER_NAMES = {"id", "uuid", "index", "rowid", "recordid"}


def _is_missing(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in _MISSING_VALUES
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def _as_float(value: str) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _resolve_target_column(
    target_column: str | None,
    column_names: list[str],
    column_count: int,
) -> int:
    if target_column is None:
        return column_count - 1

    target = target_column.strip()
    if not target:
        raise ValueError("Target column cannot be empty")
    try:
        index = int(target)
    except ValueError:
        matches = [
            index
            for index, name in enumerate(column_names)
            if name.casefold() == target.casefold()
        ]
        if len(matches) != 1:
            raise ValueError(f"Target column {target!r} was not found uniquely")
        return matches[0]

    if index < 0:
        index += column_count
    if not 0 <= index < column_count:
        raise ValueError(f"Target column index {target!r} is out of range")
    return index


def _is_identifier_column(column_name: str) -> bool:
    separated_camel_case = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", column_name)
    words = re.findall(r"[a-z0-9]+", separated_camel_case.casefold())
    normalized = "".join(words)
    return (
        not normalized
        or normalized.startswith("unnamed")
        or any(word in _IDENTIFIER_NAMES for word in words)
    )
