from pathlib import Path

import numpy as np


def load_psi_matrix(
    psi_path: str | Path,
    reads_path: str | Path | None = None,
    min_reads: int = 8,
    delimiter: str = "\t",
) -> tuple[list[str], np.ndarray]:
    event_ids: list[str] = []
    rows: list[list[float]] = []
    with open(psi_path, encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            return [], np.zeros((0, 0), dtype=np.float32)
        for line in handle:
            fields = line.rstrip("\n").split(delimiter)
            if len(fields) < 2:
                continue
            event_ids.append(fields[0])
            values = [_parse_float(value) for value in fields[1:]]
            rows.append(values)
    psi = np.asarray(rows, dtype=np.float32)
    if reads_path is not None and psi.size > 0:
        reads = _load_numeric(reads_path, delimiter)
        keep = np.nanmedian(reads, axis=1) >= float(min_reads)
        psi = psi[keep]
        event_ids = [event_ids[i] for i in range(len(event_ids)) if keep[i]]
    np.clip(psi, 0.0, 1.0, out=psi)
    return event_ids, psi


def _parse_float(value: str) -> float:
    token = value.strip()
    if token == "" or token.upper() in {"NA", "NAN", "NULL"}:
        return float("nan")
    return float(token)


def _load_numeric(path: str | Path, delimiter: str) -> np.ndarray:
    rows: list[list[float]] = []
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            fields = line.rstrip("\n").split(delimiter)
            if len(fields) < 2:
                continue
            rows.append([_parse_float(value) for value in fields[1:]])
    return np.asarray(rows, dtype=np.float32)
