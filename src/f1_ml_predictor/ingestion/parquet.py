"""Atomic, source-versioned Parquet partition writes."""

import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_partition(table: pa.Table, path: Path, source_hash: str, schema_version: str) -> bool:
    metadata = {
        b"schema_version": schema_version.encode("ascii"),
        b"source_sha256": source_hash.encode("ascii"),
    }
    if path.exists() and pq.read_schema(path).metadata == metadata:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        pq.write_table(table.replace_schema_metadata(metadata), temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return True
