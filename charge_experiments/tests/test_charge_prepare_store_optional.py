"""End-to-end prepare_store test against the real, already-downloaded
dashMoleculesSDF_v2.sdf. Skipped entirely if that file is absent -- matches
cosmo_experiments' *_optional.py pattern for tests needing the real store."""

from __future__ import annotations

from pathlib import Path

import pytest

_REAL_SDF = Path.home() / "tmp" / "dash_molecules" / "dashMoleculesSDF_v2.sdf"

pytestmark = pytest.mark.skipif(
    not _REAL_SDF.exists(), reason="real dashMoleculesSDF_v2.sdf not present locally"
)


def test_parse_first_n_records_of_the_real_sdf(tmp_path):
    """Doesn't parse the whole 8.3GB file (would take too long for a test
    run) -- copies just the first two records' worth of bytes (up to the
    second '$$$$' terminator) into a small temp file and parses that,
    exercising the real parser against the real file's real property names
    and formatting."""
    from charge_experiments.prepare_store import parse_dash_molecules

    with _REAL_SDF.open("rb") as f:
        chunk = f.read(1 << 20)  # 1 MiB, far more than two records need
    text = chunk.decode("utf-8", errors="replace")
    first = text.index("$$$$")
    second = text.index("$$$$", first + 1)
    end = second + len("$$$$\n")
    sample_path = tmp_path / "sample.sdf"
    sample_path.write_bytes(chunk[:end])

    out_path = tmp_path / "molecules.parquet"
    parse_dash_molecules(sample_path, out_path)

    import pandas as pd

    df = pd.read_parquet(out_path)
    assert len(df) == 2
    assert df.loc[0, "chembl_id"] == "CHEMBL185198"
