"""Tests for prepare_store.py's prepare_ua_store (needs cosmolayer + rdkit +
the real chaos-store, same skip pattern as the other *_optional.py files).

Builds a small, throwaway united-atom store from a subsample of the real
store rather than coarse-graining the full ~53k-molecule store, which is the
whole point of testing against a subsample: fast, but still real chaos-store
molecules and real cosmolayer machinery, not synthetic data.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cosmolayer")
pytest.importorskip("rdkit")

from sieve_experiments.data import DEFAULT_STORES_ROOT

STORE_NAME = "chaos-store"
STORES_ROOT = DEFAULT_STORES_ROOT
STORE = STORES_ROOT / STORE_NAME
pytestmark = pytest.mark.skipif(not STORE.exists(), reason="chaos-store absent")


def _split_subsample(tmp_path, n=60, seed=0):
    from cosmolayer.store import SegmentStore
    from sieve_experiments.prepare_store import split_chaos_store

    sub = SegmentStore.load(STORE).subsample(n, shuffle_seed=seed)
    src_dir = tmp_path / "mini-store"
    sub.save(src_dir)
    split_chaos_store(src_dir)
    return src_dir


def test_prepare_ua_store_builds_a_valid_united_atom_store(tmp_path):
    from cosmolayer.store import SegmentStore
    from sieve_experiments.prepare_store import prepare_ua_store

    _split_subsample(tmp_path)
    prepare_ua_store("mini-store", "mini-store-ua", stores_root=tmp_path)

    aa = SegmentStore.load(tmp_path / "mini-store")
    ua = SegmentStore.load(tmp_path / "mini-store-ua")

    assert len(ua.molecules_df) == len(aa.molecules_df)
    assert len(ua.atoms_df) < len(aa.atoms_df), "coarse-graining must shrink atom count"
    assert "biased_split" in ua.molecules_df.columns
    # the split must be carried through UNCHANGED, not recomputed -- same
    # molecules in the same train/val/test roles, so AA-vs-UA numbers are a
    # controlled comparison
    assert list(ua.molecules_df["biased_split"]) == list(
        aa.molecules_df["biased_split"]
    )
    assert list(ua.molecules_df["split"]) == list(aa.molecules_df["split"])


def test_prepare_ua_store_preserves_molecule_level_truth_exactly(tmp_path):
    """No segment is ever dropped by coarse-graining -- only atom
    partitioning changes. Molecule-level sigma profiles/areas must therefore
    be bit-identical between the AA and UA stores."""
    import numpy as np
    from cosmolayer.store import SegmentStore
    from sieve_experiments.prepare_store import prepare_ua_store

    _split_subsample(tmp_path)
    prepare_ua_store("mini-store", "mini-store-ua", stores_root=tmp_path)

    aa = SegmentStore.load(tmp_path / "mini-store")
    ua = SegmentStore.load(tmp_path / "mini-store-ua")

    aa_profiles = aa.compute_molecule_sigma_profiles("cosmo-sac-2010")
    ua_profiles = ua.compute_molecule_sigma_profiles("cosmo-sac-2010")

    np.testing.assert_array_equal(
        np.asarray(aa_profiles.profiles), np.asarray(ua_profiles.profiles)
    )
    np.testing.assert_array_equal(
        np.asarray(aa_profiles.areas), np.asarray(ua_profiles.areas)
    )


def test_prepare_ua_store_is_idempotent(tmp_path, caplog):
    from sieve_experiments.prepare_store import prepare_ua_store

    _split_subsample(tmp_path)
    prepare_ua_store("mini-store", "mini-store-ua", stores_root=tmp_path)
    mtime_before = (tmp_path / "mini-store-ua" / "molecules.parquet").stat().st_mtime

    prepare_ua_store("mini-store", "mini-store-ua", stores_root=tmp_path)
    mtime_after = (tmp_path / "mini-store-ua" / "molecules.parquet").stat().st_mtime

    assert mtime_before == mtime_after, "a second call must not rebuild"


def test_prepare_ua_store_requires_the_source_to_be_split(tmp_path):
    import pandas as pd
    from cosmolayer.store import SegmentStore
    from sieve_experiments.prepare_store import prepare_ua_store

    # A subsample of the real chaos-store inherits ITS already-split columns
    # (subsample requires "split" to already exist), so strip both split
    # columns back off after saving to get a genuinely unsplit source store.
    sub = SegmentStore.load(STORE).subsample(20, shuffle_seed=1)
    src_dir = tmp_path / "unsplit-store"
    sub.save(src_dir)
    molecules_path = src_dir / "molecules.parquet"
    df = pd.read_parquet(molecules_path).drop(columns=["split", "biased_split"])
    df.to_parquet(molecules_path)

    with pytest.raises(ValueError, match="biased_split"):
        prepare_ua_store("unsplit-store", "unsplit-store-ua", stores_root=tmp_path)


def test_prepare_ua_store_requires_the_source_to_exist(tmp_path):
    from sieve_experiments.prepare_store import prepare_ua_store

    with pytest.raises(ValueError, match="not a complete store"):
        prepare_ua_store("does-not-exist", "does-not-exist-ua", stores_root=tmp_path)
