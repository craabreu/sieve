"""Optional-data tests against the real chaos-store: the harness's data path
checked against ground truth cosmolayer computes directly, not just against
itself. Skipped unless both cosmolayer and the store are present, same
pattern as tests/test_benchmark.py.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pytest.importorskip("cosmolayer")
pytest.importorskip("rdkit")

STORE_NAME = "chaos-store"
STORES_ROOT = pathlib.Path(__file__).resolve().parents[1] / "stores"
STORE = STORES_ROOT / STORE_NAME
pytestmark = pytest.mark.skipif(not STORE.exists(), reason="chaos-store absent")

LIMIT = 200


@pytest.fixture(scope="module")
def loaded():
    from sieve_experiments.data import load_molecule_set

    return load_molecule_set(
        STORE_NAME,
        scheme="cosmo-sac-2010",
        split_column="biased_split",
        limit=LIMIT,
        stores_root=STORES_ROOT,
    )


def test_prepare_store_is_idempotent(tmp_path, caplog):
    """Running prepare_store against the already-prepared real store does
    nothing and says so -- no re-download, no re-split."""
    from sieve_experiments.prepare_store import prepare_store

    with caplog.at_level("INFO", logger="sieve_experiments"):
        prepare_store(STORE_NAME, stores_root=STORES_ROOT)
    assert "skipping download" in caplog.text
    assert "nothing to do" in caplog.text


def test_grid_is_51_points_bin_width_0_001(loaded):
    mset, _ = loaded
    assert mset.grid.num_points == 51
    assert mset.grid.max_abs_sigma == pytest.approx(0.025)
    assert mset.grid.bin_width == pytest.approx(0.001)


def test_split_masks_partition_the_loaded_molecules(loaded):
    mset, masks = loaded
    total = np.zeros(mset.n_molecules, dtype=int)
    for mask in masks.values():
        total += mask.astype(int)
    # every loaded molecule is in exactly one of train/val/test
    np.testing.assert_array_equal(total, 1)


def test_biased_split_extrapolates_train_lt_val_lt_test():
    """The headline split really does train-on-small, test-on-large -- the
    manifest's train/val/test_mean_num_atoms fields exist to prove this per
    run; this proves it holds on the real store directly."""
    from sieve_experiments.data import load_molecule_set

    mset, masks = load_molecule_set(
        STORE_NAME,
        scheme="cosmo-sac-2010",
        split_column="biased_split",
        stores_root=STORES_ROOT,
    )
    means = {name: float(np.mean(mset.num_atoms[mask])) for name, mask in masks.items()}
    assert means["train"] < means["val"] < means["test"]


def test_molecule_sum_of_atom_profiles_matches_cosmolayer_directly():
    """The highest-value test in this module: build atom-level profiles via
    the same path Sieve/DASH/etc. will, sum them to molecule level, and check
    against cosmolayer's own compute_molecule_sigma_profiles -- not against
    ourselves. The atom -> molecule rollup fails silently otherwise."""
    from cosmolayer.store import SegmentStore
    from sieve_experiments.data import molecule_sum

    store = SegmentStore.load(STORE)
    df = store.molecules_df.iloc[:LIMIT].reset_index(drop=True)

    atom_table = store.compute_atom_sigma_profiles(scheme="cosmo-sac-2010")
    n_atoms_loaded = int(df.num_atoms.sum())
    atom_profile = (
        np.asarray(atom_table.profiles[:n_atoms_loaded], dtype=np.float64)
        * np.asarray(atom_table.areas[:n_atoms_loaded], dtype=np.float64)[:, None]
    )
    mol_id = np.repeat(np.arange(len(df)), df.num_atoms.to_numpy())
    rolled_up = molecule_sum(atom_profile, mol_id, len(df))

    mol_table = store.compute_molecule_sigma_profiles(scheme="cosmo-sac-2010")
    expected = (
        np.asarray(mol_table.profiles[: len(df)], dtype=np.float64)
        * np.asarray(mol_table.areas[: len(df)], dtype=np.float64)[:, None]
    )
    np.testing.assert_allclose(rolled_up, expected, rtol=1e-4, atol=1e-6)


def test_net_charge_close_to_summed_segment_charges():
    """RDKit formal charge (our "known input") vs. the store's own summed
    COSMO segment charge -- they should agree to a small tolerance."""
    from cosmolayer.store import SegmentStore
    from rdkit import Chem
    from sieve_experiments.data import load_molecule_set

    mset, _ = load_molecule_set(
        STORE_NAME,
        scheme="cosmo-sac-2010",
        split_column="biased_split",
        limit=LIMIT,
        stores_root=STORES_ROOT,
    )
    store = SegmentStore.load(STORE)
    df = store.molecules_df.iloc[:LIMIT].reset_index(drop=True)
    ai = np.asarray(store.atom_indices)
    n_nodes = int(np.max(ai)) + 1
    atom_charge = np.bincount(ai, weights=np.asarray(store.charges), minlength=n_nodes)
    mol_id = np.repeat(np.arange(len(df)), df.num_atoms.to_numpy())
    n_atoms_loaded = int(df.num_atoms.sum())
    from sieve_experiments.data import molecule_sum

    cosmo_net_charge = molecule_sum(atom_charge[:n_atoms_loaded], mol_id, len(df))

    params = Chem.SmilesParserParams()
    params.removeHs = False
    for smi in df.smiles[:5]:
        assert Chem.MolFromSmiles(smi, params) is not None

    diffs = np.abs(cosmo_net_charge - mset.net_charge)
    assert np.median(diffs) < 0.2, "median |COSMO charge - formal charge| too large"


def test_limit_matches_full_load_on_the_first_n_molecules():
    """--limit N truncates, it does not resample -- the same first N
    molecules whether or not a limit is applied."""
    from sieve_experiments.data import load_molecule_set

    limited, _ = load_molecule_set(
        STORE_NAME,
        scheme="cosmo-sac-2010",
        split_column="biased_split",
        limit=10,
        stores_root=STORES_ROOT,
    )
    full, _ = load_molecule_set(
        STORE_NAME,
        scheme="cosmo-sac-2010",
        split_column="biased_split",
        stores_root=STORES_ROOT,
    )
    assert limited.mol_area is not None
    assert full.mol_area is not None
    assert limited.smiles == full.smiles[:10]
    np.testing.assert_allclose(limited.mol_area, full.mol_area[:10])


def test_load_atom_truth_rolls_up_to_the_same_molecule_profile(loaded):
    """load_atom_truth (used by DASH's fit_atoms) is a completely separate
    code path from load_molecule_set -- check the two agree via the atom ->
    molecule rollup, not against themselves."""
    from sieve_experiments.data import load_atom_truth, molecule_sum

    mset, _ = loaded
    atom_profile, atom_area, atom_charge = load_atom_truth(
        STORE_NAME,
        scheme="cosmo-sac-2010",
        smiles=mset.smiles,
        num_atoms=mset.num_atoms,
        stores_root=STORES_ROOT,
    )
    mol_id = mset.atom_mol_id
    assert mset.mol_profile is not None
    assert mset.mol_area is not None
    np.testing.assert_allclose(
        molecule_sum(atom_profile, mol_id, mset.n_molecules),
        mset.mol_profile,
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        molecule_sum(atom_area, mol_id, mset.n_molecules),
        mset.mol_area,
        rtol=1e-4,
        atol=1e-6,
    )
    assert atom_charge.shape == (mset.n_atoms,)


def test_load_atom_truth_charge_is_scheme_consistent_not_raw(loaded):
    """atom_charge must be the averaged-scheme charge (area * sigma, from the
    SAME cosmo-sac-2010 profile as atom_profile), not the raw pre-averaging
    store.charges -- those are different quantities (COSMO-SAC's sigma
    averaging redistributes charge across atom boundaries, so raw per-atom
    charge and averaged-profile-consistent per-atom charge disagree by
    ~0.01 mean abs, a ~60% relative gap on this store). Two independent
    checks that atom_charge is the averaged one: it integrates to the same
    value as sigma-weighting atom_profile, and it rolls up to the
    molecule-level mol_charge already loaded via load_molecule_set (which
    has always used the correctly-averaged cosmolayer field)."""
    from sieve_experiments.data import DEFAULT_GRID, load_atom_truth, molecule_sum

    mset, _ = loaded
    atom_profile, _atom_area, atom_charge = load_atom_truth(
        STORE_NAME,
        scheme="cosmo-sac-2010",
        smiles=mset.smiles,
        num_atoms=mset.num_atoms,
        stores_root=STORES_ROOT,
    )
    implied_charge = (atom_profile * DEFAULT_GRID.values[None, :]).sum(axis=1)
    np.testing.assert_allclose(atom_charge, implied_charge, atol=1e-6)

    assert mset.mol_charge is not None
    np.testing.assert_allclose(
        molecule_sum(atom_charge, mset.atom_mol_id, mset.n_molecules),
        mset.mol_charge,
        rtol=1e-4,
        atol=1e-6,
    )


def test_load_atom_truth_rejects_a_smiles_not_in_the_store(loaded):
    from sieve_experiments.data import load_atom_truth

    mset, _ = loaded
    with pytest.raises(KeyError, match="not found"):
        load_atom_truth(
            STORE_NAME,
            scheme="cosmo-sac-2010",
            smiles=[*mset.smiles[:3], "not-a-real-smiles"],
            num_atoms=np.concatenate([mset.num_atoms[:3], [1]]),
            stores_root=STORES_ROOT,
        )


def test_end_to_end_run_on_real_store_limit_50(tmp_path):
    """A real, small (--limit LIMIT, i.e. 200) run of the global_mean floor
    predictor against the real store, through the same execute() path a real
    experiment uses -- not a synthetic fixture.

    Named "_limit_50" for history; the limit itself is LIMIT (200), not 50:
    biased_split's train fraction is 80%, and on this store the first 50
    rows (in store order) land entirely in train -- an empty val/test would
    make ``np.isfinite(w1_norm_mean)`` below false by construction, not a
    real check. LIMIT=200 (already used by the ``loaded`` fixture above) is
    the smallest round number confirmed to give a non-empty val and test.
    """
    from sieve_experiments.config import DataCfg, ExperimentCfg, PredictorCfg, RunCfg
    from sieve_experiments.data import load_molecule_set
    from sieve_experiments.runner import execute

    cfg = ExperimentCfg(
        run=RunCfg(experiment="store-smoke", seed=0),
        data=DataCfg(
            store=STORE_NAME, scheme="cosmo-sac-2010", split_column="biased_split"
        ),
        predictor=PredictorCfg(name="global_mean", params={}),
    )
    mset, masks = load_molecule_set(
        STORE_NAME,
        scheme=cfg.data.scheme,
        split_column=cfg.data.split_column,
        limit=LIMIT,
        stores_root=STORES_ROOT,
    )
    assert masks["val"].sum() > 0 and masks["test"].sum() > 0  # guard the premise
    result = execute(
        cfg, mset, masks, runs_root=tmp_path, allow_dirty=True, tracking=None
    )
    assert result.run_dir.is_dir()
    assert np.isfinite(result.metrics["profile/w1_norm_mean"])
    assert result.metrics["n_test"] > 0
