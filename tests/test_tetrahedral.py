"""Tetrahedral handedness (spec 2026-09-24-tetrahedral-handedness-design.md, §7)."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

import sieve
from sieve.config import KIND_AWARE, KIND_BLIND, SieveConfig
from sieve.io.rdkit_adapter import build_codes, from_rdkit
from sieve.level import blind_targets, class_kinds
from sieve.refine import refine

BOTH = ("cis_trans", "tetrahedral")


def _mols(smiles, *, hs=True):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return [Chem.AddHs(m) for m in mols] if hs else mols


def _config(mols, *, stereo=BOTH, depth=4, **kw):
    codes, edges = build_codes(mols, ["element"])
    return SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes=codes,
        edge_codes=edges,
        max_wl_depth=depth,
        stereo=stereo,
        **kw,
    )


def _batch(mols, cfg, *, seed=0, node_order=None):
    b = from_rdkit(mols, y=None, config=cfg, node_order=node_order)
    y = np.random.default_rng(seed).normal(size=(b.n_nodes, 1))
    return dataclasses.replace(b, y=y)


def _stripped(mols):
    out = [Chem.Mol(m) for m in mols]
    for m in out:
        for a in m.GetAtoms():
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return out


def _inverted(mols):
    out = [Chem.Mol(m) for m in mols]
    for m in out:
        for a in m.GetAtoms():
            a.InvertChirality()
    return out


def test_track_order_is_normalised_and_duplicates_refused():
    mols = _mols(["CC"])
    a = _config(mols, stereo=("tetrahedral", "cis_trans"))
    b = _config(mols, stereo=("cis_trans", "tetrahedral"))
    assert a.stereo == BOTH and a.schema_version == b.schema_version
    with pytest.raises(ValueError, match="twice"):
        _config(mols, stereo=("cis_trans", "cis_trans"))


def test_three_digests_and_cis_trans_unchanged():
    mols = _mols(["CC"])
    digests = {
        _config(mols, stereo=s).schema_version
        for s in ((), ("cis_trans",), BOTH)
    }
    assert len(digests) == 3
    assert _config(mols, stereo=BOTH).stereo_radices == (4, 4)


from sieve.batch import NodeBatch, concat_batches


def _star_batch(centres):
    """Atom 0 bonded to atoms 1..4; two copies side by side."""
    src = [0, 1, 0, 2, 0, 3, 0, 4]
    dst = [1, 0, 2, 0, 3, 0, 4, 0]
    src += [s + 5 for s in src]
    dst += [d + 5 for d in dst]
    return NodeBatch(
        node_attrs=np.zeros((10, 1), np.int64),
        edge_src=np.array(src),
        edge_dst=np.array(dst),
        edge_attrs=np.zeros((16, 1), np.int64),
        graph_id=np.array([0] * 5 + [1] * 5),
        stereo_centres=np.array(centres, np.int64).reshape(-1, 6),
    )


@pytest.mark.parametrize(
    "row, message",
    [
        ([0, 1, 2, 3, 6, 1], "adjacent"),
        ([0, 1, -1, 3, 4, 1], "only in n3"),
        ([0, 1, 2, 3, -1, 1], "degree"),
        ([0, 1, 2, 3, 4, 0], "parity"),
        ([0, 1, 2, 3, 99, 1], "range"),
    ],
)
def test_stereo_centres_are_validated(row, message):
    with pytest.raises(ValueError, match=message):
        _star_batch([row])


def test_a_centre_listed_twice_is_refused():
    with pytest.raises(ValueError, match="once"):
        _star_batch([[0, 1, 2, 3, 4, 1], [0, 4, 3, 2, 1, 1]])


def test_stereo_centres_follow_slicing_and_concat():
    b = _star_batch([[0, 1, 2, 3, 4, 1], [5, 6, 7, 8, 9, -1]])
    second = b[b.graph_id == 1]
    np.testing.assert_array_equal(second.stereo_centres, [[0, 1, 2, 3, 4, -1]])
    both = concat_batches([second, second])
    np.testing.assert_array_equal(
        both.stereo_centres, [[0, 1, 2, 3, 4, -1], [5, 6, 7, 8, 9, -1]]
    )
    with pytest.raises(ValueError, match="stereo_centres"):
        concat_batches([second, dataclasses.replace(second, stereo_centres=None)])


from sieve.io.rdkit_adapter import _stereo_centre_rows


def test_enantiomers_give_opposite_parity_on_identical_rows():
    (r,) = _stereo_centre_rows(_mols(["F[C@H](Cl)Br"])[0])
    (s,) = _stereo_centre_rows(_mols(["F[C@@H](Cl)Br"])[0])
    assert r[:5] == s[:5] and r[5] == -s[5]


def test_a_sulfoxide_has_a_virtual_fourth_neighbour():
    (row,) = _stereo_centre_rows(_mols(["C[S@](=O)CC"])[0])
    assert row[4] == -1


def test_untagged_and_two_neighbour_centres_give_no_row():
    assert _stereo_centre_rows(_mols(["CC(C)CC"])[0]) == []
    assert _stereo_centre_rows(Chem.MolFromSmiles("C[P@H]CC")) == []


def test_the_batch_carries_centres_only_under_the_track():
    mols = _mols(["F[C@H](Cl)Br", "C/C=C/C"])
    both = from_rdkit(mols, config=_config(mols))
    ct = from_rdkit(mols, config=_config(mols, stereo=("cis_trans",)))
    tet = from_rdkit(mols, config=_config(mols, stereo=("tetrahedral",)))
    assert both.stereo_centres.shape == (1, 6) and both.stereo_bonds.shape == (1, 7)
    assert ct.stereo_centres is None
    assert tet.stereo_bonds is None and tet.stereo_centres.shape == (1, 6)


def test_parallel_featurisation_carries_centres():
    mols = _mols(["F[C@H](Cl)Br", "N[C@@H](C)C(=O)O"] * 4)
    cfg = _config(mols)
    seq = from_rdkit(mols, config=cfg)
    par = from_rdkit(mols, config=cfg, n_jobs=2)
    np.testing.assert_array_equal(seq.stereo_centres, par.stereo_centres)
