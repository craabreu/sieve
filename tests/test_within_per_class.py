"""Per-class within-structure variance (within-structure-variance spec, section 4).

The per-class sums must follow exactly the membership `y` follows. The
cleanest check is to feed a copy of `y` through them: with within_sse = y and
within_n = 1 on every atom, each class's within_n must equal its count and its
within_sse / within_n its mean, under every stereo track.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import sieve
from tests.helpers import chain_batch, simple_config, split_batch

CIS_TRANS = [
    "C/C=C/C",
    r"C/C=C\C",
    "C/C=C/CC",
    r"C/C=C\CC",
    "C/C(F)=C(Cl)/C",
    r"C/C(F)=C(Cl)\C",
    "CCCC",
    "C/C=C/Br",
]
CHIRAL = [
    "N[C@@H](C)C(=O)O",
    "N[C@H](C)C(=O)O",
    "C[C@H](O)[C@H](N)C",
    "C[C@H](O)[C@@H](N)C",
    "OC[C@@H](O)[C@H](O)C=O",
    "CC(C)C",
]


def _echo(batch, seed=0):
    """y positive, and the within arrays a copy of it with one member each."""
    y = np.random.default_rng(seed).uniform(0.1, 1.0, size=(batch.n_nodes, 1))
    return dataclasses.replace(
        batch, y=y, within_sse=y.copy(), within_n=np.ones(batch.n_nodes)
    )


def _stereo_batch(smiles, stereo):
    pytest.importorskip("rdkit")
    from rdkit import Chem

    from sieve.config import SieveConfig
    from sieve.io.rdkit_adapter import build_codes, from_rdkit

    mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in smiles]
    codes, edges = build_codes(mols, ["element"])
    cfg = SieveConfig(
        target_dim=1,
        attribute_levels=(("element",),),
        attribute_codes=codes,
        edge_codes=edges,
        max_wl_depth=4,
        stereo=stereo,
    )
    return _echo(from_rdkit(mols, config=cfg)), cfg


def _assert_echoes_y(model):
    for lvl in model.levels:
        assert lvl.within_sse is not None and lvl.within_n is not None
        np.testing.assert_array_equal(lvl.within_n, lvl.count.astype(np.float64))
        filled = lvl.count > 0
        np.testing.assert_allclose(
            lvl.within_sse[filled] / lvl.within_n[filled, None],
            lvl.mean[filled],
            rtol=1e-12,
        )


@pytest.mark.parametrize(
    ("smiles", "stereo"),
    [
        (CIS_TRANS, ()),
        (CIS_TRANS, ("cis_trans",)),
        (CHIRAL, ("cis_trans", "tetrahedral")),
    ],
    ids=["blind", "cis_trans", "both_tracks"],
)
def test_within_sums_follow_y_under_every_track(smiles, stereo):
    batch, cfg = _stereo_batch(smiles, stereo)
    _assert_echoes_y(sieve.fit(batch, cfg))


def test_within_sums_follow_y_on_a_plain_chain():
    _assert_echoes_y(sieve.fit(_echo(chain_batch(12, graphs=4)), simple_config()))


def test_a_batch_without_within_arrays_has_no_per_class_sums():
    m = sieve.fit(chain_batch(12, graphs=4), simple_config())
    assert all(lvl.within_sse is None and lvl.within_n is None for lvl in m.levels)


def test_per_class_sums_leave_the_class_statistics_unchanged():
    b = chain_batch(12, graphs=4)
    plain = sieve.fit(b, simple_config())
    rng = np.random.default_rng(3)
    with_sums = sieve.fit(
        dataclasses.replace(
            b, within_sse=rng.uniform(size=(b.n_nodes, 1)), within_n=np.ones(b.n_nodes)
        ),
        simple_config(),
    )
    for p, w in zip(plain.levels, with_sums.levels, strict=True):
        np.testing.assert_array_equal(p.count, w.count)
        np.testing.assert_array_equal(p.mean, w.mean)
        np.testing.assert_array_equal(p.msd, w.msd)


def test_blind_class_sums_add_up_to_the_pooled_sums():
    from sieve.config import KIND_BLIND
    from sieve.level import class_kinds

    batch, cfg = _stereo_batch(CIS_TRANS, ("cis_trans",))
    rng = np.random.default_rng(5)
    batch = dataclasses.replace(
        batch,
        within_sse=rng.uniform(size=(batch.n_nodes, 1)),
        within_n=rng.integers(1, 4, size=batch.n_nodes).astype(np.float64),
    )
    m = sieve.fit(batch, cfg)
    assert m.within_sse is not None
    for lvl in m.levels:
        assert lvl.within_sse is not None and lvl.within_n is not None
        blind = (class_kinds(lvl) & KIND_BLIND) > 0
        np.testing.assert_allclose(lvl.within_sse[blind].sum(axis=0), m.within_sse)
        assert lvl.within_n[blind].sum() == pytest.approx(m.within_n)


# --------------------------------------------------------- merge and I/O --


def test_merge_of_two_fits_echoes_y_like_the_fit_of_their_union():
    b = _echo(chain_batch(12, graphs=6))
    cfg = simple_config()
    mask = b.graph_id < 3
    merged = sieve.fit(split_batch(b, mask), cfg).merge(
        sieve.fit(split_batch(b, ~mask), cfg)
    )
    _assert_echoes_y(merged)


def test_stereo_merge_echoes_y():
    batch, cfg = _stereo_batch(CHIRAL, ("cis_trans", "tetrahedral"))
    mask = batch.graph_id < 3
    merged = sieve.fit(split_batch(batch, mask), cfg).merge(
        sieve.fit(split_batch(batch, ~mask), cfg)
    )
    _assert_echoes_y(merged)


def test_merge_with_a_level_without_sums_keeps_the_other_side():
    b = chain_batch(12, graphs=6)
    cfg = simple_config()
    mask = b.graph_id < 3
    with_sums = sieve.fit(_echo(split_batch(b, mask)), cfg)
    without = sieve.fit(split_batch(b, ~mask), cfg)
    for merged in (with_sums.merge(without), without.merge(with_sums)):
        for lvl in merged.levels:
            assert lvl.within_n is not None and lvl.within_sse is not None
        level0 = merged.levels[0]
        assert level0.within_n is not None
        assert float(level0.within_n.sum()) == pytest.approx(float(mask.sum()))


def test_the_empty_model_keeps_per_class_sums_under_merge():
    from sieve.model import SieveModel

    m = sieve.fit(_echo(chain_batch(12, graphs=4)), simple_config())
    e = SieveModel.empty(m.config)
    _assert_echoes_y(m.merge(e))
    _assert_echoes_y(e.merge(m))


def test_save_load_round_trips_per_class_sums(tmp_path):
    from sieve.model import SieveModel

    m = sieve.fit(_echo(chain_batch(12, graphs=4)), simple_config())
    path = tmp_path / "m.npz"
    m.save(path)
    back = SieveModel.load(path)
    for a, b in zip(m.levels, back.levels, strict=True):
        assert a.within_sse is not None and b.within_sse is not None
        assert a.within_n is not None and b.within_n is not None
        np.testing.assert_array_equal(a.within_sse, b.within_sse)
        np.testing.assert_array_equal(a.within_n, b.within_n)


def test_a_file_without_per_class_sums_keeps_its_keys(tmp_path):
    from sieve.model import SieveModel

    m = sieve.fit(chain_batch(12, graphs=4), simple_config())
    path = tmp_path / "m.npz"
    m.save(path)
    assert not any("within" in f for f in np.load(path).files)
    assert all(lvl.within_sse is None for lvl in SieveModel.load(path).levels)


# ----------------------------------------------------- predictive variance --


def _model_with_class_sums(seed=0):
    b = chain_batch(12, graphs=4, seed=seed)
    rng = np.random.default_rng(seed + 10)
    n = rng.integers(1, 5, size=b.n_nodes).astype(np.float64)
    sse = rng.uniform(0.0, 2e-4, size=(b.n_nodes, 1)) * n[:, None]
    return sieve.fit(
        dataclasses.replace(b, within_sse=sse, within_n=n),
        simple_config(max_wl_depth=2),
    )


def test_infinite_shrinkage_is_phase_one_bit_for_bit():
    from sieve.uncertainty import WITHIN_SHRINKAGE, predictive_variance

    assert WITHIN_SHRINKAGE == np.inf
    m = _model_with_class_sums()
    # phase 1: the same model with its per-class sums stripped
    pooled_only = dataclasses.replace(
        m,
        levels=tuple(
            dataclasses.replace(lvl, within_sse=None, within_n=None) for lvl in m.levels
        ),
    )
    for a, b in zip(
        predictive_variance(m), predictive_variance(pooled_only), strict=True
    ):
        np.testing.assert_array_equal(a, b)


def test_finite_shrinkage_is_the_hand_computed_blend():
    from sieve.uncertainty import predictive_variance

    m = _model_with_class_sums()
    beta = 3.0
    pooled = m.within_variance
    base = predictive_variance(m)  # pooled sigma2_w in every class
    blended = predictive_variance(m, within_shrinkage=beta)
    for lvl, b0, b1 in zip(m.levels, base, blended, strict=True):
        assert lvl.within_sse is not None and lvl.within_n is not None
        s2c = (lvl.within_sse + beta * pooled) / (lvl.within_n[:, None] + beta)
        np.testing.assert_allclose(b1, b0 - pooled + s2c, rtol=1e-12)


def test_zero_shrinkage_uses_each_class_s_own_ratio_and_pooled_where_empty():
    from sieve.uncertainty import predictive_variance

    m = _model_with_class_sums()
    base = predictive_variance(m)
    own = predictive_variance(m, within_shrinkage=0.0)
    pooled = m.within_variance
    for lvl, b0, b1 in zip(m.levels, base, own, strict=True):
        assert lvl.within_sse is not None and lvl.within_n is not None
        n = lvl.within_n[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(n > 0, lvl.within_sse / n, pooled)
        np.testing.assert_allclose(b1, b0 - pooled + ratio, rtol=1e-12)
        assert np.isfinite(b1).all()


def test_a_model_without_per_class_sums_ignores_the_shrinkage():
    from sieve.uncertainty import predictive_variance

    m = sieve.fit(chain_batch(12, graphs=4), simple_config()).with_within_structure(
        1e-3, 10
    )
    for a, b in zip(
        predictive_variance(m),
        predictive_variance(m, within_shrinkage=1.0),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
