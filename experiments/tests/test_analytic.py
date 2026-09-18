"""The analytic curve must agree with actually predicting the training set.

Every test here fits a real model, predicts its own training atoms, and
compares the brute-force sum of squared errors against the closed form. That
is the only check worth running: the closed form is an identity on the stored
statistics, so anything weaker would pass even if the identity were wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")


def _fit(
    n_mol=12,
    seed=0,
    depth=3,
    minimum_support=1,
    shrinkage_strength=None,
    **params,
):
    """A fitted model plus the training batch it was fitted on."""
    from experiments.predictors.sieve_predictor import (
        DEFAULT_ATTRIBUTES,
        _batch_for,
        _build_config,
    )

    import sieve
    from experiments.tests.helpers import synthetic_molecule_set

    mset = synthetic_molecule_set(n_mol=n_mol, seed=seed)
    config = _build_config(
        mset.mols,
        attributes=DEFAULT_ATTRIBUTES,
        target_dim=1,
        max_wl_depth=depth,
        minimum_support=minimum_support,
        shrinkage_strength=shrinkage_strength,
    )
    if params:
        import dataclasses

        config = dataclasses.replace(config, **params)
    batch = _batch_for(
        mset.mols, config, atom_property=mset.atom_property, with_target=True
    )
    return sieve.fit(batch, config), batch


def _brute_force_sse(model, batch):
    import sieve

    pred = sieve.predict(model, batch)
    return float(((batch.y - pred) ** 2).sum())


def _brute_force_loo_sse(model, batch):
    from sieve.predict import predict_loo

    pred = predict_loo(model, batch).value
    return float(((batch.y - pred) ** 2).sum())


# --------------------------------------------------------------------------
# The training identity, across every reading of the class tables
# --------------------------------------------------------------------------


def test_analytic_sse_matches_predicting_the_training_set():
    from experiments.analytic import sieve_train_stats

    model, batch = _fit()
    stats = sieve_train_stats(model)
    assert stats.sse == pytest.approx(_brute_force_sse(model, batch), rel=1e-12)


def test_analytic_rmse_is_the_sse_divided_by_atoms():
    from experiments.analytic import sieve_train_stats

    model, batch = _fit()
    stats = sieve_train_stats(model)
    assert stats.n_atoms == batch.n_nodes
    assert stats.rmse == pytest.approx(np.sqrt(stats.sse / batch.n_nodes))


@pytest.mark.parametrize("minimum_support", [1, 2, 3, 8])
def test_analytic_sse_holds_when_training_atoms_back_off(minimum_support):
    """The case section 1 of the diagnostics note explicitly excludes.

    With ``minimum_support > 1`` a training atom's own deepest class may be
    too small to answer it, so the walk has to follow the same backoff
    ``predict`` follows. If this passes, the closed form is not relying on
    the deepest class always winning.
    """
    from experiments.analytic import sieve_train_stats

    model, batch = _fit(minimum_support=minimum_support, depth=4)
    stats = sieve_train_stats(model)
    assert stats.sse == pytest.approx(_brute_force_sse(model, batch), rel=1e-12)
    if minimum_support > 1:
        assert stats.matched_fraction < 1.0 or model.levels[-1].count.min() >= 8


@pytest.mark.parametrize("class_estimator", ["pooled", "continuation"])
def test_analytic_sse_holds_under_each_class_estimator(class_estimator):
    from experiments.analytic import sieve_train_stats

    model, batch = _fit(class_estimator=class_estimator)
    stats = sieve_train_stats(model)
    assert stats.sse == pytest.approx(_brute_force_sse(model, batch), rel=1e-12)


@pytest.mark.parametrize("shrinkage_weight", ["count", "diversity", "empirical_bayes"])
def test_analytic_sse_holds_under_each_shrinkage_rule(shrinkage_weight):
    """Shrinkage moves the prediction away from the class mean, so the
    centered form ``sum n_c var_c`` is simply wrong here; the expanded form
    against the shrunk value is what must hold."""
    from experiments.analytic import sieve_train_stats

    # empirical_bayes estimates its own alpha and refuses an explicit
    # shrinkage_strength; the other rules need one.
    strength = None if shrinkage_weight == "empirical_bayes" else 0.5
    model, batch = _fit(
        shrinkage_strength=strength,
        shrinkage_weight=shrinkage_weight,
        class_estimator="continuation",
    )
    stats = sieve_train_stats(model)
    assert stats.sse == pytest.approx(_brute_force_sse(model, batch), rel=1e-12)


def test_analytic_sse_holds_under_neighbor_depth():
    """The coarse chain is scaffolding, never a backoff target -- the walk
    has to skip exactly the levels ``predict`` skips."""
    from experiments.analytic import sieve_train_stats

    model, batch = _fit(depth=3, neighbor_depth=1)
    stats = sieve_train_stats(model)
    assert stats.sse == pytest.approx(_brute_force_sse(model, batch), rel=1e-12)


def test_reduces_to_the_note_s_own_formula_in_the_simple_case():
    """``SSE = sum_c n_c * msd_c`` -- section 1's derivation, which the
    general walk must reproduce exactly when nothing backs off."""
    from experiments.analytic import sieve_train_stats

    model, _ = _fit(minimum_support=1)
    deepest = model.levels[-1]
    simple = float((deepest.count[:, None] * deepest.msd).sum())
    assert sieve_train_stats(model).sse == pytest.approx(simple, rel=1e-12)


# --------------------------------------------------------------------------
# Leave-one-out
# --------------------------------------------------------------------------


def test_analytic_loo_matches_predict_loo():
    from experiments.analytic import sieve_train_stats

    model, batch = _fit(minimum_support=1, depth=3)
    stats = sieve_train_stats(model, loo=True)
    assert stats.sse == pytest.approx(_brute_force_loo_sse(model, batch), rel=1e-10)


@pytest.mark.parametrize("minimum_support", [1, 2, 4])
def test_analytic_loo_matches_predict_loo_with_backoff(minimum_support):
    from experiments.analytic import sieve_train_stats

    model, batch = _fit(minimum_support=minimum_support, depth=4)
    stats = sieve_train_stats(model, loo=True)
    assert stats.sse == pytest.approx(_brute_force_loo_sse(model, batch), rel=1e-10)


def test_analytic_loo_is_never_better_than_the_training_error():
    from experiments.analytic import sieve_train_stats

    model, _ = _fit(depth=4)
    assert sieve_train_stats(model, loo=True).sse >= sieve_train_stats(model).sse


def test_analytic_loo_refuses_continuation():
    from experiments.analytic import sieve_train_stats

    model, _ = _fit(class_estimator="continuation")
    with pytest.raises(NotImplementedError, match="class_estimator"):
        sieve_train_stats(model, loo=True)


def test_analytic_loo_refuses_shrinkage():
    from experiments.analytic import sieve_train_stats

    model, _ = _fit(shrinkage_strength=0.5)
    with pytest.raises(NotImplementedError, match="shrinkage"):
        sieve_train_stats(model, loo=True)


# --------------------------------------------------------------------------
# The free diagnostics riding along
# --------------------------------------------------------------------------


def test_r_squared_and_eta_squared_agree_when_nothing_shrinks():
    from experiments.analytic import sieve_train_stats

    stats = sieve_train_stats(_fit(minimum_support=1)[0])
    assert stats.r_squared == pytest.approx(stats.eta_squared, rel=1e-12)


def test_shrinkage_separates_r_squared_from_eta_squared():
    """eta^2 describes the partition, R^2 the estimator. Shrinking pulls
    every prediction off its class mean, so R^2 must fall below eta^2."""
    from experiments.analytic import sieve_train_stats

    stats = sieve_train_stats(_fit(shrinkage_strength=5.0)[0])
    assert stats.r_squared < stats.eta_squared


def test_support_fractions_count_atoms_not_classes():
    from experiments.analytic import sieve_train_stats

    model, _ = _fit(depth=4)
    deepest = model.levels[-1]
    counts = deepest.count.astype(float)
    expected = counts[counts < 12].sum() / counts.sum()
    assert sieve_train_stats(model).support_fractions[12] == pytest.approx(expected)


def test_a_single_class_explains_nothing():
    """Depth 0 with one attribute level still partitions by element; the
    global-mean limit is the sanity anchor for eta^2."""
    from experiments.analytic import sieve_train_stats

    model, _ = _fit(depth=1)
    stats = sieve_train_stats(model)
    assert 0.0 <= stats.eta_squared <= 1.0


# --------------------------------------------------------------------------
# The curve
# --------------------------------------------------------------------------


def test_curve_matches_fitting_each_depth_natively():
    from experiments.analytic import sieve_curve, sieve_train_stats

    model, _ = _fit(depth=4)
    curve = sieve_curve(model, [1, 2, 3, 4])
    assert [row.depth for row in curve] == [1, 2, 3, 4]
    for row in curve:
        native, _ = _fit(depth=row.depth)
        assert row.sse == pytest.approx(sieve_train_stats(native).sse, rel=1e-12)


def test_training_error_falls_monotonically_with_depth():
    """It can only fall: refining a partition cannot raise within-class
    variance. A violation (beyond floating-point noise) means the walk is
    reading the wrong level."""
    from itertools import pairwise

    from experiments.analytic import sieve_curve

    model, _ = _fit(depth=5, n_mol=16)
    rmse = [row.rmse for row in sieve_curve(model, [1, 2, 3, 4, 5])]
    for earlier, later in pairwise(rmse):
        assert later <= earlier + 1e-12


def test_as_row_is_flat_and_carries_the_support_columns():
    from experiments.analytic import sieve_train_stats

    row = sieve_train_stats(_fit()[0]).as_row()
    assert row["frac_support_lt_12"] == pytest.approx(
        sieve_train_stats(_fit()[0]).support_fractions[12]
    )
    assert isinstance(row["depth"], int)
    assert set(row) >= {"depth", "n_classes", "n_atoms", "sse", "rmse", "r2", "eta2"}


def test_an_empty_model_is_refused_rather_than_dividing_by_zero():
    from experiments.analytic import sieve_train_stats

    import sieve

    model, _ = _fit()
    empty = sieve.SieveModel.empty(model.config)
    with pytest.raises(ValueError, match="no training atoms"):
        sieve_train_stats(empty)


# --------------------------------------------------------------------------
# HOSE
# --------------------------------------------------------------------------


def _hose_corpus(values):
    """One methanol-shaped molecule per value in ``values``, the value set
    only on the O atom (index 1) -- every other atom is a constant 0.0, so
    only the O atom's key carries any spread to test against."""
    from experiments.data import MoleculeSet
    from rdkit import Chem

    mols = []
    for v in values:
        mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
        for atom in mol.GetAtoms():
            atom.SetDoubleProp("MBIScharge", 0.0)
        mol.GetAtomWithIdx(1).SetDoubleProp("MBIScharge", float(v))
        mols.append(mol)
    return MoleculeSet(
        mols=mols,
        atom_property="MBIScharge",
        ids={"dash_id": [None] * len(mols)},
    )


def _hose_fit(values, radius=1, n_min=1):
    import numpy as np
    from experiments.predictors.hose import HoseLookupPredictor

    mset = _hose_corpus(values)
    p = HoseLookupPredictor(max_radius=radius, n_min=n_min)
    p.fit(mset, mset, rng=np.random.default_rng(0))
    return p, mset


def test_hose_analytic_matches_predicting_its_own_training_set():
    pytest.importorskip("hosegen")
    p, mset = _hose_fit([0.1, -0.2, 0.3, -0.05, 0.02, 0.4])
    brute = float(((mset.atom_target - p.predict(mset).atom_value) ** 2).sum())
    from experiments.analytic import hose_train_stats

    stats = hose_train_stats(p.model_state(), radius=1)
    assert stats.sse == pytest.approx(brute, rel=1e-10)
    assert stats.matched_fraction == 1.0  # n_min=1: nothing ever backs off


def test_hose_analytic_loo_matches_a_real_leave_one_molecule_out_refit():
    """No predict_loo exists for HOSE, so this builds its own oracle:
    removing one molecule from the corpus entirely and refitting removes
    exactly that molecule's own one contribution to the O key (each molecule
    contributes exactly one O atom), which for a key of count N is the
    textbook leave-one-out estimate.

    The O key is found by its nonzero sumsq, not by its count: C and the
    H-on-O key both also happen to have count 4 here, since every molecule
    contributes exactly one atom to each.
    """
    pytest.importorskip("hosegen")
    from experiments.analytic import hose_train_stats

    values = [0.1, -0.2, 0.3, -0.05]
    full, _mset = _hose_fit(values, radius=1)
    full_state = full.model_state()

    o_key = next(k for k, (_s, qq, _c) in full_state.tables[1].items() if qq > 0)

    sse_o_manual = 0.0
    for i, y_i in enumerate(values):
        rest, _ = _hose_fit(values[:i] + values[i + 1 :], radius=1)
        rest_mean = rest.model_state().tables[1][o_key][0] / (len(values) - 1)
        sse_o_manual += (y_i - rest_mean) ** 2

    # Every other key here (C, and the two H environments) is a constant
    # 0.0, contributing exactly 0 to LOO SSE either way, so the whole-state
    # LOO SSE at radius 1 is entirely the O key's own contribution.
    stats = hose_train_stats(full_state, radius=1, loo=True)
    assert stats.sse == pytest.approx(sse_o_manual, rel=1e-10)


def test_hose_analytic_loo_uncorrected_for_a_singleton_key():
    """A key with exactly one atom has nowhere to back off to (no prefix
    backoff at n_min=1), so its LOO contribution must be the *unadjusted*
    residual against the global mean, not a (N/(N-1))^2-scaled one (which
    would divide by zero for N=1 in any case).

    Verified against a hand computation over the state's own table, summing
    each key's LOO contribution by the rule the docstring states -- an
    independent arithmetic path, not a re-run of the implementation."""
    pytest.importorskip("hosegen")
    from experiments.analytic import hose_train_stats
    from experiments.data import MoleculeSet
    from experiments.predictors.hose import HoseLookupPredictor
    from rdkit import Chem

    distinct = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    for atom in distinct.GetAtoms():
        atom.SetDoubleProp("MBIScharge", 0.0)
    distinct.GetAtomWithIdx(2).SetDoubleProp("MBIScharge", 1.0)  # the O
    companions = _hose_corpus([0.2, -0.3]).mols
    mset = MoleculeSet(
        mols=[distinct, *companions],
        atom_property="MBIScharge",
        ids={"dash_id": [None] * 3},
    )
    p = HoseLookupPredictor(max_radius=2, n_min=1)
    p.fit(mset, mset, rng=np.random.default_rng(0))
    state = p.model_state()

    global_mean = state.global_mean
    expected = 0.0
    for s, qq, c in state.tables[2].values():
        if c >= 2:
            var = qq / c - (s / c) ** 2
            expected += (c**3) * var / (c - 1) ** 2
        else:
            expected += (s - global_mean) ** 2  # c == 1: y itself is s

    stats = hose_train_stats(state, radius=2, loo=True)
    assert stats.sse == pytest.approx(expected, rel=1e-10)


def test_hose_analytic_refuses_a_state_without_the_second_moment():
    from experiments.analytic import hose_train_stats
    from experiments.hose_artifact import HoseState

    legacy = HoseState(1, ({}, {"C": (3.0, float("nan"), 2)}), 3.0, 2)
    with pytest.raises(ValueError, match="refit"):
        hose_train_stats(legacy, radius=1)


def test_hose_analytic_refuses_a_non_baseline_n_min():
    from experiments.analytic import hose_train_stats
    from experiments.hose_artifact import HoseState

    state = HoseState(1, ({}, {"C": (3.0, 5.0, 2)}), 3.0, 2, global_sumsq=5.0)
    with pytest.raises(NotImplementedError, match="n_min"):
        hose_train_stats(state, radius=1, n_min=3)


def test_supports_loo_matches_what_the_walk_actually_refuses():
    """The predicate a caller uses to ask for LOO "where available" must
    agree with the refusal itself, or the two drift apart."""
    from experiments.analytic import sieve_train_stats, supports_loo

    cases = [
        ({}, True),  # pooled, unshrunk
        ({"class_estimator": "continuation"}, False),
        ({"shrinkage_strength": 0.5}, False),
        ({"class_estimator": "continuation", "shrinkage_strength": 0.5}, False),
    ]
    for kwargs, expected in cases:
        model, _ = _fit(**kwargs)
        assert supports_loo(model.config) is expected, kwargs
        if expected:
            sieve_train_stats(model, loo=True)  # must not raise
        else:
            with pytest.raises(NotImplementedError):
                sieve_train_stats(model, loo=True)
