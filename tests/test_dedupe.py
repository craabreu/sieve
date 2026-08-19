import numpy as np

import sieve.dedupe
from sieve.dedupe import dense_rows


def test_identical_rows_share_an_id():
    m = np.array([[1, 2], [3, 4], [1, 2], [5, 6]], np.int64)
    labels, uniq = dense_rows(m)
    assert labels[0] == labels[2]
    assert labels[0] != labels[1]
    assert uniq.shape[0] == 3


def test_labels_are_dense_from_zero():
    m = np.array([[9, 9], [4, 4], [9, 9], [1, 1]], np.int64)
    labels, _uniq = dense_rows(m)
    assert sorted(set(labels.tolist())) == [0, 1, 2]
    assert labels.dtype == np.int64


def test_unique_rows_are_indexed_by_label():
    """The representative row for class j must actually be a row of class j.

    This is the assertion that catches the np.unique return-order trap: if
    `index` and `inverse` are swapped, uniq[labels[i]] stops equalling m[i].
    """
    rng = np.random.default_rng(0)
    m = rng.integers(0, 5, size=(500, 4)).astype(np.int64)
    labels, uniq = dense_rows(m)
    assert np.array_equal(uniq[labels], m)


def test_matches_a_dict_based_reference():
    rng = np.random.default_rng(1)
    m = rng.integers(0, 3, size=(200, 3)).astype(np.int64)
    labels, _ = dense_rows(m)
    seen, ref = {}, []
    for row in map(tuple, m):
        ref.append(seen.setdefault(row, len(seen)))
    # ids may be numbered differently, but the partition must be identical
    assert _same_partition(labels, np.array(ref))


def _same_partition(a, b):
    return (
        len({(int(x), int(y)) for x, y in zip(a, b, strict=True)})
        == len(set(a.tolist()))
        == len(set(b.tolist()))
    )


def test_the_same_input_always_produces_the_same_ids():
    """Ids must be a deterministic function of the input.

    Independently fitted shards are merged by looking signatures up against
    each other (design.md 5.3), so two runs over the same rows have to agree.
    """
    rng = np.random.default_rng(7)
    m = rng.integers(0, 4, size=(40, 3)).astype(np.int64)
    first_labels, first_uniq = dense_rows(m)
    second_labels, second_uniq = dense_rows(m.copy())
    assert first_labels.tolist() == second_labels.tolist()
    assert np.array_equal(first_uniq, second_uniq)
    assert first_uniq.shape[0] == 31


def test_ids_are_not_ordered_by_row_content():
    """Pins the *absence* of an ordering guarantee.

    Grouping is by a 64-bit row key, so ids come out in key order. Nothing may
    depend on class 0 holding the smallest row; this test exists so that a
    change quietly reintroducing that assumption gets noticed.
    """
    rng = np.random.default_rng(7)
    m = rng.integers(0, 4, size=(40, 3)).astype(np.int64)
    _labels, uniq = dense_rows(m)
    ascending = all(list(uniq[i]) < list(uniq[i + 1]) for i in range(uniq.shape[0] - 1))
    assert not ascending


def test_a_heavily_duplicated_row_keeps_its_representative():
    """Every row of a class is byte-identical, so recovering representatives by
    scattering all of them must land the same values as picking the first."""
    m = np.tile(np.array([[7, 7, 7]], np.int64), (1000, 1))
    m[500] = [1, 2, 3]
    labels, uniq = dense_rows(m)
    assert uniq.shape[0] == 2
    assert np.array_equal(uniq[labels], m)


def test_handles_non_contiguous_input():
    m = np.arange(40, dtype=np.int64).reshape(10, 4)[:, ::2]
    labels, uniq = dense_rows(m)
    assert np.array_equal(uniq[labels], m)


def test_a_hash_collision_falls_back_instead_of_merging_classes(monkeypatch):
    """The failure mode the fallback exists for.

    Grouping by a 64-bit key is only a *guess* that equal keys mean equal rows.
    A collision would silently fuse two distinct environments into one class --
    no crash, just a quietly wrong model. Forcing every row onto one key proves
    the verification catches it and the exact path takes over.
    """
    monkeypatch.setattr(
        sieve.dedupe, "_row_keys", lambda m: np.zeros(m.shape[0], np.uint64)
    )
    m = np.array([[1, 2], [3, 4], [1, 2], [5, 6]], np.int64)
    labels, uniq = dense_rows(m)
    assert uniq.shape[0] == 3, "collided rows must not be fused into one class"
    assert np.array_equal(uniq[labels], m)
    assert labels[0] == labels[2] and labels[0] != labels[1]


def test_a_partial_hash_collision_falls_back(monkeypatch):
    """Only *some* rows colliding is the realistic case, and the harder one:
    the verification has to notice a single fused pair among correct groups."""
    real = sieve.dedupe._row_keys

    def collide_two(m):
        keys = real(m).copy()
        keys[m[:, 0] % 2 == 1] = np.uint64(0)  # fuse every odd-first-column row
        return keys

    monkeypatch.setattr(sieve.dedupe, "_row_keys", collide_two)
    rng = np.random.default_rng(3)
    m = rng.integers(0, 6, size=(300, 3)).astype(np.int64)
    labels, uniq = dense_rows(m)
    assert np.array_equal(uniq[labels], m)
    assert uniq.shape[0] == len({tuple(r) for r in m.tolist()})


def test_row_keys_match_a_recorded_baseline():
    """Keys must be a pure function of the bytes -- identical in every process.

    Independently fitted shards are combined by matching signatures against
    each other, so a worker process that keyed rows differently from its parent
    would mint duplicate classes rather than merge them. Anything seed- or
    address-dependent (``hash()`` being the obvious trap) fails this, as does
    any unintended change to the mixing.
    """
    m = np.ascontiguousarray(np.arange(24, dtype=np.int64).reshape(8, 3))
    assert [int(x) for x in sieve.dedupe._row_keys(m)] == [
        9276827215579975395,
        752523549488575333,
        16189849825120920578,
        14145680651066150495,
        6081766414869024413,
        109111657846721951,
        10589562359658575017,
        11685874297919304909,
    ]


def test_low_bits_of_the_key_are_not_linear_in_the_input():
    """Without a finalizer, bit 0 of an FNV-over-8-byte-lanes key is exactly
    the XOR of bit 0 of every column -- a linear relation that involves more
    columns as rows get wider. It does not by itself cause collisions (all 64
    bits still have to agree) but it puts the usable entropy under 64 bits."""
    rng = np.random.default_rng(11)
    m = np.ascontiguousarray(rng.integers(0, 2**20, size=(50_000, 5)).astype(np.int64))
    h = sieve.dedupe._row_keys(m)

    linear = np.full(m.shape[0], sieve.dedupe._FNV_OFFSET & np.uint64(1), np.uint64)
    for j in range(m.shape[1]):
        linear ^= m[:, j].view(np.uint64) & np.uint64(1)
    assert not np.array_equal(h & np.uint64(1), linear)


def test_flipping_one_input_bit_changes_about_half_the_key_bits():
    """Strict avalanche: a one-bit input change should look like a fresh key."""
    rng = np.random.default_rng(12)
    m = np.ascontiguousarray(rng.integers(0, 2**40, size=(20_000, 5)).astype(np.int64))
    base = sieve.dedupe._row_keys(m)
    for col, bit in ((0, 0), (4, 0), (2, 37)):
        flipped = m.copy()
        flipped[:, col] ^= np.int64(1) << np.int64(bit)
        changed = base ^ sieve.dedupe._row_keys(np.ascontiguousarray(flipped))
        mean_bits = np.mean([bin(int(x)).count("1") for x in changed[:2000]])
        assert 28 < mean_bits < 36, (
            f"flipping column {col} bit {bit} changed {mean_bits:.1f} of 64 "
            f"key bits; a well-mixed key changes ~32"
        )


def test_non_int64_rows_are_deduplicated_correctly():
    """The key mixer reinterprets 8-byte lanes; narrower dtypes must still work."""
    m = np.array([[1, 2], [3, 4], [1, 2]], np.int32)
    labels, uniq = dense_rows(m)
    assert np.array_equal(uniq[labels], m)
    assert uniq.shape[0] == 2
    assert labels[0] == labels[2]
