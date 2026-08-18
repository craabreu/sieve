"""The fitted model and the fit entry point (design.md 5.1, 7, 10.1)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from sieve.batch import AtomBatch
from sieve.config import SieveConfig
from sieve.level import FrozenLevel, fit_level, global_stats
from sieve.refine import refine


@dataclass(frozen=True)
class SieveModel:
    """An immutable, exactly sized fitted model (design.md 5.1).

    There is no mutable accumulator and no ``partial_fit``: incremental
    training is ``model.merge(fit(batch))``, and parallel fitting is a fold
    over independently fitted shards.
    """

    config: SieveConfig
    levels: tuple[FrozenLevel, ...]
    global_count: int
    global_mean: np.ndarray
    global_msd: np.ndarray

    @classmethod
    def empty(cls, config: SieveConfig) -> SieveModel:
        """The identity of the merge monoid (design.md 5.4)."""
        d = config.target_dim
        levels = tuple(
            FrozenLevel(
                np.zeros((0, 1), np.int64),
                np.zeros(0, np.int64),
                np.zeros((0, d)),
                np.zeros((0, d)),
                np.zeros(0, np.int32),
            )
            for _ in range(config.n_levels)
        )
        return cls(config, levels, 0, np.zeros(d), np.zeros(d))

    def with_params(self, **kw) -> SieveModel:
        """A new model sharing the same arrays, with inference params changed.

        ``n_min`` and ``alpha`` are read at prediction time, so sweeping them
        never requires refitting.
        """
        bad = set(kw) - {"n_min", "alpha", "chunk_size"}
        if bad:
            raise ValueError(f"with_params only changes inference params, got {bad}")
        return replace(self, config=replace(self.config, **kw))

    def merge(self, other: SieveModel) -> SieveModel:
        """Combine two models. Named `merge` because `a + b` reads as ensembling."""
        from sieve.merge import merge_models

        return merge_models(self, other)

    def __add__(self, other):
        """Ergonomic alias so `sum(models, SieveModel.empty(cfg))` works."""
        if other == 0:
            return self
        return self.merge(other)

    __radd__ = __add__

    def predict(self, batch: AtomBatch) -> np.ndarray:
        from sieve.predict import predict as _predict

        return _predict(self, batch)

    def predict_detailed(self, batch: AtomBatch):
        from sieve.predict import predict_detailed as _predict_detailed

        return _predict_detailed(self, batch)

    def predict_loo(self, batch: AtomBatch):
        from sieve.predict import predict_loo as _predict_loo

        return _predict_loo(self, batch)

    def save(self, path) -> None:
        """Write a single .npz (design.md 9.1).

        Shrunk estimates are deliberately absent: they are derived, and storing
        them would let alpha drift out of sync with the values it produced.
        """
        import json

        from sieve.config import FORMAT_VERSION

        cfg = self.config
        blob = {
            "format_version": FORMAT_VERSION,
            "schema_version": cfg.schema_version,
            "target_dim": cfg.target_dim,
            "attribute_levels": [list(g) for g in cfg.attribute_levels],
            "attribute_codes": {k: dict(v) for k, v in cfg.attribute_codes.items()},
            "edge_codes": dict(cfg.edge_codes),
            "max_wl_depth": cfg.max_wl_depth,
            "neighbour_schema": cfg.neighbour_schema,
            "n_min": cfg.n_min,
            "alpha": cfg.alpha,
            "chunk_size": cfg.chunk_size,
        }
        arrays = {
            "config": np.frombuffer(json.dumps(blob).encode(), np.uint8),
            "global": np.concatenate(
                [[float(self.global_count)], self.global_mean, self.global_msd]
            ),
        }
        for k, lvl in enumerate(self.levels):
            arrays[f"level_{k}_vocab"] = lvl.signatures
            arrays[f"level_{k}_count"] = lvl.count
            arrays[f"level_{k}_mean"] = lvl.mean
            arrays[f"level_{k}_msd"] = lvl.msd
            arrays[f"level_{k}_parent"] = lvl.parent
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path) -> SieveModel:
        import json

        from sieve.config import FORMAT_VERSION, SieveConfig
        from sieve.level import FrozenLevel

        data = np.load(path, allow_pickle=False)
        blob = json.loads(bytes(data["config"]).decode())
        if blob["format_version"] != FORMAT_VERSION:
            raise ValueError(
                f"unsupported format_version {blob['format_version']}; "
                f"this build reads {FORMAT_VERSION}. Refusing to guess."
            )
        cfg = SieveConfig(
            target_dim=blob["target_dim"],
            attribute_levels=tuple(tuple(g) for g in blob["attribute_levels"]),
            attribute_codes=blob["attribute_codes"],
            edge_codes=blob["edge_codes"],
            max_wl_depth=blob["max_wl_depth"],
            neighbour_schema=(
                None
                if blob["neighbour_schema"] is None
                else tuple(blob["neighbour_schema"])
            ),
            n_min=blob["n_min"],
            alpha=blob["alpha"],
            chunk_size=blob["chunk_size"],
        )
        if cfg.schema_version != blob["schema_version"]:
            raise ValueError("schema_version does not match the stored config")
        d = cfg.target_dim
        g = data["global"]
        levels = tuple(
            FrozenLevel(
                data[f"level_{k}_vocab"],
                data[f"level_{k}_count"],
                data[f"level_{k}_mean"],
                data[f"level_{k}_msd"],
                data[f"level_{k}_parent"],
            )
            for k in range(cfg.n_levels)
        )
        return cls(cfg, levels, int(g[0]), g[1 : 1 + d], g[1 + d : 1 + 2 * d])


def fit(batch: AtomBatch, config: SieveConfig) -> SieveModel:
    """Fit a model to one corpus.

    When ``config.chunk_size`` is set the batch is fitted in pieces and folded,
    which is the whole of the "streaming" story: chunk size is a memory
    decision, and combining chunks is the merge of design.md 5.2.
    """
    if batch.y is None:
        raise ValueError("fit requires targets; batch.y is None")
    if batch.y.shape[1] != config.target_dim:
        raise ValueError(
            f"target_dim is {config.target_dim} but batch.y has "
            f"{batch.y.shape[1]} columns"
        )

    if config.chunk_size is not None and config.chunk_size < batch.n_atoms:
        from sieve.merge import fold

        graphs = np.unique(batch.graph_id)
        n_parts = max(1, -(-batch.n_atoms // config.chunk_size))
        shards = []
        for part in np.array_split(graphs, n_parts):
            mask = np.isin(batch.graph_id, part)
            shards.append(
                fit(_sub_batch(batch, mask), replace(config, chunk_size=None))
            )
        return fold(shards, config)

    levels_lbl = refine(batch, config)
    levels = tuple(fit_level(lv, batch.y) for lv in levels_lbl)
    n, mean, msd = global_stats(batch.y)
    return SieveModel(config, levels, n, mean, msd)


def _sub_batch(batch: AtomBatch, mask: np.ndarray) -> AtomBatch:
    """Atoms where `mask` is True, with edges reindexed onto the subset."""
    idx = np.flatnonzero(mask)
    remap = np.full(batch.n_atoms, -1, np.int64)
    remap[idx] = np.arange(idx.size)
    keep = mask[batch.edge_src] & mask[batch.edge_dst]
    return AtomBatch(
        node_attrs=batch.node_attrs[idx],
        edge_src=remap[batch.edge_src[keep]],
        edge_dst=remap[batch.edge_dst[keep]],
        edge_attr=batch.edge_attr[keep],
        graph_id=batch.graph_id[idx],
        y=None if batch.y is None else batch.y[idx],
        elements=None if batch.elements is None else batch.elements[idx],
    )
