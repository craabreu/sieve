"""S3's subset composition, classified by what actually differs geometrically."""
from __future__ import annotations
import pathlib
import json, sys, time
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import CanonicalRankAtoms
sys.path.insert(0, "experiments")
from experiments.collapse import _group_indices, _strip_stereo  # noqa: E402
from experiments.data import blob_to_mol  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rebuild_stereo_numbers import geometry_signature, potential_centres  # noqa: E402

RDLogger.DisableLog("rdApp.*")

def main(parquet, tag):
    df = pd.read_parquet(parquet, columns=["mol", "split", "collapse_key"])
    df = df[df["split"] == "test"].reset_index(drop=True)
    mols = [blob_to_mol(b) for b in df["mol"]]
    keys = [str(k) for k in df["collapse_key"]]
    stripped = [_strip_stereo(m) for m in mols]
    orders = {i: np.argsort(np.asarray(CanonicalRankAtoms(s))) for i, s in enumerate(stripped)}
    blind = _group_indices([Chem.MolToSmiles(s) for s in stripped])
    subset = {g: r for g, r in blind.items() if len({keys[i] for i in r}) > 1}

    buckets = {}
    for g, rows in subset.items():
        chirs, geos = set(), set()
        for i in rows:
            c, e = geometry_signature(mols[i], orders[i], potential_centres(stripped[i]))
            chirs.add(c); geos.add(e)
        base = next(iter(chirs))
        flipped = tuple((p, -v) for p, v in base)
        chir_differs = not (chirs <= {base, flipped})
        geo_differs = len(geos) > 1
        name = ("both" if chir_differs and geo_differs
                else "tetrahedral only" if chir_differs
                else "E/Z only" if geo_differs
                else "neither (geometrically identical)")
        b = buckets.setdefault(name, {"groups": 0, "conformers": 0})
        b["groups"] += 1; b["conformers"] += len(rows)

    total_c = sum(b["conformers"] for b in buckets.values())
    out = {"tag": tag, "subset_groups": len(subset), "subset_conformers": total_c,
           "buckets": {k: {**v, "share": v["conformers"] / total_c} for k, v in buckets.items()}}
    print(json.dumps(out, indent=2), flush=True)
    json.dump(out, open(f"/tmp/stereo-compose-{tag}.json", "w"), indent=2)

main(sys.argv[1], sys.argv[2])
