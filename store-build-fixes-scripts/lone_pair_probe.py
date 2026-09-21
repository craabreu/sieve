"""Does RDKit handle a 3-coordinate stereocentre, and can we score it from 3D?"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

print("--- does RDKit perceive 3-coordinate S stereo at all? ---")
for smi in ["C[S@](=O)c1ccccc1", "C[S@@](=O)c1ccccc1", "C[S@](=N)c1ccccc1", "C[P@](C)c1ccccc1"]:
    m = Chem.MolFromSmiles(smi)
    s = [a for a in m.GetAtoms() if a.GetSymbol() in ("S", "P")][0]
    pot = [(e.type.name, e.centeredOn) for e in Chem.FindPotentialStereo(m)]
    print(f"  {smi:26s} degree={s.GetDegree()} tag={s.GetChiralTag()} "
          f"CIP={s.GetPropsAsDict().get('_CIPCode', '-')} potential={pot}")

print("\n--- are lone pairs ever atoms in the graph? ---")
m = Chem.AddHs(Chem.MolFromSmiles("C[S@](=O)c1ccccc1"))
print("  atoms after AddHs:", sorted({a.GetSymbol() for a in m.GetAtoms()}),
      " num atoms:", m.GetNumAtoms())

print("\n--- round trip through 3D: does AssignStereochemistryFrom3D recover the tag? ---")
for smi in ["C[S@](=O)c1ccccc1", "C[S@@](=O)c1ccccc1"]:
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE)
    before = [str(a.GetChiralTag()) for a in m.GetAtoms() if a.GetSymbol() == "S"][0]
    Chem.AssignStereochemistryFrom3D(m)
    after = [str(a.GetChiralTag()) for a in m.GetAtoms() if a.GetSymbol() == "S"][0]
    # the 3-coordinate signed volume: central atom stands in for the lone pair
    s = [a for a in m.GetAtoms() if a.GetSymbol() == "S"][0]
    xyz = np.asarray(m.GetConformer().GetPositions())
    nb = sorted(n.GetIdx() for n in s.GetNeighbors())
    v = xyz[nb] - xyz[s.GetIdx()]
    vol = float(np.dot(np.cross(v[0], v[1]), v[2]))
    print(f"  {smi:22s} embedded={before:22s} from3D={after:22s} signed volume={vol:+.3f}")
