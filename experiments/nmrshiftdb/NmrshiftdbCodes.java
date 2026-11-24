import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import org.openscience.cdk.DefaultChemObjectBuilder;
import org.openscience.cdk.config.Isotopes;
import org.openscience.cdk.interfaces.IAtom;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IBond;
import org.openscience.cdk.tools.HOSECodeGenerator;
import org.openscience.cdk.tools.manipulator.AtomContainerManipulator;
import org.openscience.nmrshiftdb.util.ExtendedHOSECodeGenerator;

/**
 * HOSE codes of every atom of a batch of molecules, generated as NMRShiftDB
 * generates them, with the CDK and the extended generator that NMRShiftDB
 * ships (revision 2699 of https://sourceforge.net/p/nmrshiftdb2/code/).
 *
 * Usage: java NmrshiftdbCodes <spheres> <query-spheres>
 *
 * For each atom, three codes, tab separated, one atom per line, in the order
 * of the atoms of each molecule:
 *
 *  1. the code NMRShiftDB stores for the atom (SubmittingData.saveMolecule):
 *     a hydrogen is coded on the molecule with its hydrogens, any other atom
 *     on the molecule with its hydrogens removed, at <spheres> spheres;
 *  2. the code it stores alongside, with ring sizes (the same method):
 *     ExtendedHOSECodeGenerator on the molecule with its hydrogens, falling
 *     back to the CDK generator with ring sizes when that one throws, as
 *     AdminTasks2.executeMakeHose does;
 *  3. the code PredictionTool.generalPredict queries with when no
 *     stereo-aware code is requested: a hydrogen is coded on the molecule with
 *     its hydrogens, any other atom on the molecule with its hydrogens made
 *     implicit, at <query-spheres> spheres. Omitted when <query-spheres> is 0.
 *
 * A code the generator cannot produce is written as an empty field, and the
 * caller treats it as matching nothing.
 *
 * Input, on stdin, one block per molecule:
 *
 *   n_atoms n_bonds
 *   symbol formal_charge aromatic        (n_atoms lines; aromatic is 0 or 1)
 *   i j order aromatic                   (n_bonds lines; 0-based atoms)
 *
 * Every hydrogen is an explicit atom, and no atom carries implicit hydrogens,
 * as on the molecules NMRShiftDB reads from its database
 * (DBMolecule.getAsCDKMolecule). The molecules carry no wedge bonds, so the
 * extended generator writes no stereo descriptors.
 */
public class NmrshiftdbCodes {

    public static void main(String[] args) throws Exception {
        int spheres = Integer.parseInt(args[0]);
        int querySpheres = Integer.parseInt(args[1]);
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        BufferedWriter out = new BufferedWriter(new OutputStreamWriter(System.out, StandardCharsets.UTF_8), 1 << 20);
        Isotopes isotopes = Isotopes.getInstance();
        String line;
        while ((line = in.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;
            String[] head = line.split(" ");
            int nAtoms = Integer.parseInt(head[0]);
            int nBonds = Integer.parseInt(head[1]);
            IAtomContainer mol = DefaultChemObjectBuilder.getInstance().newInstance(IAtomContainer.class);
            for (int i = 0; i < nAtoms; i++) {
                String[] f = in.readLine().trim().split(" ");
                IAtom atom = isotopes.configure(new org.openscience.cdk.Atom(f[0]));
                atom.setMassNumber(null);
                atom.setFormalCharge(Integer.parseInt(f[1]));
                if (f[2].equals("1")) atom.setIsAromatic(true);
                atom.setImplicitHydrogenCount(0);
                mol.addAtom(atom);
            }
            for (int b = 0; b < nBonds; b++) {
                String[] f = in.readLine().trim().split(" ");
                IBond bond = DefaultChemObjectBuilder.getInstance().newInstance(IBond.class,
                        mol.getAtom(Integer.parseInt(f[0])), mol.getAtom(Integer.parseInt(f[1])),
                        order(Integer.parseInt(f[2])));
                if (f[3].equals("1")) bond.setIsAromatic(true);
                mol.addBond(bond);
            }
            writeCodes(mol, spheres, querySpheres, out);
        }
        out.flush();
    }

    static IBond.Order order(int order) {
        // AtomUtils.getOrderFromInt
        return order == 0 ? IBond.Order.ZERO : order == 1 ? IBond.Order.SINGLE
                : order == 2 ? IBond.Order.DOUBLE : order == 3 ? IBond.Order.TRIPLE : IBond.Order.QUADRUPLE;
    }

    /** The molecule with its hydrogens removed and counted as implicit on
     * their neighbors, as AtomUtils.convertImplicitToExplicitHydrogens does
     * (despite its name) for PredictionTool. The heavy atoms keep their
     * relative order. */
    static IAtomContainer hydrogensImplicit(IAtomContainer mol) throws Exception {
        IAtomContainer copy = mol.clone();
        for (IAtom atom : copy.atoms()) {
            if (atom.getImplicitHydrogenCount() == null) atom.setImplicitHydrogenCount(0);
        }
        List<IAtom> hydrogens = new ArrayList<>();
        for (IAtom atom : copy.atoms()) {
            if (atom.getSymbol().equals("H")) {
                IAtom neighbor = copy.getConnectedAtomsList(atom).get(0);
                neighbor.setImplicitHydrogenCount(neighbor.getImplicitHydrogenCount() + 1);
                hydrogens.add(atom);
            }
        }
        for (IAtom atom : hydrogens) copy.removeAtomAndConnectedElectronContainers(atom);
        return copy;
    }

    static void writeCodes(IAtomContainer mol, int spheres, int querySpheres, BufferedWriter out)
            throws Exception {
        HOSECodeGenerator hcg = new HOSECodeGenerator();
        ExtendedHOSECodeGenerator ehcg = new ExtendedHOSECodeGenerator();
        IAtomContainer storedWithout = AtomContainerManipulator.removeHydrogens(mol.clone());
        IAtomContainer queryWithout = querySpheres > 0 ? hydrogensImplicit(mol) : null;
        int heavy = 0;
        for (int i = 0; i < mol.getAtomCount(); i++) {
            IAtom atom = mol.getAtom(i);
            boolean hydrogen = atom.getSymbol().equals("H");
            String stored;
            String ring;
            String query = null;
            try {
                stored = hydrogen ? hcg.getHOSECode(mol, atom, spheres, false)
                        : hcg.getHOSECode(storedWithout, storedWithout.getAtom(heavy), spheres, false);
            } catch (Exception ex) {
                stored = "";
            }
            try {
                ring = ehcg.getHOSECode(mol, atom, spheres, true);
            } catch (Exception ex) {
                try {
                    ring = hcg.getHOSECode(mol, atom, spheres, true);
                } catch (Exception ex2) {
                    ring = "";
                }
            }
            if (querySpheres > 0) {
                try {
                    query = hydrogen ? hcg.getHOSECode(mol, atom, querySpheres, false)
                            : hcg.getHOSECode(queryWithout, queryWithout.getAtom(heavy), querySpheres, false);
                } catch (Exception ex) {
                    query = "";
                }
            }
            if (!hydrogen) heavy++;
            out.write(stored);
            out.write('\t');
            out.write(ring);
            if (query != null) {
                out.write('\t');
                out.write(query);
            }
            out.write('\n');
        }
    }
}
