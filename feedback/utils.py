import hashlib
import os
import subprocess
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem


def canonicalize_smiles(smiles):
    if smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def from_smi_2_pdbqt(smiles, output_dir="./temp/pdbqt", keep_pdbqt=False):
    try:
        canonical_smiles = canonicalize_smiles(smiles)
        if canonical_smiles is None:
            return "invalid_smiles"

        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            return "invalid_smiles"

        mol = Chem.AddHs(mol)

        params = AllChem.ETKDG()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            return "embed_failed"

        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol)
            else:
                return "mmff_missing_params"
        except Exception as e:
            print(f"[WARN] MMFFOptimize failed: {e}")
            return "mmff_failed"

        digest = hashlib.md5(canonical_smiles.encode("utf-8")).hexdigest()
        sdf_path = os.path.join(tempfile.gettempdir(), digest + ".sdf")
        Chem.MolToMolFile(mol, sdf_path)

        if keep_pdbqt:
            os.makedirs(output_dir, exist_ok=True)
            output_pdbqt = os.path.join(output_dir, digest + ".pdbqt")
        else:
            tmp_pdbqt = tempfile.NamedTemporaryFile(
                suffix=".pdbqt", prefix=digest + "_", delete=False
            )
            output_pdbqt = tmp_pdbqt.name
            tmp_pdbqt.close()

        if keep_pdbqt and os.path.exists(output_pdbqt):
            return output_pdbqt

        cmd = [
            "obabel",
            sdf_path,
            "-i",
            "sdf",
            "-o",
            "pdbqt",
            "-O",
            output_pdbqt,
            "--h",
            "-xb",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(output_pdbqt):
            return "obabel_failed"

        return output_pdbqt

    except Exception as e:
        print(f"[WARN] from_smi_2_pdbqt failed for {smiles}: {e}")
        return "failed"

    finally:
        if "sdf_path" in locals() and os.path.exists(sdf_path):
            os.remove(sdf_path)


if __name__ == "__main__":
    smi = "CCc1ccc(-c2cc(C(=O)[O-])c3c(-c4ccc(CC)cc4)[nH]nc3n2)cc1"
    print(from_smi_2_pdbqt(smi, "./temp/pdbqt"))
