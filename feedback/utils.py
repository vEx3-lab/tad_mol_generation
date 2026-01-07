import tempfile, subprocess, os, hashlib
from rdkit import Chem
from rdkit.Chem import AllChem

def from_smi_2_pdbqt(smiles, output_dir='./temp/pdbqt'):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "invalid_smiles"

        mol = Chem.AddHs(mol)

        # 3D embedding
        params = AllChem.ETKDG()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            return "embed_failed"

        # force field optimization
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol)
            else:
                return "mmff_missing_params"
        except Exception as e:
            print(f"[WARN] MMFFOptimize failed: {e}")
            return "mmff_failed"

        # 临时 SDF
        sdf_name = hashlib.md5(smiles.encode()).hexdigest() + ".sdf"
        sdf_path = os.path.join(tempfile.gettempdir(), sdf_name)
        Chem.MolToMolFile(mol, sdf_path)

        # pdbqt 输出
        pdbqt_name = hashlib.md5(smiles.encode()).hexdigest() + ".pdbqt"
        output_pdbqt = os.path.join(output_dir, pdbqt_name)
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            'obabel',
            sdf_path,
            '-i', 'sdf',
            '-o', 'pdbqt',
            '-O', output_pdbqt,
            '--h',
            '-xb'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(output_pdbqt):
            return "obabel_failed"

        return output_pdbqt  # 返回生成的 pdbqt 文件路径

    except Exception as e:
        print(f"[WARN] from_smi_2_pdbqt failed for {smiles}: {e}")
        return "failed"

    finally:
        if 'sdf_path' in locals() and os.path.exists(sdf_path):
            os.remove(sdf_path)

# 测试
smi = 'CCc1ccc(-c2cc(C(=O)[O-])c3c(-c4ccc(CC)cc4)[nH]nc3n2)cc1'
print(from_smi_2_pdbqt(smi, './temp/pdbqt'))
