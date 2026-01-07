import os

import pandas as pd
from vina import Vina
from feedback.utils import from_smi_2_pdbqt

def batch_scores_from_vina(smiles_list, receptor_file = './8SC7.pdbqt',pdbqt_dir='./temp/pdbqt', output_dir='./vina_results/'):
    """
    批量对 SMILES 分子进行 AutoDock Vina 打分

    Args:
        smiles_list (list[str]): SMILES 字符串列表
        pdbqt_dir (str): pdbqt 临时存放目录
        output_dir (str): Vina 输出目录

    Returns:
        dict: {smile: {"score": float or None, "status": str}}
    """
    os.makedirs(pdbqt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 初始化 Vina 对象
    receptor_file =receptor_file
    v = Vina(sf_name='vina')
    v.set_receptor(receptor_file)
    v.compute_vina_maps(
        center=[24.6971, 7.5657, 58.6465],
        box_size=[79.498, 76.783, 61.505]
    )

    results = {}

    for smi in smiles_list:
        # 先生成 pdbqt
        pdbqt_path = from_smi_2_pdbqt(smi, pdbqt_dir)

        if pdbqt_path in ["invalid_smiles", "embed_failed", "obabel_failed"]:
            results[smi] = {"score": None, "status": pdbqt_path}
            continue

        ligand_name = os.path.basename(pdbqt_path)

        try:
            v.set_ligand_from_file(pdbqt_path)

            # 快速打分模式
            v.dock(exhaustiveness=2, n_poses=1)
            score = float(v.score()[0])

            # 输出 pose
            output_pdbqt = os.path.join(
                output_dir, ligand_name.replace('.pdbqt', '_out.pdbqt')
            )
            v.write_poses(output_pdbqt, n_poses=1, overwrite=True)

            results[smi] = {"score": score, "status": "success"}

        except Exception as e:
            results[smi] = {"score": None, "status": f"docking_failed: {e}"}

    return results


