import pandas as pd
from rdkit import Chem
import sascorer  # 确保 sascorer.py 在当前目录或路径下


def calculate_sa_for_csv(input_file, output_file):
    # 读取 CSV
    df = pd.read_csv(input_file)

    # 假设你的 SMILES 列名为 'smiles'
    # 如果是 SELFIES，建议先转换回 SMILES 或确保列名正确
    sa_scores = []

    print("开始计算 SA Score...")
    for index, row in df.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)

        if mol:
            score = sascorer.calculateScore(mol)
            sa_scores.append(score)
        else:
            sa_scores.append(None)  # 如果分子式非法则返回空值

    # 将结果添加到新列
    df['sa_score'] = sa_scores

    # 保存结果
    df.to_csv(output_file, index=False)
    print(f"计算完成！结果已保存至: {output_file}")


# 调用函数
calculate_sa_for_csv('../model/decoder_only_tfm_best/pretrained_generated_smiles.csv', 'pretrained_generated_smiles_with_sa.csv')
calculate_sa_for_csv('../model/decoder_only_tfm_best/grpo_generated_smiles.csv', 'grpo_generated_smiles_with_sa.csv')
calculate_sa_for_csv('../data/htvs_molecules_with_selfies.csv', 'htvs_molecules_wih_sa.csv')
calculate_sa_for_csv('../data/sp_molecules_with_selfies.csv', 'sp_molecules_with_sa.csv')
