import pandas as pd
import selfies as sf
import ast
input_csv = "../model/decoder_only_tfm/pretrained_generated_smiles.csv"   # 你的输入文件
output_csv = "../model/decoder_only_tfm/generated_smiles.csv"   # 输出文件

df = pd.read_csv(input_csv)

generated_smiles = []

for row in df["generated_smiles"]:  #
    try:
        # 将字符串 "{'selfies': '...', 'smiles': '...'}" 转为 dict
        data = ast.literal_eval(row)

        selfies_str = data["selfies"]

        # 解码 SELFIES → SMILES
        smiles = sf.decoder(selfies_str)

    except Exception as e:
        smiles = ""  # 失败就空

    generated_smiles.append(smiles)

out_df = pd.DataFrame({"generated_smiles": generated_smiles})
out_df.to_csv(output_csv, index=False)

print("Finished! Saved to:", output_csv)
