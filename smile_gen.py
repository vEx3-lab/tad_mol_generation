import pandas as pd
from generate import generate_smiles
import torch
from tqdm import tqdm
from dataloader import SMILESTokenizer, create_vocabulary
from model.decoder_only_tfm import decoder_only_tfm
from config.load_config import load_config
import os
# ===== 读取配置文件 =====
config_path = "./config/decoder_only_tfm_config.yaml"
config = load_config(config_path)

model_name = config["model_name"]
device = config["device"]
# 自动替换路径中的 {model_name}
save_dir = config["paths"]["save_dir"].format(model_name=model_name)
log_dir = config["paths"]["log_dir"].format(model_name=model_name)
# ===== 载入数据 =====
data_file = config["paths"]["data_file"]
data = pd.read_csv(data_file)["smiles"].tolist()

tokenizer = SMILESTokenizer()
vocab = create_vocabulary(data, tokenizer)

# ===== SMILES 生成 =====
model_class = decoder_only_tfm
model = model_class(vocab_size=len(vocab)).to(device)

# 这里载入最后一个 fold 的最优模型
best_model_path = os.path.join(save_dir, f"decoder_only_tfm_finetuned.pt")
# model.load_state_dict(torch.load(best_model_path, map_location=device))
# model.eval()
model_path = best_model_path
num_samples = config["generate"]["num_samples"]
temperature = config["generate"]["temperature"]
top_k = config["generate"]["top_k"]
max_len = config["generate"]["max_len"]

generated_smiles = []
for _ in tqdm(range(num_samples),desc = 'generating...'):
    smi = generate_smiles(
        model_name=model_name,
        vocab=vocab,
        tokenizer=tokenizer,
        model_path=model_path,
        device=device,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k
    )
    generated_smiles.append(smi)

# 保存生成结果
gen_path = os.path.join(save_dir, "fine_generated_smiles.csv")
pd.DataFrame({"generated_smiles": generated_smiles}).to_csv(gen_path, index=False, encoding="utf-8-sig")
print(f"Generated SMILES saved to {gen_path}")
