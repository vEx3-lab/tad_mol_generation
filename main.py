import os
import pandas as pd
import torch
from tqdm import tqdm

from dataloader import SMILESTokenizer, create_vocabulary
from model.decoder_only_tfm import decoder_only_tfm
from train import train_cv
from fine_tuning import fine_tune
from generate import generate_smiles
from config.load_config import load_config

# ===== 配置 =====
config_path = "./config/decoder_only_tfm_config.yaml"
config = load_config(config_path)

model_name = config["model_name"]
device = torch.device(config["device"])

# ===== 数据与 tokenizer =====
pretrain_data_file = config["paths"]["data_file"]
pretrain_smiles = pd.read_csv(pretrain_data_file)["smiles"].tolist()
# pretrain_selfies = pd.read_csv(pretrain_data_file)["selfies"].tolist()

finetune_data_file = config["paths"]["fine_data_file"]
finetune_smiles = pd.read_csv(finetune_data_file)["smiles"].tolist()
# pretrain_selfies = pd.read_csv(finetune_data_file)["selfies"].tolist()


tokenizer = SMILESTokenizer()
vocab = create_vocabulary(pretrain_smiles, tokenizer)

# ===== 保存目录 =====
save_dir = config["paths"]["save_dir"].format(model_name=model_name)
os.makedirs(save_dir, exist_ok=True)

# ===== 预训练 =====
print("=== Pretraining on large dataset ===")
model_cfg = config["model"]
pretrained_model = decoder_only_tfm(
    vocab_size=len(vocab),
    d_model=model_cfg["d_model"],
    n_heads=model_cfg["n_heads"],
    n_layers=model_cfg["n_layers"],
    max_len=model_cfg["max_len"],
    dropout=model_cfg["dropout"]
).to(device)

# 可选：K-Fold 训练
train_cv(
    smiles_list=pretrain_smiles,
    vocab=vocab,
    tokenizer=tokenizer,
    model_name=model_name,
    epochs=int(config["train"]["epochs"]),
    batch_size=int(config["train"]["batch_size"]),
    k_folds=int(config["train"]["k_folds"]),
    device=device,
    lr=float(config["train"]["lr"]),
    random_state=int(config["train"]["random_state"]),
)

# 加载最优预训练权重
best_pretrain_path = os.path.join(save_dir, "best_model_fold1.pt")
pretrained_state_dict = torch.load(best_pretrain_path, map_location=device)
pretrained_model.load_state_dict(pretrained_state_dict, strict=False)

# =====  生成 SMILES（预训练模型） =====
num_samples = config["generate"]["num_samples"]
temperature = config["generate"]["temperature"]
top_k = config["generate"]["top_k"]
max_len = config["generate"]["max_len"]

print("Generating SMILES from pretrained model...")
pretrain_generated = []
pretrained_model.eval()
with torch.no_grad():
    for _ in tqdm(range(num_samples), desc="Generating Pretrained"):
        smi = generate_smiles(
            model_name=model_name,
            vocab=vocab,
            tokenizer=tokenizer,
            model=pretrained_model,  # ⚡ 不再用 model_path
            device=device,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k
        )
        pretrain_generated.append(smi)

pretrain_gen_path = os.path.join(save_dir, "pretrain_generated_smiles.csv")
pd.DataFrame({"generated_smiles": pretrain_generated}).to_csv(pretrain_gen_path, index=False, encoding="utf-8-sig")
print(f"Pretrained SMILES saved to {pretrain_gen_path}")

# ===== 微调 =====
print("=== Fine-tuning on small dataset ===")
# 复用预训练模型对象
finetune_model = pretrained_model
fine_cfg = config["fine_tune"]
finetune_model, _ = fine_tune(
    smiles_list=finetune_smiles,
    vocab=vocab,
    tokenizer=tokenizer,
    model=finetune_model,
    model_name=model_name,
    epochs=fine_cfg["epochs"],
    batch_size=fine_cfg["batch_size"],
    lr=float(fine_cfg["lr"]),
    device=device,
    save_name=fine_cfg["save_name"].format(model_name=model_name)
)

# ===== 生成 SMILES（微调模型） =====
print("Generating SMILES from fine-tuned model...")
finetune_generated = []
finetuned_model_path = os.path.join(save_dir, fine_cfg["save_name"].format(model_name=model_name))
finetune_model.eval()
with torch.no_grad():
    for _ in tqdm(range(num_samples), desc="Generating Fine-tuned"):
        smiles = generate_smiles(
            model_name=model_name,
            vocab=vocab,
            tokenizer=tokenizer,
            model=finetune_model,  # ⚡ 不再用 model_path
            device=device,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k
        )
        finetune_generated.append(smiles)

finetune_gen_path = os.path.join(save_dir, "finetune_generated_smiles.csv")
pd.DataFrame({"generated_smiles": finetune_generated}).to_csv(finetune_gen_path, index=False, encoding="utf-8-sig")
print(f"Fine-tuned SMILES saved to {finetune_gen_path}")
