import torch
from tqdm import tqdm
import pandas as pd
import os
from model.decoder_only_tfm_smile_v1 import DecoderOnlyTransformer
from dataloader import SMILESTokenizer, create_vocabulary
if __name__ == "__main__":
    # ===== 配置 =====
    config_path = "./config/decoder_only_tfm_config.yaml"
    from config.load_config import load_config
    config = load_config(config_path)
    model_name = config["model_name"]
    device = config["device"]

    # ===== 数据和 vocab =====
    train_data_path = config["paths"]["data_file"]
    train_data = pd.read_csv(train_data_path)["smiles"].tolist()
    tokenizer = SMILESTokenizer()
    vocab = create_vocabulary(train_data, tokenizer)

    # ===== 初始化模型 =====
    model_cfg = config["model"]
    model = DecoderOnlyTransformer(
        vocab_size=len(vocab),
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        max_len=model_cfg["max_len"],
        dropout=model_cfg["dropout"]
    )
    model = model.to(device)

    # ===== 加载微调模型 =====
    fine_model_path = f"./model/{model_name}/{config['fine_tune']['save_name'].format(model_name=model_name)}"
    state_dict = torch.load(fine_model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded fine-tuned model from {fine_model_path}")

    # ===== 生成 SMILES =====
    num_samples = config["generate"]["num_samples"]
    temperature = config["generate"]["temperature"]
    top_k = config["generate"]["top_k"]
    max_len = config["generate"]["max_len"]

    generated_smiles = []
    for _ in tqdm(range(num_samples), desc="Generating SMILES"):
        smi = generate_smiles(model, vocab, tokenizer, device=device, max_len=max_len,
                              temperature=temperature, top_k=top_k)
        generated_smiles.append(smi)

    # ===== 保存结果 =====
    save_dir = config["paths"]["save_dir"].format(model_name=model_name)
    os.makedirs(save_dir, exist_ok=True)
    gen_path = os.path.join(save_dir, "generated_smiles.csv")
    pd.DataFrame({"generated_smiles": generated_smiles}).to_csv(gen_path, index=False, encoding="utf-8-sig")
    print(f"Generated SMILES saved to {gen_path}")