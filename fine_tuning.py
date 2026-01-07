import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
from dataloader import SMILESDataset,SMILESTokenizer,create_vocabulary
from model.decoder_only_tfm import decoder_only_tfm
from config.load_config import load_config
def fine_tune(smiles_list, vocab, tokenizer, model,
              model_name="decoder_only_tfm_",
              epochs=10, batch_size=128,
              device="cuda", lr=1e-5,
              save_name="finetuned_model.pt"):
    """
    对已有模型进行微调
    """
    device = torch.device(device)
    os.makedirs(f"./model/{model_name}/", exist_ok=True)

    # 数据加载
    dataset = SMILESDataset(smiles_list, vocab, tokenizer)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model = model.to(device)
    model.train()

    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}"):
            x, y = x.long().to(device), y.long().to(device)
            optimizer.zero_grad()
            logits = model(x)

            if torch.isnan(logits).any():
                print("⚠ Warning: NaN logits detected, skipping batch")
                continue

            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Fine-tune Epoch {epoch+1} | Train Loss: {epoch_loss:.4f}")

    # 保存微调模型
    save_path = f"./model/{model_name}/{save_name}"
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuned model saved to {save_path}")

    # 保存损失曲线
    df = pd.DataFrame({
        "epoch": list(range(1, epochs + 1)),
        "train_loss": train_losses
    })
    df.to_csv(f"./model/{model_name}/finetune_loss.csv", index=False)
    print(f"Fine-tune loss curve saved to ./model/{model_name}/finetune_loss.csv")

    return model, train_losses


if __name__ =='__main__':
    # ===== 载入配置与数据 =====
    config = load_config('./config/decoder_only_tfm_config.yaml')
    model_name = config["model_name"]

    train_data = config["paths"]["data_file"]
    # train_data = pd.read_csv(train_data)["smiles"].tolist()

    data_file = config["paths"]["fine_data_file"]
    data = pd.read_csv(data_file)["selfies"].tolist()

    tokenizer = SMILESTokenizer()
    vocab = create_vocabulary(train_data, tokenizer)

    # ===== 初始化模型（使用config参数） =====
    model_cfg = config["model"]
    model = decoder_only_tfm(
        vocab_size=len(vocab),
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        max_len=model_cfg["max_len"],
        dropout=model_cfg["dropout"]
    )

    # ===== 加载预训练权重 =====
    checkpoint_path = f"./model/{model_name}/best_model_fold2.pt"
    state_dict = torch.load(checkpoint_path, map_location=config["device"])
    # model.load_state_dict(state_dict) # , strict=False
    model_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict)
    print(f" Loaded {len(filtered_state_dict)} compatible layers from {checkpoint_path}")

    # ===== 微调 =====
    fine_cfg = config["fine_tune"]
    model, fine_losses = fine_tune(
        smiles_list=data,
        vocab=vocab,
        tokenizer=tokenizer,
        model=model,
        model_name=model_name,
        epochs=fine_cfg["epochs"],
        batch_size=fine_cfg["batch_size"],
        lr=float(fine_cfg["lr"]),
        device=config["device"],
        save_name=fine_cfg["save_name"].format(model_name=model_name)
    )
