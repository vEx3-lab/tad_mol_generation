import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import os
from sklearn.model_selection import KFold
from data.data_utils import SelfiesDataset
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_cv(
    seq_list,
    vocab,
    model_name,
    model_builder,         #  用 builder 生成模型，而不是传入同一个模型
    epochs=5,
    batch_size=128,
    k_folds=3,
    device="cuda",
    lr=1e-3,
    random_state=42
):

    os.makedirs(f"./model/{model_name}/", exist_ok=True)
    device = torch.device(device)

    fold_results = []

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    full_dataset = SelfiesDataset(seq_list, vocab)

    for fold, (train_idx, val_idx) in enumerate(kf.split(seq_list)):
        set_seed(random_state + fold)  # 设置随机种子

        print(f"\n=== Fold {fold+1}/{k_folds} ===")

        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,

        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        # -------------------------
        # 每个 fold 创建一个新的模型实例
        # -------------------------
        model = model_builder().to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        best_model_path = f"./model/{model_name}/best_model_fold{fold+1}.pt"

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for x, y in tqdm(train_loader, desc=f"Fold {fold+1} Train Ep {epoch+1}"):
                x, y = x.long().to(device), y.long().to(device)
                optimizer.zero_grad()

                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # eval
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.long().to(device), y.long().to(device)
                    logits = model(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"Fold {fold+1} Epoch {epoch+1} | Train {train_loss:.4f} | Val {val_loss:.4f}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"  Saved Best Model: {best_val_loss:.4f}")

        # save curve
        pd.DataFrame({
            "epoch": range(1, epochs+1),
            "train_loss": train_losses,
            "val_loss": val_losses
        }).to_csv(f"./model/{model_name}/loss_fold_{fold+1}.csv", index=False)

        fold_results.append({
            "fold": fold+1,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_val_loss": best_val_loss
        })

    pd.DataFrame(fold_results).to_csv(f"./model/{model_name}/cv_summary.csv", index=False)
    print("\nK-Fold CV Completed ")


if __name__ == '__main__':

    import os
    import pandas as pd
    import torch
    from torch import nn
    from train import train_cv  # 你的 k-fold 训练函数
    from config import load_config
    from data.data_utils import SelfiesDataset, selfies_vocab
    # from model.decoder_only_tfm_1_12 import decoder_only_tfm_1_12
    from model.v4 import decoder_only_lm  # 新版模型

    # -----------------------------
    # 配置加载
    # -----------------------------
    config_path = "../config/decoder_only_tfm_config.yaml"
    config = load_config(config_path)
    model_cfg = config["model"]

    device = torch.device(config["device"])
    model_name = config["model_name"]

    # 数据
    pretrain_data_file = config["paths"]["data_file"]
    pretrain_selfies = pd.read_csv(pretrain_data_file)["selfies"].tolist()
    vocab = selfies_vocab(pretrain_selfies)

    # 保存目录
    save_dir = config["paths"]["save_dir"].format(model_name=model_name+ 'v2')
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # 每个 fold 创建新模型的 builder
    # -----------------------------
    def model_builder_tfm():
        return decoder_only_lm(
            vocab_size=len(vocab),
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            max_len=model_cfg["max_len"],
            dropout=model_cfg["dropout"]
        )


    # run KFold pretraining
    train_cv(
        seq_list=pretrain_selfies,
        vocab=vocab,
        model_name=model_name,
        model_builder=model_builder_tfm,  #
        epochs=int(config["train"]["epochs"]),
        batch_size=int(config["train"]["batch_size"]),
        k_folds=int(config["train"]["k_folds"]),
        device=device,
        lr=float(config["train"]["lr"]),
        random_state=config["train"]["random_state"],
    )
