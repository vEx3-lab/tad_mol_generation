# train_cv.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import os, json
from sklearn.model_selection import KFold
from dataloader import SMILESDataset
from dataloader import SMILESTokenizer, create_vocabulary  # 你的分词器定义文件
from model.decoder_only_tfm import decoder_only_tfm

def train_cv(smiles_list, vocab, tokenizer, model_name,
             epochs=5, batch_size=128, k_folds=3, device="cuda",
             lr=1e-3, random_state=42):
    '''version_1 :
    仅接受输入smile序列数据
    '''

    os.makedirs(f"./model/{model_name}/", exist_ok=True)
    device = torch.device(device)
    fold_results = []

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    full_dataset = SMILESDataset(smiles_list, vocab, tokenizer)

    for fold, (train_idx, val_idx) in enumerate(kf.split(smiles_list)):
        print(f"\n=== Fold {fold + 1}/{k_folds} ===")

        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = decoder_only_tfm(vocab_size=len(vocab)).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        best_model_path = f"./model/{model_name}/best_model_fold{fold + 1}.pt"

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for x, y in tqdm(train_loader, desc=f"Fold {fold+1} Train Epoch {epoch+1}"):
                x, y = x.long().to(device), y.long().to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.long().to(device), y.long().to(device)
                    logits = model(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    val_loss += loss.item()
                val_loss /= len(val_loader)

            print(f"Fold {fold+1} Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f" Saved best model for fold {fold+1}: {best_val_loss:.4f}")

        # 保存loss曲线
        pd.DataFrame({"epoch": range(1, epochs+1),
                      "train_loss": train_losses,
                      "val_loss": val_losses}
                     ).to_csv(f"./model/{model_name}/loss_fold_{fold+1}.csv", index=False)

        fold_results.append({
            "fold": fold + 1,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_val_loss": best_val_loss
        })


    # 汇总结果
    pd.DataFrame(fold_results).to_csv(f"./model/{model_name}/cv_summary.csv", index=False)
    print("\n Cross-validation complete! Summary saved to cv_summary.csv")





