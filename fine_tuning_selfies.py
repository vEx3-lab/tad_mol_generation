# import torch
# import torch.nn as nn
# import torch.optim as optim
# from fontTools.ttx import ttList
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import pandas as pd
# import os
# from datetime import datetime
# from data.data_utils import selfies_vocab,SelfiesDataset
# from dataloader import SMILESDataset,SMILESTokenizer,create_vocabulary
# from model.decoder_only_tfm import decoder_only_tfm
# from config.load_config import load_config
#
# class FineTuner():
#     def __init__(self):
#         self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self._config = load_config('./config/decoder_only_tfm_config.yaml')
#
#         self._model_name = self._config["model_name"]
#         self._train_data_path = self._config["paths"]["data_file"]
#         self._train_data = pd.read_csv(self._train_data_path)["selfies"].tolist()
#         self._fine_data_path = self._config["paths"]["fine_data_file"]
#         self._fine_data = pd.read_csv(self._fine_data_path)["selfies"].tolist()
#
#         self._vocab = selfies_vocab(self._train_data)
#         self._d_model = self._config["model"]["d_model"]
#         self._n_heads = self._config["model"]["n_heads"]
#         self._n_layers = self._config["model"]["n_layers"]
#         self._max_len = self._config["model"]["max_len"]
#         self._dropout = self._config["model"]["dropout"]
#         self._epochs = self._config["fine_tune"]["epochs"]
#         self._batch_size = self._config["fine_tune"]["batch_size"]
#         self._lr = float(self._config["fine_tune"]["lr"])
#         self._device = self._config["device"]
#         # self._save_name =self._config["fine_tune"]["save_name"].format( model_name=self._model_name,
#         #                                                                timestamp=self._timestamp
#         #                                                                )
#
#         #  初始化模型
#         self._model = decoder_only_tfm(
#             vocab_size=len(self._vocab),
#             d_model=self._d_model,
#             n_heads=self._n_heads,
#             n_layers=self._n_layers,
#             max_len=self._max_len,
#             dropout=self._dropout
#         )
#
#         self._start_model = self._config["fine_tune"]["start_model"]
#
#     def fine_tuning(self):
#
#         device = torch.device(self._device)
#         os.makedirs(f"./model/{self._model_name}/{self._timestamp}/", exist_ok=True)
#
#         dataset = SelfiesDataset(self._fine_data, self._vocab)
#         train_loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
#
#         pad_id = self._vocab["<PAD>"]
#         criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
#
#         optimizer = optim.AdamW(self._model.parameters(), lr = self._lr, weight_decay=0.01)
#
#         model = self._model.to(device)
#         model.train()
#         state_dict = torch.load(self._start_model, map_location=device,weights_only=True)
#         model_dict = model.state_dict()
#         filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
#         model_dict.update(filtered_state_dict)
#         model.load_state_dict(model_dict)
#
#         losses = []
#         best_loss = float("inf")
#         for epoch in range(self._epochs):
#             total = 0.0
#
#             for x, y in tqdm(train_loader, desc=f"[Fine-tune {epoch+1}]"):
#                 x, y = x.to(device), y.to(device)
#
#                 optimizer.zero_grad()
#
#                 logits = model(x)
#
#                 if logits.dim() == 3 and logits.size(1) != y.size(1):
#                     logits = logits.transpose(0, 1)
#
#                 loss = criterion(
#                     logits.contiguous().view(-1, logits.size(-1)),
#                     y.contiguous().view(-1)
#                 )
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optimizer.step()
#
#                 total += loss.item()
#
#             avg = total / len(train_loader)
#             losses.append(avg)
#             if avg < best_loss:
#                 best_loss = avg
#                 torch.save(
#                     model.state_dict(),
#                     f"./model/{self._model_name}/{self._timestamp}/best.pt"
#                 )
#
#             print(f"Epoch {epoch+1}: {avg:.4f}")
#
#         torch.save(model.state_dict(), f"./model/{self._model_name}/{self._timestamp}/finetuned.pt")
#
#         pd.DataFrame({"epoch": range(1, self._epochs+1), "train_loss": losses}) \
#             .to_csv(f"./model/{self._model_name}/{self._timestamp}/finetune_loss.csv", index=False)
#
#         return model, losses
#
#
# if __name__ =='__main__':
#     finetuner = FineTuner()
#     finetuner.fine_tuning()


import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import yaml
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

from data.data_utils import selfies_vocab, SelfiesDataset
from model.decoder_only_tfm import decoder_only_tfm
from config.load_config import load_config


class FineTuner():
    def __init__(self):
        # 基础配置
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._config = load_config("./config/decoder_only_tfm_config.yaml")

        self._model_name = self._config["model_name"]
        self._device = torch.device(self._config["device"])

        # 数据
        train_data_path = self._config["paths"]["data_file"]
        fine_data_path = self._config["paths"]["fine_data_file"]

        self._train_data = pd.read_csv(train_data_path)["selfies"].tolist()
        self._fine_data = pd.read_csv(fine_data_path)["selfies"].tolist()

        # ⚠️ 强烈建议：实际使用中应直接 load pretrain vocab
        self._vocab = selfies_vocab(self._train_data)

        # 模型参数
        model_cfg = self._config["model"]
        self._d_model = model_cfg["d_model"]
        self._n_heads = model_cfg["n_heads"]
        self._n_layers = model_cfg["n_layers"]
        self._max_len = model_cfg["max_len"]
        self._dropout = model_cfg["dropout"]

        # 训练参数
        ft_cfg = self._config["fine_tune"]
        self._epochs = ft_cfg["epochs"]
        self._batch_size = ft_cfg["batch_size"]
        self._lr = float(ft_cfg["lr"])
        self._start_model = ft_cfg["start_model"]

        # 实验目录
        self._save_dir = f"./model/{self._model_name}/{self._timestamp}"
        os.makedirs(self._save_dir, exist_ok=True)

        # 初始化模型
        self._model = decoder_only_tfm(
            vocab_size=len(self._vocab),
            d_model=self._d_model,
            n_heads=self._n_heads,
            n_layers=self._n_layers,
            max_len=self._max_len,
            dropout=self._dropout,
        ).to(self._device)

    def _load_pretrained(self):
        """只加载 shape 匹配的参数，防止 vocab / head 不一致"""
        state_dict = torch.load(
            self._start_model,
            map_location=self._device,
            weights_only=True
        )

        model_dict = self._model.state_dict()
        filtered = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        model_dict.update(filtered)
        self._model.load_state_dict(model_dict)

        print(f"[INFO] Loaded {len(filtered)}/{len(model_dict)} parameters from pretrained model")

    def fine_tuning(self):

        # 数据加载
        dataset = SelfiesDataset(self._fine_data, self._vocab)
        train_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
        )

        pad_id = self._vocab["<PAD>"]
        criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

        optimizer = optim.AdamW(
            self._model.parameters(),
            lr=self._lr,
            weight_decay=0.01
        )

        # 载入预训练权重
        self._model.train()
        self._load_pretrained()

        # 训练
        best_loss = float("inf")
        losses = []

        for epoch in range(1, self._epochs + 1):
            total_loss = 0.0

            for x, y in tqdm(train_loader, desc=f"[Fine-tune {epoch}]"):
                x = x.to(self._device)
                y = y.to(self._device)

                optimizer.zero_grad()

                logits = self._model(x)  # [B, T, V]

                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            losses.append(avg_loss)

            print(f"Epoch {epoch}: {avg_loss:.4f}")

            # ===== 保存 best model =====
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    self._model.state_dict(),
                    os.path.join(self._save_dir, "best.pt")
                )

        # 保存最终模型 & 日志
        torch.save(
            self._model.state_dict(),
            os.path.join(self._save_dir, "finetuned.pt")
        )

        pd.DataFrame({
            "epoch": list(range(1, self._epochs + 1)),
            "train_loss": losses
        }).to_csv(
            os.path.join(self._save_dir, "finetune_loss.csv"),
            index=False
        )

        # 保存 config 快照（论文 & 复现必备）
        with open(os.path.join(self._save_dir, "config.yaml"), "w") as f:
            yaml.dump(self._config, f)

        # DONE 标记
        with open(os.path.join(self._save_dir, "DONE"), "w") as f:
            f.write("ok")

        return self._model, losses


if __name__ == "__main__":
    finetuner = FineTuner()
    finetuner.fine_tuning()
