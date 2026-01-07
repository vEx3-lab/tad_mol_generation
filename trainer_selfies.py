
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

class trainer():
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

        self._vocab = selfies_vocab(self._train_data)

        # 模型参数
        model_cfg = self._config["model"]
        self._d_model = model_cfg["d_model"]
        self._n_heads = model_cfg["n_heads"]
        self._n_layers = model_cfg["n_layers"]
        self._max_len = model_cfg["max_len"]
        self._dropout = model_cfg["dropout"]

        # 训练参数
        train_cfg = self._config["train"]
        self._epochs = train_cfg["epochs"]
        self._batch_size = train_cfg["batch_size"]
        self._lr = float(train_cfg["lr"])

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