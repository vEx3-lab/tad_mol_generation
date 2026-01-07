import os
import pandas as pd
import torch
from data.data_utils import SelfiesDataset,selfies_vocab
from model.decoder_only_tfm import decoder_only_tfm
from model.bi_lstm import bi_lstm
from train_selfies import train_cv
from config.load_config import load_config
import pickle

###########################################
#                Part 1
#   Pretrain Transformer Generator (Selfies)
###########################################
#
config_path = "./config/decoder_only_tfm_config.yaml"
config = load_config(config_path)

model_name = config["model_name"]
device = torch.device(config["device"])

# data
pretrain_data_file = config["paths"]["data_file"]
pretrain_selfies = pd.read_csv(pretrain_data_file)["selfies"].tolist()

vocab = selfies_vocab(pretrain_selfies)

# save directory
save_dir = config["paths"]["save_dir"].format(model_name=model_name)
os.makedirs(save_dir, exist_ok=True)

print("=== Pretraining Decoder-Only Transformer ===")

model_cfg = config["model"]

#  使用 model_builder 而不是直接构建模型
def model_builder_tfm():
    return decoder_only_tfm(
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
    model_builder=model_builder_tfm,   #
    epochs=int(config["train"]["epochs"]),
    batch_size=int(config["train"]["batch_size"]),
    k_folds=int(config["train"]["k_folds"]),
    device=device,
    lr=float(config["train"]["lr"]),
    random_state=config["train"]["random_state"],
)
# import pickle
#
# with open(f"./model/{model_name}/vocab.pkl", "wb") as f:
#     pickle.dump(vocab, f)


###########################################
#                Part 2
#         Pretrain Bi-LSTM Generator
###########################################

# config_path = "config/bi_lstm_config.yaml"
# config = load_config(config_path)
#
# model_name = config["model_name"]
# device = torch.device(config["device"])
#
# pretrain_data_file = config["paths"]["data_file"]
# pretrain_selfies = pd.read_csv(pretrain_data_file)["selfies"].tolist()
#
# vocab = selfies_vocab(pretrain_selfies)
#
# save_dir = config["paths"]["save_dir"].format(model_name=model_name)
# os.makedirs(save_dir, exist_ok=True)
#
# print("=== Pretraining Bi-LSTM Generator ===")
#
# model_cfg = config["model"]
#
# #
# def model_builder_lstm():
#     return bi_lstm(
#         vocab_size=len(vocab),
#         embed_dims=model_cfg["embed_dims"],
#         hidden_dims=model_cfg["hidden_dims"],
#         n_layers=model_cfg["n_layers"],
#         dropout=model_cfg["dropout"],
#         max_len=model_cfg["max_len"]
#     )
#
# train_cv(
#     seq_list=pretrain_selfies,
#     vocab=vocab,
#     model_name=model_name,
#     model_builder=model_builder_lstm,
#     epochs=int(config["train"]["epochs"]),
#     batch_size=int(config["train"]["batch_size"]),
#     k_folds=int(config["train"]["k_folds"]),
#     device=device,
#     lr=float(config["train"]["lr"]),
#     random_state=config["train"]["random_state"],
# )
#
# with open(f"./model/{model_name}/vocab.pkl", "wb") as f:
#     pickle.dump(vocab, f)
