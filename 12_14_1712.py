import os
import pandas as pd
from datetime import datetime
from generate_selfies import generate_selfies
import selfies as sf

from model.decoder_only_tfm import decoder_only_tfm


def sample_selfies_batch_from_generate_selfies(
    model_name, vocab, model, batch_size=16, max_len=80,
    temperature=1.0, top_k=None, device="cuda",
    save_dir=None  # 新增参数，指定保存目录
):
    """
    调用现有 generate_selfies 函数生成 batch 数据，但训练时仍保持 model.train()
    并可将生成的 SMILES 保存到 CSV 文件
    """
    batch_token_ids = []
    batch_selfies = []
    batch_smiles = []

    # 记录原本模式
    model_mode_backup = model.training
    model.eval()

    for _ in range(batch_size):
        result = generate_selfies(
            model_name=model_name,
            vocab=vocab,
            model=model,
            device=device,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k
        )

        # 转 token_ids
        token_ids = [vocab[t] for t in sf.split_selfies(result["selfies"])]

        batch_token_ids.append(token_ids)
        batch_selfies.append(result["selfies"])
        batch_smiles.append(result["smiles"])

    # 恢复原来的模式
    if model_mode_backup:
        model.train()
    else:
        model.eval()

    # ===== 保存 CSV =====
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"sampled_smiles_{timestamp}.csv")
        df = pd.DataFrame({
            "selfies": batch_selfies,
            "smiles": batch_smiles
        })
        df.to_csv(save_path, index=False)
        print(f"[Info] Saved sampled SMILES to {save_path}")

    return batch_token_ids, batch_selfies, batch_smiles
import torch
import pandas as pd
from data.data_utils import selfies_vocab
from config.load_config import load_config
import os
# ===== 读取配置文件 =====
config_path = "./config/decoder_only_tfm_config.yaml"
# config_path = "config/bi_lstm_config.yaml"
config = load_config(config_path)

model_name = config["model_name"]
device = config["device"]
# 自动替换路径中的 {model_name}
save_dir = config["paths"]["save_dir"].format(model_name=model_name)
log_dir = config["paths"]["log_dir"].format(model_name=model_name)
# ===== 载入数据 =====
data_file = config["paths"]["data_file"]
data = pd.read_csv(data_file)["selfies"].tolist()

vocab = selfies_vocab(data)
model_cfg = config['model']
# ===== SMILES 生成 =====
# 这里载入最后一个 fold 的最优模型
best_model_path = os.path.join(save_dir, f"best_model_fold2.pt")

model_path = best_model_path
print(model_path)
num_samples = config["generate"]["num_samples"]
temperature = config["generate"]["temperature"]
top_k = config["generate"]["top_k"]
max_len = config["generate"]["max_len"]
agent =  decoder_only_tfm(
            vocab_size=len(vocab),
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            max_len=model_cfg["max_len"],
            dropout=model_cfg["dropout"]
        ).to(device)
state_dict = torch.load(best_model_path, map_location=device)
# 尽量兼容加载（允许少量不匹配）
agent.load_state_dict(state_dict, strict=False)
sample_selfies_batch_from_generate_selfies(
        model = agent,
        batch_size=16,
        model_name=model_name,
        vocab=vocab,
        device=device,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k,
        save_dir='./12_14_temp_smile/' # 新增参数，指定保存目录
)



import pandas as pd
from generate import generate_selfies
from tqdm import tqdm
from data.data_utils import selfies_vocab
from config.load_config import load_config
import os
# ===== 读取配置文件 =====
config_path = "./config/decoder_only_tfm_config.yaml"
# config_path = "config/bi_lstm_config.yaml"
config = load_config(config_path)

model_name = config["model_name"]
device = config["device"]
# 自动替换路径中的 {model_name}
save_dir = config["paths"]["save_dir"].format(model_name=model_name)
log_dir = config["paths"]["log_dir"].format(model_name=model_name)
# ===== 载入数据 =====
data_file = config["paths"]["data_file"]
data = pd.read_csv(data_file)["selfies"].tolist()

vocab = selfies_vocab(data)
model_cfg = config['model']
# ===== SMILES 生成 =====
# 这里载入最后一个 fold 的最优模型
best_model_path = os.path.join(save_dir, f"best_model_fold2.pt")

model_path = best_model_path
print(model_path)
num_samples = config["generate"]["num_samples"]
temperature = config["generate"]["temperature"]
top_k = config["generate"]["top_k"]
max_len = config["generate"]["max_len"]

generated_selfies = []
generated_smiles = []

for _ in tqdm(range(16), desc='generating...'):
    result = generate_selfies(
        model_name=model_name,
        vocab=vocab,
        model_path=model_path,
        device=device,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k
    )

    generated_selfies.append(result["selfies"])
    generated_smiles.append(result["smiles"])

# 保存两种格式
gen_path = './12_14_temp_smile/gen_.csv'
pd.DataFrame({
    "selfies": generated_selfies,
    "smiles": generated_smiles
}).to_csv(gen_path, index=False)

print(f"Generated SELFIES & SMILES saved to {gen_path}")
