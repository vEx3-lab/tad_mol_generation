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

from feedback.vina_scores import batch_scores_from_vina
vina_results = batch_scores_from_vina(generated_smiles ,receptor_file = './feedback/8sc7.pdbqt',pdbqt_dir='./feedback/temp/pdbqt', output_dir='./feedback/vina_results/')

