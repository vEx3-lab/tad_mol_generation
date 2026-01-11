import pandas as pd
import pickle
from tqdm import tqdm
from dataloader import SMILESTokenizer, Vocabulary, create_vocabulary, SmilesDataset
# -----------------------------
# 参数配置
# -----------------------------
INPUT_CSV = "../data/htvs_molecules_with_selfies.csv"       # 输入文件
OUTPUT_PICKLE = "../data/preprocessed.pkl"  # 预处理输出文件
MAX_LEN = 120                       # 最大长度（根据数据可调整）

# -----------------------------
# 1. 读取原始数据
# -----------------------------
print("📥 Loading SMILES data ...")
df = pd.read_csv(INPUT_CSV)
smiles_list = df["smiles"].astype(str).tolist()
print(f"Loaded {len(smiles_list)} SMILES samples")

# -----------------------------
# 2. 构建分词器和词表
# -----------------------------
print("🔤 Building tokenizer & vocabulary ...")
tokenizer = SMILESTokenizer()
vocab = create_vocabulary(smiles_list, tokenizer)

print(f"Vocab size = {len(vocab)}")
print("Example tokens:", list(vocab.tokens())[:20])

# -----------------------------
# 3. 将 SMILES 转换为 token id 序列
# -----------------------------
print("🧩 Tokenizing and encoding SMILES ...")
encoded_data = []
for smi in tqdm(smiles_list):
    tokens = tokenizer.tokenize(smi)
    ids = vocab.encode(tokens)
    if len(ids) < MAX_LEN:
        ids += [vocab["<pad>"]] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    encoded_data.append(ids)

# -----------------------------
# 4. 保存预处理结果
# -----------------------------
data_dict = {
    "smiles": smiles_list,
    "encoded": encoded_data,
    "vocab_tokens": vocab.tokens(),
    "vocab_dict": vocab._tokens
}

with open(OUTPUT_PICKLE, "wb") as f:
    pickle.dump(data_dict, f)

print(f"✅ Done! Preprocessed data saved to {OUTPUT_PICKLE}")
