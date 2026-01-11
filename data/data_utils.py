'''
此文件用于数据处理等操作
# date：2025/11/6
'''


import pandas as pd
import selfies as sf
from tqdm import tqdm
import re
from torch.utils.data import Dataset
import torch

def simle_2_selfies(input_smile_file_path, output_selfies_file_path):
    '''
    将smile数据转换为对应的selfies数据格式
    :param input_smile_file_path:
    :param output_selfies_file_path:
    :return:
    '''
    # 读取 CSV 文件
    # df = pd.read_csv(input_smile_file_path, header=None)
    # df = df['augmented']
    df = pd.read_csv(input_smile_file_path,header=None)
    selfies_list = []
    for smi in tqdm(df[0], desc="Converting SMILES to SELFIES"):
        try:
            selfies_str = sf.encoder(smi)
        except Exception as e:
            selfies_str = None
        selfies_list.append(selfies_str)

    df["selfies"] = selfies_list

    df = df.dropna(subset=["selfies"]).reset_index(drop=True)
    # 保存结果
    df.to_csv(output_selfies_file_path, index=False)


class SMILESTokenizer:
    """
    ##############      字符级别分词器smile

    Deals with the tokenization and untokenization of SMILES."""


    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)")
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string."""
        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["<SOS>"] + tokens + ["<EOS>"]
        return tokens

    def untokenize(self, tokens):
        """Untokenizes a SMILES string."""
        smi = ""
        for token in tokens:
            if token == "<EOS>":
                break
            if token != "<SOS>":
                smi += token
        return smi


# class SelfiesVocab:
#     def __init__(self, extra_specials=None):
#         # 从 selfies 获取“语义稳健”字母表（包含大多数 token） 即 字母和数字间的映射
#         # 区别去smile构建词汇表，sf库中已包含所有selfies词汇
#         alphabet = list(sf.get_semantic_robust_alphabet())
#         specials = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
#         if extra_specials:
#             specials += extra_specials
#
#         self.id2token = {}
#         self.token2id = {}
#         idx = 0
#         for s in specials:
#             self.token2id[s] = idx
#             self.id2token[idx] = s
#             idx += 1
#         for tok in alphabet:
#             if tok not in self.token2id:
#                 self.token2id[tok] = idx
#                 self.id2token[idx] = tok
#                 idx += 1
#
#     def __len__(self):
#         return len(self.token2id)
#
#     def __getitem__(self, token):
#         # 得到 token -> id（字符不存在时返回 [UNK]）
#         return self.token2id.get(token, self.token2id["<UNK>"])
#
#     def id2tok(self, idx):
#         return self.id2token.get(idx, "<UNK>")
#
#     def tokens(self):
#         return list(self.token2id.keys())


class selfies_vocab:
    def __init__(self, selfies_list, extra_specials=None):
        """
        selfies_list: 训练集中的 SELFIES 字符串列表
        """

        # ====== 1. special tokens ======
        specials = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        if extra_specials:
            specials += extra_specials

        # ====== 2. 从数据中抽取所有 token ======
        token_set = set()
        for seq in selfies_list:
            tokens = sf.split_selfies(seq)     # 核心函数！
            token_set.update(tokens)

        # ====== 3. 构造 token2id / id2token ======
        self.token2id = {}
        self.id2token = {}

        idx = 0
        for s in specials:
            self.token2id[s] = idx
            self.id2token[idx] = s
            idx += 1

        for tok in sorted(list(token_set)):   # 固定排序保证稳定性
            if tok not in self.token2id:
                self.token2id[tok] = idx
                self.id2token[idx] = tok
                idx += 1

    def __len__(self):
        return len(self.token2id)

    def __getitem__(self, token):
        """ token → id """
        return self.token2id.get(token, self.token2id["<UNK>"])

    def id2tok(self, idx):
        """ id → token """
        return self.id2token.get(idx, "<UNK>")

    def tokens(self):
        return list(self.token2id.keys())



class SelfiesDataset(Dataset):
    def __init__(self, selfies_list, vocab, max_len=None):
        self.selfies_list = selfies_list
        self.vocab = vocab
        # 计算最大长度
        self.max_len = max_len or max(len(list(sf.split_selfies(s))) + 2 for s in selfies_list)

    def __len__(self):
        return len(self.selfies_list)

    def __getitem__(self, idx):
        seq = self.selfies_list[idx]
        tokens = list(sf.split_selfies(seq))
        tokens = ["<SOS>"] + tokens + ["<EOS>"]

        token_ids = [self.vocab[t] for t in tokens]

        # pad
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab["<PAD>"]] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]

        x = torch.tensor(token_ids[:-1], dtype=torch.long)
        y = torch.tensor(token_ids[1:], dtype=torch.long)
        return x, y
