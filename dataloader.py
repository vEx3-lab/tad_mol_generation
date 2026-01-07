# dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
import re

# dataloader.py

class Vocabulary:
    '''
    构建smile字符串的词汇表
    '''
    def __init__(self, tokens=None):
        self.token2id = {}
        self.id2token = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        # 添加特殊 token
        self.add_token(self.pad_token)
        self.add_token(self.unk_token)

        if tokens:
            for t in tokens:
                self.add_token(t)

    def add_token(self, token):
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

    # 新增 add 方法，让 create_vocabulary 调用没问题
    add = add_token

    def __getitem__(self, token):
        return self.token2id.get(token, self.token2id[self.unk_token])

    def get(self, token, default=None):
        if default is None:
            default = self.token2id[self.unk_token]
        return self.token2id.get(token, default)

    def __len__(self):
        return len(self.token2id)

    def decode(self, ids):
        return [self.id2token.get(i, self.unk_token) for i in ids]


class SMILESTokenizer:
    """Character-level tokenizer for SMILES"""
    REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)")
    }
    REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    def tokenize(self, smi, with_begin_and_end=True):
        def split_by(s, regexps):
            if not regexps:
                return list(s)
            r = self.REGEXPS[regexps[0]]
            parts = r.split(s)
            tokens = []
            for i, p in enumerate(parts):
                if i % 2 == 0:
                    tokens += split_by(p, regexps[1:])
                else:
                    tokens.append(p)
            return tokens
        tokens = split_by(smi, self.REGEXP_ORDER)
        if with_begin_and_end:
            tokens = ["<SOS>"] + tokens + ["<EOS>"]
        return tokens

    def untokenize(self, tokens):
        smi = ""
        for t in tokens:
            if t == "<EOS>":
                break
            if t != "<SOS>":
                smi += t
        return smi


class SMILESDataset(Dataset):
    def __init__(self, smiles_list, vocab, tokenizer, max_len=None):
        self.smiles_list = smiles_list
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len or max(len(tokenizer.tokenize(s)) for s in smiles_list)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        seq = self.smiles_list[idx]
        tokens = self.tokenizer.tokenize(seq)  # 使用 tokenize 方法
        token_ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]

        # pad 到 max_len
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab["<PAD>"]] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]

        # 训练用，输入 x 是去掉最后一个 token，y 是去掉第一个 token
        x = torch.tensor(token_ids[:-1], dtype=torch.long)
        y = torch.tensor(token_ids[1:], dtype=torch.long)
        return x, y

def get_dataloader(smiles_list, vocab, tokenizer, batch_size=32, shuffle=True):
    dataset = SMILESDataset(smiles_list, vocab, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



def create_vocabulary(smiles_list, tokenizer):
    """
    构建 Vocabulary 对象
    smiles_list: SMILES/SELFIES 序列列表
    tokenizer: SMILESTokenizer 对象
    """
    vocab = Vocabulary()  # 自动加入 <pad> 和 <unk>
    for seq in smiles_list:
        tokens = tokenizer.tokenize(seq)  # 注意这里使用 tokenize() 方法
        for t in tokens:
            vocab.add(t)
    return vocab



