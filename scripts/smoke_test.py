import tempfile
import sys
from pathlib import Path


# 允许从项目根目录直接运行 `python scripts/smoke_test.py`。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import selfies as sf
import torch

from data.data_utils import SelfiesDataset, load_selfies_vocab, save_selfies_vocab, selfies_vocab
from model.decoder_only_tfm import decoder_only_tfm


def main():
    smiles = ["CCO", "CCN", "c1ccccc1"]
    selfies_list = [sf.encoder(smi) for smi in smiles]
    vocab = selfies_vocab(selfies_list)

    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmpdir:
        vocab_path = f"{tmpdir}/vocab.json"
        save_selfies_vocab(vocab, vocab_path)
        loaded_vocab = load_selfies_vocab(vocab_path)
        assert loaded_vocab.token2id == vocab.token2id

    dataset = SelfiesDataset(selfies_list, vocab, max_len=12)
    x, y = dataset[0]
    assert x.shape == y.shape

    model = decoder_only_tfm(
        vocab_size=len(vocab),
        d_model=32,
        n_heads=4,
        n_layers=1,
        max_len=12,
        dropout=0.1,
        pad_token_id=vocab["<PAD>"],
    )
    logits = model(x.unsqueeze(0))
    assert logits.shape == (1, x.numel(), len(vocab))
    assert torch.isfinite(logits).all()

    print("smoke_test: ok")


if __name__ == "__main__":
    main()
