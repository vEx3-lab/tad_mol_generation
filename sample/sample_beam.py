import torch
import torch.nn.functional as F
import selfies as sf
import math


def generate_selfies_beam(model, vocab, device="cuda", max_len=80, beam_width=5, temperature=1.0, beam_temperature=1.0, noise_std=0.0):
    """
    基于 SELFIES grammar 的 Beam Search 生成器

    参数:
        model: Decoder model
        vocab: SelfiesVocab instance
        device: "cuda" or "cpu"
        max_len: 最大 token 数
        beam_width: Beam Search 宽度
        temperature: softmax 温度
        beam_temperature: Beam 内采样温度（增加多样性）
        noise_std: 添加到 logit 的噪声标准差

    返回:
        dict 包含:
            "selfies": 最佳路径的 SELFIES 字符串
            "smiles": 解码后的 SMILES 字符串
            "beams": 所有 beam 结果列表 [(selfies, smiles, score), ...]
    """
    model = model.to(device)
    model.eval()

    try:
        semantic_constraints = sf.get_semantic_constraints()
    except Exception:
        semantic_constraints = None

    def allowed_next_tokens_from_prefix(prefix_tokens):
        toks = [t for t in prefix_tokens if t not in ("<SOS>", "<EOS>", "<PAD>")]
        prefix_sf = "".join(toks)

        try:
            next_tokens = sf.next_selfies_tokens(prefix_sf)
            if next_tokens:
                return set(next_tokens)
        except Exception:
            pass

        if semantic_constraints is not None:
            if len(toks) == 0:
                allowed = set()
                for tok, meta in semantic_constraints.items():
                    allowed.add(tok)
                return allowed
            else:
                last_tok = toks[-1]
                info = semantic_constraints.get(last_tok, {})
                nxt = info.get("next") or info.get("allowed_next") or info.get("next_tokens")
                if nxt:
                    return set(nxt)
                return set(semantic_constraints.keys())

        return set(vocab.tokens())

    beams = [({"tokens": ["<SOS>"], "log_prob": 0.0})]

    for step in range(max_len):
        all_candidates = []

        for beam in beams:
            if beam["tokens"][-1] == "<EOS>":
                all_candidates.append(beam)
                continue

            input_ids = torch.tensor([[vocab[tok] for tok in beam["tokens"]]], device=device)
            logits = model(input_ids)[0, -1, :]
            logits = logits / (temperature if temperature > 0 else 1.0)

            # 添加噪声以增加多样性
            if noise_std > 0:
                noise = torch.randn_like(logits) * noise_std
                logits = logits + noise

            allowed_tokens = allowed_next_tokens_from_prefix(beam["tokens"])
            allowed_ids = [vocab[token] for token in allowed_tokens if token in vocab.token2id]

            if len(allowed_ids) == 0:
                allowed_mask = torch.ones_like(logits, dtype=torch.bool, device=device)
            else:
                allowed_mask = torch.zeros_like(logits, dtype=torch.bool, device=device)
                allowed_mask[allowed_ids] = True

            logits_masked = logits.clone()
            logits_masked[~allowed_mask] = -1e9

            # 使用 beam_temperature 控制采样多样性
            if beam_temperature != 1.0:
                logits_masked = logits_masked / beam_temperature

            probs = F.softmax(logits_masked, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, min(beam_width, probs.size(0)))

            for prob, token_id in zip(topk_probs.tolist(), topk_ids.tolist()):
                new_tokens = beam["tokens"] + [vocab.id2token[token_id]]
                new_log_prob = beam["log_prob"] + math.log(prob + 1e-12)
                all_candidates.append({
                    "tokens": new_tokens,
                    "log_prob": new_log_prob
                })

        all_candidates = [c for c in all_candidates if c["tokens"][-1] != "<SOS>"]

        if len(all_candidates) == 0:
            break

        all_candidates.sort(key=lambda x: x["log_prob"], reverse=True)
        beams = all_candidates[:beam_width]

        if all(b["tokens"][-1] == "<EOS>" for b in beams):
            break

    results = []
    for beam in beams:
        core_tokens = [t for t in beam["tokens"] if t not in ("<SOS>", "<EOS>", "<PAD>")]
        selfies_str = "".join(core_tokens)

        smiles = None
        try:
            smiles = sf.decoder(selfies_str)
        except Exception:
            smiles = None

        results.append({
            "selfies": selfies_str,
            "smiles": smiles,
            "log_prob": beam["log_prob"]
        })

    results.sort(key=lambda x: x["log_prob"], reverse=True)

    return {
        "selfies": results[0]["selfies"],
        "smiles": results[0]["smiles"],
        "beams": results
    }


def sample_selfies_batch_beam(model, vocab, batch_size=16, max_len=80, beam_width=5, temperature=1.0, beam_temperature=1.0, noise_std=0.0, device="cuda"):
    """
    Beam Search 批量采样

    参数:
        model: Decoder model
        vocab: SelfiesVocab instance
        batch_size: 生成的样本数量
        max_len: 最大 token 数
        beam_width: Beam Search 宽度
        temperature: softmax 温度
        beam_temperature: Beam 内采样温度（增加多样性）
        noise_std: 添加到 logit 的噪声标准差
        device: "cuda" or "cpu"

    返回:
        batch_smiles: SMILES 列表
    """
    batch_token_ids = []
    batch_selfies = []
    batch_smiles = []

    for _ in range(batch_size):
        result = generate_selfies_beam(
            model=model,
            vocab=vocab,
            device=device,
            max_len=max_len,
            beam_width=beam_width,
            temperature=temperature,
            beam_temperature=beam_temperature,
            noise_std=noise_std
        )

        if result["smiles"] is not None:
            selfies_str = result["selfies"]
            core_tokens = list(sf.split_selfies(selfies_str))
            token_ids = [vocab[tok] for tok in core_tokens]
            batch_token_ids.append(token_ids)
            batch_selfies.append(selfies_str)
            batch_smiles.append(result["smiles"])

    return batch_token_ids, batch_smiles


if __name__ == '__main__':
    from model.decoder_only_tfm import decoder_only_tfm
    from config.load_config import load_config
    import pandas as pd
    from data.data_utils import selfies_vocab
    from tqdm import tqdm
    import os

    config = load_config('../config/decoder_only_tfm_config.yaml')
    model_cfg = config["model"]

    data_file = config["paths"]["data_file"]
    data = pd.read_csv(data_file)["selfies"].tolist()
    device = config["device"]
    vocab = selfies_vocab(data)

    model = decoder_only_tfm(
        vocab_size=len(vocab),
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        max_len=model_cfg["max_len"],
        dropout=model_cfg["dropout"]
    ).to(device)

    model_path = '../rl/feedback/best_models/best_reward_20260405_111610.pt'
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    num_samples = config["generate"]["num_samples"]
    max_len = config["generate"]["max_len"]
    beam_width = 5
    temperature = 1.0

    generated_selfies = []
    generated_smiles = []

    for _ in tqdm(range(num_samples), desc='generating with beam search...'):
        result = generate_selfies_beam(
            model=model,
            vocab=vocab,
            device=device,
            max_len=max_len,
            beam_width=beam_width,
            temperature=temperature
        )

        generated_selfies.append(result["selfies"])
        generated_smiles.append(result["smiles"])

    gen_path = "./generated_smiles_beam.csv"
    pd.DataFrame({
        "selfies": generated_selfies,
        "smiles": generated_smiles
    }).to_csv(gen_path, index=False)
