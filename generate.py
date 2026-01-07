
import torch
import torch.nn.functional as F
from model.decoder_only_tfm import decoder_only_tfm
import selfies as sf
from model.bi_lstm import  bi_lstm
from config.load_config import load_config
def generate_selfies(model_name, vocab, model_path=None,
                     device="cuda", max_len=80, temperature=1.0, top_k=None):
    """
    基于 SELFIES grammar 的生成器 —— 保证生成 SELFIES 合法性

    参数:
        model_name: 'decoder_only_tfm_smile_v1' (只用于选择模型类)
        vocab: SelfiesVocab instance (需包含 token2id / id2token)
        tokenizer: (可选) 若需要把 SELFIES -> SMILES，使用 selfies.sf.decoder
        model: 如果已有模型对象可以直接传入（优先）
        model_path: 如果没有 model 对象，传入权重路径将加载（strict=False）
        device: "cuda" or "cpu"
        max_len: 最大 token 数（包含 <SOS>/<EOS>）
        temperature: softmax 温度
        top_k: top-k 采样（int 或 None）

    返回:
        dict 包含:
            "selfies": 生成的 SELFIES 字符串 (tokens join)
            "smiles": 解码后的 SMILES 字符串（若能解码成功，否则 None）
    """
    # ===== 创建 / 加载模型 =====
    if model_name == 'decoder_only_tfm':
        model_cfg = load_config('./config/decoder_only_tfm_config.yaml')["model"]
        model = decoder_only_tfm(
            vocab_size=len(vocab),
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            max_len=model_cfg["max_len"],
            dropout=model_cfg["dropout"]
        ).to(device)

    elif model_name == 'bi_lstm':  # ⬅ 新增的分支
        model_cfg = load_config('./config/bi_lstm_config.yaml')["model"]
        model = bi_lstm(
            vocab_size=len(vocab),
            embed_dims=model_cfg["embed_dims"],
            hidden_dims=model_cfg["hidden_dims"],
            n_layers=model_cfg["n_layers"],
            dropout=model_cfg["dropout"],
            max_len=model_cfg["max_len"]
            ).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    # print(model_name,model_path)
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)
        # 尽量兼容加载（允许少量不匹配）
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    # ===== 预处理：semantic constraints（用于降级方案） =====
    try:
        semantic_constraints = sf.get_semantic_constraints()
    except Exception:
        semantic_constraints = None

    # helper: 给定 prefix tokens（list），返回 allowed token set（SELFIES tokens）
    def allowed_next_tokens_from_prefix(prefix_tokens):
        """
        返回一个 set of token strings（例如 '[C]', '[O]', ... 或特殊 token '<EOS>' 等）
        优先尝试 sf.next_selfies_tokens（若 available），否则退回到 semantic_constraints 基于上一个 token 的简单规则。
        """
        # prefix_selfies_str：SELFIES 格式的字符串（不含 <SOS>/<EOS>）
        # 我们需要拼接成 selfies 字符串形式（sf.split_selfies 的逆）
        # 若 prefix_tokens 包含专用标记 '<SOS>'，去掉它
        toks = [t for t in prefix_tokens if t not in ("<SOS>", "<EOS>", "<PAD>")]
        prefix_sf = "".join(toks)

        # 优先使用 next_selfies_tokens（若你的 selfies 版本提供）
        try:
            next_tokens = sf.next_selfies_tokens(prefix_sf)
            if next_tokens:
                return set(next_tokens)
        except Exception:
            pass

        # 退回：用 semantic_constraints：基于最后一个 token
        if semantic_constraints is not None:
            if len(toks) == 0:
                # 起始位置：允许所有能作为首 token 的 token
                allowed = set()
                for tok, meta in semantic_constraints.items():
                    # meta 里没有统一字段表示“可以作为首 token”，这里启发式允许所有 tokens
                    allowed.add(tok)
                return allowed
            else:
                last_tok = toks[-1]
                info = semantic_constraints.get(last_tok, {})
                # info 里一般能包含 "next"、"allowed_next" 等，尝试常见键
                nxt = info.get("next") or info.get("allowed_next") or info.get("next_tokens")
                if nxt:
                    return set(nxt)
                # 若没有返回，退回到允许所有（保守）
                return set(semantic_constraints.keys())

        # 最后退回：不做约束（不推荐）
        return set(vocab.tokens())

    # ===== 生成循环 =====
    generated_tokens = ["<SOS>"]

    for step in range(max_len):
        input_ids = torch.tensor([[vocab[tok] for tok in generated_tokens]], device=device)
        with torch.no_grad():
            logits = model(input_ids)# [1, T, V]

        logits = logits[0, -1, :]  # [V]
        logits = logits / (temperature if temperature > 0 else 1.0)

        # ----- 构建 mask -----
        allowed_tokens = allowed_next_tokens_from_prefix(generated_tokens)
        # map allowed token strings -> ids (只有存在于 vocab 的 token 才是候选)
        allowed_ids = [vocab[token] for token in allowed_tokens if token in vocab.token2id]

        if len(allowed_ids) == 0:
            # 退回到全体 vocab（避免死锁）；但通常不应该发生
            allowed_mask = torch.ones_like(logits, dtype=torch.bool, device=device)
        else:
            allowed_mask = torch.zeros_like(logits, dtype=torch.bool, device=device)
            allowed_mask[allowed_ids] = True

        # 将不合法位置置为非常小的值
        logits_masked = logits.clone()
        logits_masked[~allowed_mask] = -1e9

        probs = F.softmax(logits_masked, dim=-1)

        # Top-k sampling 可选（在 mask 后再做 topk）
        if top_k is not None:
            top_probs, top_idx = torch.topk(probs, min(top_k, probs.size(0)))
            top_probs = top_probs / (top_probs.sum() + 1e-12)
            next_id = top_idx[torch.multinomial(top_probs, 1)].item()
        else:
            next_id = torch.multinomial(probs, 1).item()

        next_token = vocab.id2token[next_id]

        # 处理特殊 token
        if next_token == "<EOS>":
            generated_tokens.append(next_token)
            break

        # 防止意外无限循环：如果 next_token 是 PAD/UNK，则跳出
        if next_token in ("<PAD>", "<UNK>"):
            # 这里选择停止生成（也可选择跳过并继续）
            break

        generated_tokens.append(next_token)

    # ===== 返回 SELFIES 字符串与对应 SMILES（若能 decode） =====
    # strip special tokens
    core_tokens = [t for t in generated_tokens if t not in ("<SOS>", "<EOS>", "<PAD>")]
    selfies_str = "".join(core_tokens)

    smiles = None
    try:
        smiles = sf.decoder(selfies_str)
    except Exception:
        smiles = None

    return {"selfies": selfies_str, "smiles": smiles}

