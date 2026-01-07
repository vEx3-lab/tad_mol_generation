
import torch
import torch.nn.functional as F
import selfies as sf
from feedback.vina_scores import batch_scores_from_vina
from generate_selfies import generate_selfies
# def sample_selfies_batch_grpo(
#     vocab,
#     model,
#     batch_size=16,
#     max_len=80,
#     device="cuda",
#     temperature=1.0,
#     top_k=None
# ):
#     """
#     GRPO 训练用批量 SELFIES 采样函数（v2）
#     结合原始 generate_selfies 思路，生成更高质量序列
#
#     返回:
#         batch_token_ids: List[List[int]]
#         batch_selfies:   List[str]
#         batch_smiles:    List[str | None]
#     """
#     model = model.to(device)
#     model.train()  # 保留 dropout 等随机性
#
#     # 尝试获取 semantic constraints
#     try:
#         semantic_constraints = sf.get_semantic_constraints()
#     except Exception:
#         semantic_constraints = None
#
#     def allowed_next_tokens(prefix_tokens):
#         toks = [t for t in prefix_tokens if t not in ("<SOS>", "<EOS>", "<PAD>")]
#         prefix_sf = "".join(toks)
#         try:
#             next_tokens = sf.next_selfies_tokens(prefix_sf)
#             if next_tokens:
#                 return set(next_tokens)
#         except Exception:
#             pass
#
#         if semantic_constraints:
#             if len(toks) == 0:
#                 return set(semantic_constraints.keys())
#             last_tok = toks[-1]
#             info = semantic_constraints.get(last_tok, {})
#             nxt = info.get("next") or info.get("allowed_next") or info.get("next_tokens")
#             if nxt:
#                 return set(nxt)
#             return set(semantic_constraints.keys())
#
#         return set(vocab.tokens())
#
#     batch_token_ids = []
#     batch_selfies = []
#     batch_smiles = []
#
#     sos_id = vocab["<SOS>"]
#     eos_id = vocab["<EOS>"]
#
#     for _ in range(batch_size):
#         generated_tokens = ["<SOS>"]
#         token_ids = [sos_id]
#
#         for _ in range(max_len):
#             input_ids = torch.tensor([[vocab[t] for t in generated_tokens]], device=device)
#             with torch.no_grad():
#                 logits = model(input_ids)
#             logits = logits[0, -1, :] / (temperature if temperature > 0 else 1.0)
#
#             # ----- 构建 allowed mask -----
#             allowed_tokens_set = allowed_next_tokens(generated_tokens)
#             allowed_ids = [vocab[t] for t in allowed_tokens_set if t in vocab.token2id]
#
#             if allowed_ids:
#                 mask = torch.zeros_like(logits, dtype=torch.bool, device=device)
#                 mask[allowed_ids] = True
#                 logits[~mask] = -1e9  # 不允许 token 置极小
#             # else 不 mask，允许全 vocab
#
#             probs = F.softmax(logits, dim=-1)
#
#             # top-k 采样
#             if top_k is not None:
#                 top_probs, top_idx = torch.topk(probs, min(top_k, probs.size(0)))
#                 top_probs = top_probs / (top_probs.sum() + 1e-12)
#                 next_id = top_idx[torch.multinomial(top_probs, 1)].item()
#             else:
#                 next_id = torch.multinomial(probs, 1).item()
#
#             next_token = vocab.id2token[next_id]
#
#             # 处理特殊 token
#             if next_token == "<EOS>":
#                 token_ids.append(next_id)
#                 generated_tokens.append(next_token)
#                 break
#             if next_token in ("<PAD>", "<UNK>"):
#                 continue  # 跳过，不提前结束
#
#             token_ids.append(next_id)
#             generated_tokens.append(next_token)
#
#         # 生成 SELFIES / SMILES
#         core_tokens = [t for t in generated_tokens if t not in ("<SOS>", "<EOS>", "<PAD>")]
#         selfies_str = "".join(core_tokens)
#
#         smiles = None
#         try:
#             smiles = sf.decoder(selfies_str)
#         except Exception:
#             smiles = None
#
#         batch_token_ids.append(token_ids)
#         batch_selfies.append(selfies_str)
#         batch_smiles.append(smiles)
#
#     return batch_token_ids, batch_selfies, batch_smiles
#
#
# def generate_selfies_batch(
#     vocab,  model_path=None,
#     device="cuda", batch_size=16, max_len=80,
#     temperature=1.0, top_k=None
# ):
#     """
#     基于 SELFIES grammar 的批量生成函数，返回 token_ids / selfies / smiles 列表
#     """
#
#     model = decoder_only_tfm(vocab_size=len(vocab)).to(device)
#     if model_path is not None:
#         state_dict = torch.load(model_path, map_location=device)
#         # 尽量兼容加载（允许少量不匹配）
#         model.load_state_dict(state_dict, strict=False)
#     model = model.to(device)
#     model.eval()
#
#     batch_token_ids = []
#     batch_selfies = []
#     batch_smiles = []
#
#     sos_id = vocab["<SOS>"]
#     eos_id = vocab["<EOS>"]
#
#     for _ in range(batch_size):
#         generated_tokens = ["<SOS>"]
#         token_ids = [sos_id]
#
#         for step in range(max_len):
#             input_ids = torch.tensor([[vocab[tok] for tok in generated_tokens]], device=device)
#             with torch.no_grad():
#                 if model_name == "bi_lstm":
#                     logits = model(input_ids, use_forward_only=True)
#                 else:
#                     logits = model(input_ids)
#
#             logits = logits[0, -1, :] / (temperature if temperature > 0 else 1.0)
#
#             # allowed tokens
#             toks = [t for t in generated_tokens if t not in ("<SOS>", "<EOS>", "<PAD>")]
#             prefix_sf = "".join(toks)
#             try:
#                 allowed_tokens = set(sf.next_selfies_tokens(prefix_sf))
#             except Exception:
#                 allowed_tokens = set(vocab.tokens())
#
#             allowed_ids = [vocab[t] for t in allowed_tokens if t in vocab.token2id]
#             if len(allowed_ids) == 0:
#                 allowed_mask = torch.ones_like(logits, dtype=torch.bool, device=device)
#             else:
#                 allowed_mask = torch.zeros_like(logits, dtype=torch.bool, device=device)
#                 allowed_mask[allowed_ids] = True
#
#             logits_masked = logits.clone()
#             logits_masked[~allowed_mask] = -1e9
#             probs = F.softmax(logits_masked, dim=-1)
#
#             # top-k sampling
#             if top_k is not None:
#                 top_probs, top_idx = torch.topk(probs, min(top_k, probs.size(0)))
#                 top_probs = top_probs / (top_probs.sum() + 1e-12)
#                 next_id = top_idx[torch.multinomial(top_probs, 1)].item()
#             else:
#                 next_id = torch.multinomial(probs, 1).item()
#
#             next_token = vocab.id2token[next_id]
#
#             if next_token == "<EOS>" or next_token in ("<PAD>", "<UNK>"):
#                 token_ids.append(next_id)
#                 generated_tokens.append(next_token)
#                 break
#
#             token_ids.append(next_id)
#             generated_tokens.append(next_token)
#
#         core_tokens = [t for t in generated_tokens if t not in ("<SOS>", "<EOS>", "<PAD>")]
#         selfies_str = "".join(core_tokens)
#
#         smiles = None
#         try:
#             smiles = sf.decoder(selfies_str)
#         except Exception:
#             smiles = None
#
#         batch_token_ids.append(token_ids)
#         batch_selfies.append(selfies_str)
#         batch_smiles.append(smiles)
#
#     return batch_token_ids, batch_selfies, batch_smiles
import os
import pandas as pd
import selfies as sf

def sample_selfies_batch_from_generate_selfies(
    model_name, vocab, model, batch_size=16, max_len=80, temperature=1.0, top_k=None,
    device="cuda", save_dir=None, epoch=None
):
    """
    调用现有 generate_selfies 函数生成 batch 数据，但训练时仍保持 model.train()
    并可将生成的 SMILES 保存到 CSV
    """
    batch_token_ids = []
    batch_selfies = []
    batch_smiles = []

    # 记录原本模式
    model_mode_backup = model.training

    # 暂时切换到 eval() 生成
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

    # ===== 保存到 CSV =====
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"generated_epoch{epoch}.csv" if epoch is not None else "generated.csv"
        df = pd.DataFrame({
            "selfies": batch_selfies,
            "smiles": batch_smiles
        })
        df.to_csv(os.path.join(save_dir, file_name), index=False)

    return batch_token_ids, batch_selfies, batch_smiles



import torch
import torch.nn.functional as F

def compute_log_prob_sum(model, seq_ids, pad_token_id=None):
    """
    计算整条序列的 log π(a_1:T) = sum_t log π(a_t | s_t)
    ⚠用于 PPO / GRPO ratio，不要做长度归一化
    """
    device = next(model.parameters()).device
    seq_ids = seq_ids.to(device)

    # <SOS> a1 a2 ... aT <EOS>
    input_ids  = seq_ids[:-1].unsqueeze(0)   # [1, T]
    target_ids = seq_ids[1:].unsqueeze(0)    # [1, T]

    logits = model(input_ids)                # [1, T, V]
    log_probs = F.log_softmax(logits, dim=-1)

    token_log_probs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1)
    ).squeeze(-1)                            # [1, T]

    if pad_token_id is not None:
        mask = (target_ids != pad_token_id)
        token_log_probs = token_log_probs * mask

    #  关键：sum，不是 mean
    return token_log_probs.sum(dim=-1).squeeze(0)



def compute_grpo_loss(
    agent,
    old_agent,
    batch_sequences,
    batch_rewards,
    clip_eps=0.2,
    kl_beta=0.05,
    eps=1e-6,
    pad_token_id=None
):
    device = next(agent.parameters()).device

    logp_new_list = []
    logp_old_list = []
    kl_list = []

    for seq in batch_sequences:
        seq = torch.tensor(seq, dtype=torch.long, device=device)

        if seq.size(0) < 2:
            logp_new_list.append(torch.tensor(0.0, device=device))
            logp_old_list.append(torch.tensor(0.0, device=device))
            kl_list.append(torch.tensor(0.0, device=device))
            continue

        #  用 sum log-prob
        logp_new = compute_log_prob_sum(agent, seq, pad_token_id)
        with torch.no_grad():
            logp_old = compute_log_prob_sum(old_agent, seq, pad_token_id)

        logp_new_list.append(logp_new)
        logp_old_list.append(logp_old)

        # sample-based KL(old || new)
        kl_list.append(logp_old - logp_new)

    logp_new = torch.stack(logp_new_list)   # [B]
    logp_old = torch.stack(logp_old_list)
    kl_term  = torch.stack(kl_list)

    # ---------- GRPO group-normalized advantage ----------
    rewards = torch.tensor(batch_rewards, dtype=torch.float, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)

    # ---------- PPO clipped objective ----------
    ratio = torch.exp(logp_new - logp_old)   # ⚠️ 现在这个 ratio 才是“真的”
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # ---------- KL penalty ----------
    kl_loss = kl_term.mean()

    total_loss = policy_loss + kl_beta * kl_loss
    return total_loss, policy_loss, kl_loss




def make_reward_fn_from_vina(vina_results, invalid_penalty=-10.0):
    """
    返回一个 reward_fn(smiles) -> float
    """

    def reward_fn(smiles):
        res = vina_results.get(smiles, None)

        if res is None:
            return invalid_penalty

        score = res["score"]
        status = res["status"]

        if score is None:
            return invalid_penalty

        # Vina: 越小越好 → RL: 越大越好
        return -score

    return reward_fn


import copy
import torch
import os
import pandas as pd
from datetime import datetime

def train_grpo(
    agent,
    vocab,
    optimizer,
    device="cuda",
    epochs=100,
    batch_size=16,
    max_len=80,
    clip_eps=0.2,
    kl_beta=0.05,
    temperature=1.0,
    top_k=None,
    sync_old_every=1,
    log_dir="./feedback/logs",
    save_dir="./feedback/best_models"
):
    agent.to(device)

    old_agent = copy.deepcopy(agent)
    old_agent.to(device)
    old_agent.eval()

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"grpo_training_log_{timestamp}.csv")
    pd.DataFrame(columns=["epoch", "valid_sequences", "mean_reward", "policy_loss", "kl_loss"]).to_csv(log_path, index=False)

    # ---------- 最优模型参数追踪 ----------
    best_total_loss = float("inf")
    best_total_loss_state = None
    best_reward = float("-inf")
    best_reward_state = None

    for epoch in range(1, epochs + 1):
        # ---------- 1. rollout ----------
        batch_token_ids, batch_selfies, batch_smiles = sample_selfies_batch_from_generate_selfies(
            model_name="decoder_only_tfm",
            vocab=vocab,
            model=agent,
            batch_size=batch_size,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            device=device,
            save_dir="./logs/generated_smiles",
            epoch=epoch
        )

        # ---------- 2. batch docking ----------
        vina_results = batch_scores_from_vina(
            batch_smiles,
            receptor_file='./feedback/8sc7.pdbqt',
            pdbqt_dir='./feedback/temp/pdbqt',
            output_dir='./feedback/vina_results/'
        )
        reward_fn = make_reward_fn_from_vina(vina_results)

        # ---------- 3. per-sequence reward ----------
        batch_rewards = []
        valid_sequences = []

        for seq, smi in zip(batch_token_ids, batch_smiles):
            r = reward_fn(smi)
            batch_rewards.append(r)
            valid_sequences.append(seq)

        if len(valid_sequences) < 2:
            print(f"[Epoch {epoch}] skip (too few valid samples)")
            continue

        # ---------- 4. GRPO loss ----------
        total_loss, policy_loss, kl_loss = compute_grpo_loss(
            agent=agent,
            old_agent=old_agent,
            batch_sequences=valid_sequences,
            batch_rewards=batch_rewards,
            clip_eps=clip_eps,
            kl_beta=kl_beta,
        )

        # ---------- 5. update ----------
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        # ---------- 6. sync old policy ----------
        if epoch % sync_old_every == 0:
            old_agent.load_state_dict(agent.state_dict())

        # ---------- 7. log ----------
        mean_r = sum(batch_rewards) / len(batch_rewards)
        print(
            f"[Epoch {epoch}] "
            f"valid={len(valid_sequences)} | "
            f"reward={mean_r:.3f} | "
            f"policy={policy_loss.item():.4f} | "
            f"kl={kl_loss.item():.4f}"
        )

        # ---------- 8. 保存到 CSV ----------
        df_epoch = pd.DataFrame([{
            "epoch": epoch,
            "valid_sequences": len(valid_sequences),
            "mean_reward": mean_r,
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item()
        }])
        df_epoch.to_csv(log_path, mode='a', index=False, header=False)

        # ---------- 9. 更新最优模型 ----------
        if total_loss.item() < best_total_loss:
            best_total_loss = total_loss.item()
            best_total_loss_state = copy.deepcopy(agent.state_dict())

        if mean_r > best_reward:
            best_reward = mean_r
            best_reward_state = copy.deepcopy(agent.state_dict())

    # ---------- 10. 保存最优模型 ----------
    if best_total_loss_state is not None:
        torch.save(best_total_loss_state, os.path.join(save_dir, f"best_total_loss_{timestamp}.pt"))
    if best_reward_state is not None:
        torch.save(best_reward_state, os.path.join(save_dir, f"best_reward_{timestamp}.pt"))


if __name__ == '__main__':
    import torch
    import pandas as pd
    from data.data_utils import selfies_vocab
    from config.load_config import load_config
    import os
    from model.decoder_only_tfm import decoder_only_tfm
    import torch.optim as optim
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
    agent = decoder_only_tfm(
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

    optimizer = optim.AdamW(agent.parameters(), lr=1e-4)
    train_grpo(
            agent,
            vocab,
            optimizer,
            device="cuda",
            epochs=100,
            batch_size=32,
            max_len=80,
            clip_eps=0.1,
            kl_beta=0.05,
            temperature=1.0,
            top_k=10,
            sync_old_every=5,
    )