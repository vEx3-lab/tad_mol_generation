
import torch
import torch.nn.functional as F
import selfies as sf
from feedback.vina_scores import batch_scores_from_vina
from generate_selfies import generate_selfies


def sample_selfies_batch_from_generate_selfies(
    model_name, vocab, model, model_path=None, batch_size=16,
    max_len=80, temperature=1.0, top_k=None, device="cuda"
):
    """
    调用现有 generate_selfies 函数生成 batch 数据，
    支持每次生成前用 model_path 更新参数，训练时保持 agent.train()。
    """
    import selfies as sf

    batch_token_ids = []
    batch_selfies = []
    batch_smiles = []

    # 记录原模式
    model_mode_backup = model.training

    # 暂时切换 eval() 生成
    model.eval()

    for _ in range(batch_size):
        # 每次生成前加载最新权重
        if model_path is not None:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

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

    # 恢复原模式
    if model_mode_backup:
        model.train()
    else:
        model.eval()

    return batch_token_ids, batch_selfies, batch_smiles


def compute_log_prob(model, seq_ids):
    """
    计算一条完整序列的 log πθ(a_1:T)

    参数:
        model: agent / old_agent
        seq_ids: 1D LongTensor [T+1]，包含 <SOS> ... <EOS>

    返回:
        log_prob: scalar tensor
    """
    device = next(model.parameters()).device
    seq_ids = seq_ids.to(device)

    # -------- 输入 / target 对齐 --------
    # input:  <SOS> a1 a2 ... a_{T-1}
    # target: a1    a2 ... a_T
    input_ids  = seq_ids[:-1].unsqueeze(0)   # [1, T]
    target_ids = seq_ids[1:].unsqueeze(0)    # [1, T]

    # -------- forward --------
    logits = model(input_ids)                           # [1, T, V]

    # -------- log-prob --------
    log_probs = F.log_softmax(logits, dim=-1)               # [1, T, V]

    # 取每一步真实 token 的 log-prob
    token_log_probs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1)
    ).squeeze(-1)                                           # [1, T]

    # 序列 log-prob = token log-prob 求和
    seq_log_prob = token_log_probs.sum(dim=-1)              # [1]

    return seq_log_prob.squeeze(0)



def compute_grpo_loss(
    agent,
    old_agent,
    batch_sequences,
    batch_rewards,
    clip_eps=0.2,
    kl_beta=0.05,
    eps=1e-6
):
    device = next(agent.parameters()).device

    logp_new_list = []
    logp_old_list = []
    kl_list = []

    for seq in batch_sequences:
        seq = torch.tensor(seq, dtype=torch.long, device=device)

        # 防止空序列
        if seq.size(0) < 2:
            logp_new_list.append(torch.tensor(0.0, device=device))
            logp_old_list.append(torch.tensor(0.0, device=device))
            kl_list.append(torch.tensor(0.0, device=device))
            continue

        logp_new = compute_log_prob(agent, seq)
        logp_old = compute_log_prob(old_agent, seq)

        logp_new_list.append(logp_new)
        logp_old_list.append(logp_old)
        kl_list.append(logp_old - logp_new)   # KL(old || new)

    logp_new = torch.stack(logp_new_list)   # [B]
    logp_old = torch.stack(logp_old_list)
    kl_term  = torch.stack(kl_list)

    # ---------- GRPO: group-normalized advantage ----------
    rewards = torch.tensor(batch_rewards, dtype=torch.float, device=device)
    mean = rewards.mean()
    std  = rewards.std(unbiased=False) + eps
    advantages = (rewards - mean) / std

    # ---------- PPO clipped objective ----------
    ratio = torch.exp(logp_new - logp_old.detach())

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


import copy
import torch
import torch.nn.functional as F

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
):
    agent.to(device)

    # old policy
    old_agent = copy.deepcopy(agent)
    old_agent.to(device)
    old_agent.eval()

    for epoch in range(1, epochs + 1):

        # ---------- 1. rollout (batch generation) ----------
        batch_token_ids, batch_selfies, batch_smiles = sample_selfies_batch_from_generate_selfies(
            model_name="decoder_only_tfm",
            vocab=vocab,
            model=agent,  # 使用当前 agent
            batch_size=batch_size,
            max_len=max_len,
            device=device,
            temperature=temperature,
            top_k=top_k,
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

if __name__ == '__main__':
    from config.load_config import load_config
    import os
    import pandas as pd
    from data.data_utils import selfies_vocab
    import torch.optim as optim
    from model.decoder_only_tfm import decoder_only_tfm
    # ===== 读取配置文件 =====
    config_path = "./config/decoder_only_tfm_config.yaml"
    # config_path = "config/bi_lstm_config.yaml"
    config = load_config(config_path)
    # ===== 载入数据 =====
    data_file = config["paths"]["data_file"]
    data = pd.read_csv(data_file)["selfies"].tolist()

    vocab = selfies_vocab(data)

    model_name = config["model_name"]
    device = config["device"]
    # 自动替换路径中的 {model_name}
    save_dir = config["paths"]["save_dir"].format(model_name=model_name)
    # 这里载入最后一个 fold 的最优模型
    best_model_path = os.path.join(save_dir, f"best_model_fold2.pt")
    agent = decoder_only_tfm(vocab_size=len(vocab)).to(device)

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
            batch_size=4,
            max_len=80,
            clip_eps=0.1,
            kl_beta=0.1,
            temperature=1.0,
            top_k=10,
            sync_old_every=5,
    )