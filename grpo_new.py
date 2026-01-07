
import torch.nn.functional as F
from feedback.vina_scores import batch_scores_from_vina
from generate_selfies import generate_selfies
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
    print(model_mode_backup)
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


def compute_grpo_loss(
        agent,
        old_agent,
        ref_agent,
        batch_sequences,
        batch_rewards,  # 这里的输入假设是原始 reward 列表
        clip_eps=0.2,
        kl_beta=0.05,
        pad_token_id=None,
        eps=1e-8
):
    device = next(agent.parameters()).device

    # --- 1. 组内归一化 (Group Normalization) ---
    # GRPO 核心：计算当前 batch 的优势函数
    rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
    # 计算均值和标准差
    mean_r = rewards_tensor.mean()
    std_r = rewards_tensor.std() + eps
    advantages = (rewards_tensor - mean_r) / std_r

    per_seq_policy_losses = []
    per_seq_kl_losses = []

    # --- 2. 迭代计算每个序列 ---
    for i, seq in enumerate(batch_sequences):
        seq = torch.tensor(seq, dtype=torch.long, device=device)
        if seq.size(0) < 2: continue

        A = advantages[i]  # 使用归一化后的 Advantage

        input_ids = seq[:-1].unsqueeze(0)
        target_ids = seq[1:].unsqueeze(0)

        # 获取概率
        logits_new = agent(input_ids).logits if hasattr(agent(input_ids), 'logits') else agent(input_ids) # [batch,seq_len,vocab_size]
        # print(logits_new.shape)
        logits_old = old_agent(input_ids).logits.detach() if hasattr(old_agent(input_ids), 'logits') else old_agent(
            input_ids).detach()

        logp_new = F.log_softmax(logits_new, dim=-1)
        logp_old = F.log_softmax(logits_old, dim=-1)

        token_logp_new = logp_new.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logp_old = logp_old.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        mask = (target_ids != pad_token_id) if pad_token_id is not None else torch.ones_like(target_ids)

        # --- 3. 计算 Policy Loss (带负号，因为我们要最小化 Loss) ---
        ratio = torch.exp(token_logp_new - token_logp_old)
        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A

        # PPO 目标是最大化 min(surr1, surr2)，所以 Loss 是其负数
        policy_obj = -(torch.min(surr1, surr2) * mask).sum() / mask.sum()
        per_seq_policy_losses.append(policy_obj)

        # --- 4. 计算 KL 散度惩罚 ---
        # 使用准确的 KL 散度公式：KL = exp(log_ref - log_new) - (log_ref - log_new) - 1
        # 这在比率接近 1 时非常稳定
        logits_ref = ref_agent(input_ids).logits.detach() if hasattr(ref_agent(input_ids), 'logits') else ref_agent(
            input_ids).detach()
        logp_ref = F.log_softmax(logits_ref, dim=-1)
        token_logp_ref = logp_ref.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        log_ratio = token_logp_ref - token_logp_new
        kl_per_token = torch.exp(log_ratio) - log_ratio - 1
        kl_obj = (kl_per_token * mask).sum() / mask.sum()
        per_seq_kl_losses.append(kl_obj)

    # --- 5. 汇总 ---
    if not per_seq_policy_losses:
        return torch.tensor(0.0, device=device), torch.tensor(0.0), torch.tensor(0.0)

    avg_policy_loss = torch.stack(per_seq_policy_losses).mean()
    avg_kl_loss = torch.stack(per_seq_kl_losses).mean()

    # 总损失 = 负的收益 + 正的 KL 惩罚
    total_loss = avg_policy_loss + kl_beta * avg_kl_loss

    return total_loss, avg_policy_loss, avg_kl_loss




def make_reward_fn_from_vina(vina_results, invalid_penalty=0):
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


import os
import copy
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import torch.nn.utils as nn_utils


def train_grpo(
    agent,
    vocab,
    optimizer,
    device="cuda",
    epochs=100,
    batch_size=16,
    max_len=80,
    clip_eps=0.1,
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

    ref_agent = copy.deepcopy(agent)
    ref_agent.to(device)
    ref_agent.eval()

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"grpo_training_log_{timestamp}.csv")

    # -------- CSV 表头 --------
    pd.DataFrame(columns=[
        "epoch",
        "valid_sequences",
        "mean_reward",
        "top1_reward",
        "top5_mean_reward",
        "policy_loss",
        "kl_loss",
        "kl_beta"
    ]).to_csv(log_path, index=False)

    # -------- 最优模型追踪 --------
    best_total_loss = float("inf")
    best_reward = float("-inf")
    best_total_loss_state = None
    best_reward_state = None

    # -------- KL 自适应参数 --------
    for epoch in range(1, epochs + 1):

        # ========= 1. rollout =========
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

        # ========= 2. docking =========
        vina_results = batch_scores_from_vina(
            batch_smiles,
            receptor_file='./feedback/8sc7.pdbqt',
            pdbqt_dir='./feedback/temp/pdbqt',
            output_dir='./feedback/vina_results/'
        )
        reward_fn = make_reward_fn_from_vina(vina_results)

        # ========= 3. reward 计算 =========
        batch_rewards = []
        valid_sequences = []
        valid_smiles = []

        for seq, smi in zip(batch_token_ids, batch_smiles):
            r = reward_fn(smi)
            batch_rewards.append(r)
            valid_sequences.append(seq)
            valid_smiles.append(smi)

        if len(valid_sequences) < 2:
            print(f"[Epoch {epoch}] skip (too few valid samples)")
            continue

        # ========= 4. top@5 统计 =========
        reward_smi_pairs = list(zip(batch_rewards, valid_smiles))
        reward_smi_pairs.sort(key=lambda x: x[0], reverse=True)

        top5 = reward_smi_pairs[:5]
        top5_rewards = [x[0] for x in top5]

        top1_reward = top5_rewards[0]
        top5_mean_reward = float(np.mean(top5_rewards))
        mean_reward = float(np.mean(batch_rewards))

        # ========= 5. GRPO loss =========
        total_loss, policy_loss, kl_loss = compute_grpo_loss(
            agent=agent,
            old_agent=old_agent,
            ref_agent = ref_agent,
            batch_sequences=valid_sequences,
            batch_rewards=batch_rewards,
            clip_eps=clip_eps,
            kl_beta=kl_beta,
        )

        # ========= 6. update =========
        optimizer.zero_grad()
        total_loss.backward()
        nn_utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        # ========= 7. sync old policy =========
        if epoch % sync_old_every == 0:
            old_agent.load_state_dict(agent.state_dict())

        # ========= 8. 自适应 KL beta =========
        with torch.no_grad():
            kl_val = kl_loss.item()

        #     if kl_val > 1.5 * target_kl:
        #         kl_beta *= 2.0
        #     elif kl_val < target_kl / 1.5:
        #         kl_beta /= 2.0
        #
        #     kl_beta = float(np.clip(kl_beta, kl_beta_min, kl_beta_max))

        # ========= 9. log =========
        print(
            f"[Epoch {epoch:03d}] "
            f"valid={len(valid_sequences):02d} | "
            f"meanR={mean_reward:.3f} | "
            f"top5={top5_mean_reward:.3f} | "
            f"policy={policy_loss.item():.4f} | "
            f"kl={kl_val:.2f} | "
            f"kl_beta={kl_beta:.5f}"
        )

        df_epoch = pd.DataFrame([{
            "epoch": epoch,
            "valid_sequences": len(valid_sequences),
            "mean_reward": mean_reward,
            "top1_reward": top1_reward,
            "top5_mean_reward": top5_mean_reward,
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_val,
            "kl_beta": kl_beta
        }])
        df_epoch.to_csv(log_path, mode='a', index=False, header=False)

        # ========= 10. best model =========
        if total_loss.item() < best_total_loss:
            best_total_loss = total_loss.item()
            best_total_loss_state = copy.deepcopy(agent.state_dict())

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_reward_state = copy.deepcopy(agent.state_dict())

    # ========= 11. save =========
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
    # ===== 载入数据 =====
    data_file = config["paths"]["data_file"]
    data = pd.read_csv(data_file)["selfies"].tolist()

    vocab = selfies_vocab(data)
    model_cfg = config['model']
    # ===== SMILES 生成 =====
    # 这里载入最后一个 fold 的最优模型
    # best_model_path = './model/decoder_only_tfm/20251229_140125/best.pt'
    best_model_path = './model/decoder_only_tfm_best/best_model_fold2.pt'
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
    state_dict = torch.load(best_model_path, map_location=device,weights_only=True)
    # 尽量兼容加载（允许少量不匹配）
    agent.load_state_dict(state_dict, strict=False)

    optimizer = optim.AdamW(agent.parameters(), lr=1e-4)
    train_grpo(
            agent,
            vocab,
            optimizer,
            device="cuda",
            epochs=200,
            batch_size=4,
            max_len=80,
            clip_eps=0.1,
            kl_beta=0.1,
            temperature=1.0,
            top_k=20,
            sync_old_every=1
    )