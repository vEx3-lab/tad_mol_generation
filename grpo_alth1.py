# ===== GRPO 批量训练脚本 (Decoder-Only Transformer) =====
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import selfies as sf
from datetime import datetime
from generate_selfies import generate_selfies
from feedback.vina_scores import batch_scores_from_vina


# ----------------- 1. rollout（π_old 采样） -----------------
def sample_selfies_batch_from_generate_selfies(
    model_name,
    vocab,
    model,
    batch_size=16,
    max_len=80,
    temperature=1.0,
    top_k=None,
    device="cuda",
):
    model.eval()
    batch_token_ids, batch_smiles = [], []

    with torch.no_grad():
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
            tokens = sf.split_selfies(result["selfies"])
            token_ids = [vocab[t] for t in tokens]

            if len(token_ids) >= 2:
                batch_token_ids.append(token_ids)
                batch_smiles.append(result["smiles"])

    return batch_token_ids, batch_smiles


# ----------------- 2. GRPO loss（token-level PPO） -----------------
def compute_grpo_loss_batch(
    agent,
    old_agent,
    ref_agent,
    batch_sequences,
    batch_rewards,
    clip_eps=0.2,
    kl_beta=0.05,
    pad_token_id=0,
    eps=1e-8
):
    """
    GRPO token-level loss (PPO + KL近似)
    1. policy loss: token-level clipped surrogate
    2. KL loss: D_KL ≈ pi_ref/pi_theta - log(pi_ref/pi_theta) - 1
    返回:
        total_loss, avg_policy_loss, avg_kl_loss, ratio_mean
    """
    device = next(agent.parameters()).device
    B = len(batch_sequences)

    # --- 1. advantages ---
    rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
    adv = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + eps)
    adv = adv.detach()

    per_seq_policy_losses = []
    per_seq_kl_losses = []
    ratio_means = []

    for i, seq in enumerate(batch_sequences):
        seq = torch.tensor(seq, dtype=torch.long, device=device)
        if seq.size(0) < 2:
            continue

        A = adv[i]

        input_ids = seq[:-1].unsqueeze(0)   # [1, seq_len-1]
        target_ids = seq[1:].unsqueeze(0)   # [1, seq_len-1]

        # ---- logits ----
        logits_new = agent(input_ids).logits if hasattr(agent(input_ids), 'logits') else agent(input_ids)
        logits_old = old_agent(input_ids).logits.detach() if hasattr(old_agent(input_ids), 'logits') else old_agent(input_ids).detach()
        logits_ref = ref_agent(input_ids).logits.detach() if hasattr(ref_agent(input_ids), 'logits') else ref_agent(input_ids).detach()

        # ---- log probabilities ----
        logp_new = F.log_softmax(logits_new, dim=-1)
        logp_old = F.log_softmax(logits_old, dim=-1)
        logp_ref = F.log_softmax(logits_ref, dim=-1)

        # ---- gather target token probs ----
        token_logp_new = logp_new.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logp_old = logp_old.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logp_ref = logp_ref.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        # ---- mask ----
        mask = torch.ones_like(token_logp_new)
        if pad_token_id is not None:
            mask = (target_ids != pad_token_id).float()

        # ---- policy loss (clipped PPO) ----
        ratio = torch.exp(token_logp_new - token_logp_old)
        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A
        policy_loss = -((torch.min(surr1, surr2) * mask).sum() / mask.sum())
        per_seq_policy_losses.append(policy_loss)

        # ---- KL loss (approximation) ----
        log_ratio = token_logp_ref - token_logp_new
        kl_per_token = torch.exp(log_ratio) - log_ratio - 1
        kl_loss = (kl_per_token * mask).sum() / mask.sum()
        per_seq_kl_losses.append(kl_loss)

        # ---- ratio mean ----
        ratio_means.append(ratio.mean().detach())

    if not per_seq_policy_losses:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)

    avg_policy_loss = torch.stack(per_seq_policy_losses).mean()
    avg_kl_loss = torch.stack(per_seq_kl_losses).mean()
    ratio_mean = torch.stack(ratio_means).mean()

    total_loss = avg_policy_loss + kl_beta * avg_kl_loss

    return total_loss, avg_policy_loss, avg_kl_loss, ratio_mean

# ----------------- 3. reward 函数 -----------------
def make_reward_fn_from_vina(vina_results, invalid_penalty=0):
    def reward_fn(smiles):
        res = vina_results.get(smiles, None)
        if res is None:
            return invalid_penalty
        score = res.get("score", None)
        if score is None:
            return invalid_penalty
        return -score
    return reward_fn


# ----------------- 4. 训练主循环（GRPO） -----------------
def train_grpo(
        agent,
        vocab,
        optimizer,
        device="cuda",
        iterations=100,  # 减少外循环次数
        M=4,  #  新增：中循环步数
        batch_size=8,
        max_len=80,
        mu=4,
        clip_eps=0.1,
        kl_beta=0.1,
        temperature=1.0,
        top_k=10,
        log_dir="./feedback/logs",
        save_dir="./feedback/best_models"
):
    agent.to(device)
    old_agent = copy.deepcopy(agent).to(device)
    ref_agent = copy.deepcopy(agent).to(device)

    old_agent.eval()
    ref_agent.eval()

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"grpo_training_log_{timestamp}.csv")

    # CSV列：添加step列
    if not os.path.exists(log_path):
        pd.DataFrame(columns=[
            "iteration", "step", "mean_reward", "top1_reward", "top5_mean_reward",
            "policy_loss", "kl_loss", "ratio_mean"
        ]).to_csv(log_path, index=False)

    best_total_loss = float("inf")
    best_reward = float("-inf")
    best_total_loss_state = None
    best_reward_state = None

    # ==================== 外循环：Iterations ====================
    for it in range(1, iterations + 1):
        print(f"\n{'=' * 70}")
        print(f" Iteration {it}/{iterations}")
        print(f"{'=' * 70}")

        # ===== 更新 reference model（iteration级别，只在外循环更新）=====
        ref_agent.load_state_dict(agent.state_dict())
        for p in ref_agent.parameters():
            p.requires_grad_(False)

        # ==================== 中循环：M Steps ====================
        for step in range(1, M + 1):
            print(f" [Iter {it}, Step {step}/{M}]")

            # =====  更新 old policy（每个step都更新）=====
            old_agent.load_state_dict(agent.state_dict())
            for p in old_agent.parameters():
                p.requires_grad_(False)

            # ===== Rollout（使用 old_agent 采样）=====
            batch_token_ids, batch_smiles = sample_selfies_batch_from_generate_selfies(
                model_name="decoder_only_tfm",
                vocab=vocab,
                model=old_agent,  # 使用old_agent采样
                batch_size=batch_size,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                device=device,
            )

            if len(batch_token_ids) < 2:
                print(f"   Skip: no valid samples")
                continue

            # ===== Docking =====
            vina_results = batch_scores_from_vina(
                batch_smiles,
                receptor_file="./feedback/8sc7.pdbqt",
                pdbqt_dir="./feedback/temp/pdbqt",
                output_dir="./feedback/vina_results/"
            )
            reward_fn = make_reward_fn_from_vina(vina_results)
            rewards = [reward_fn(smi) for smi in batch_smiles]

            # ===== Filter invalid =====
            valid = [(s, r) for s, r in zip(batch_token_ids, rewards) if r != 0]
            if len(valid) < 2:
                print(f"   Skip: invalid rewards")
                continue

            batch_token_ids, rewards = zip(*valid)

            mean_reward = float(np.mean(rewards))
            top1_reward = max(rewards)
            top5_mean_reward = float(np.mean(sorted(rewards, reverse=True)[:min(5, len(rewards))]))

            # ==================== 内循环：μ Policy Updates ====================
            for update_idx in range(mu):
                total_loss, pol_loss, kl_loss, r_mean = compute_grpo_loss_batch(
                    agent,
                    old_agent,
                    ref_agent,
                    batch_token_ids,
                    rewards,
                    clip_eps,
                    kl_beta,
                )

                optimizer.zero_grad()
                total_loss.backward()

                #  添加梯度裁剪
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)

                optimizer.step()

            # ===== 记录到 CSV =====
            pd.DataFrame([{
                "iteration": it,
                "step": step,
                "mean_reward": mean_reward,
                "top1_reward": top1_reward,
                "top5_mean_reward": top5_mean_reward,
                "policy_loss": pol_loss.item(),
                "kl_loss": kl_loss.item(),
                "ratio_mean": r_mean.item()
            }]).to_csv(log_path, mode="a", index=False, header=False)

            print(
                f"   meanR={mean_reward:.3f} | top1={top1_reward:.3f} | "
                f"top5={top5_mean_reward:.3f}\n"
                f"   policy={pol_loss.item():.4f} | kl={kl_loss.item():.4f} | "
                f"ratio={r_mean.item():.3f}"
            )

            # ===== 保存最优模型 =====
            if total_loss.item() < best_total_loss:
                best_total_loss = total_loss.item()
                best_total_loss_state = copy.deepcopy(agent.state_dict())
                print(f"   New best total loss: {best_total_loss:.4f}")

            if mean_reward > best_reward:
                best_reward = mean_reward
                best_reward_state = copy.deepcopy(agent.state_dict())
                print(f"   New best reward: {best_reward:.3f}")

    # ===== 最终保存模型 =====
    if best_total_loss_state is not None:
        save_path = os.path.join(save_dir, f"best_total_loss_{timestamp}.pt")
        torch.save(best_total_loss_state, save_path)
        print(f" Saved best loss model: {save_path}")

    if best_reward_state is not None:
        save_path = os.path.join(save_dir, f"best_reward_{timestamp}.pt")
        torch.save(best_reward_state, save_path)
        print(f" Saved best reward model: {save_path}")


# ================== main ==================
if __name__ == "__main__":
    from data.data_utils import selfies_vocab
    from config.load_config import load_config
    from model.decoder_only_tfm import decoder_only_tfm

    config = load_config("./config/decoder_only_tfm_config.yaml")
    data = pd.read_csv(config["paths"]["data_file"])["selfies"].tolist()
    vocab = selfies_vocab(data)

    device = config["device"]
    model_cfg = config["model"]

    agent = decoder_only_tfm(
        vocab_size=len(vocab),
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        max_len=model_cfg["max_len"],
        dropout=model_cfg["dropout"]
    ).to(device)

    agent.load_state_dict(
        torch.load(
            "./model/decoder_only_tfm_best/best_model_fold2.pt",
            map_location=device,
            weights_only=True
        ),
        strict=False
    )

    optimizer = optim.AdamW(agent.parameters(), lr=1e-6)

    train_grpo(
        agent,
        vocab,
        optimizer,
        device=device,
        iterations=500,  #
        M=2,  #
        batch_size=8,
        mu=4,
        clip_eps=0.2,
        kl_beta=0.05,
        temperature=1.0,
        top_k=10,
    )
