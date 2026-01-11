# ===== GSPO 批量训练脚本 (Decoder-Only Transformer) =====
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import selfies as sf
from datetime import datetime
from feedback.vina_scores import batch_scores_from_vina
from utils import make_reward_fn_from_vina,sample_selfies_batch_from_generate_selfies

# -----------------  GSPO loss（sequence-level PPO） -----------------
def compute_gspo_loss_batch(
    agent,
    old_agent,
    ref_agent,
    batch_sequences,
    batch_rewards,
    clip_eps=0.2,
    kl_beta=0.05,
    eps=1e-8,
):
    device = next(agent.parameters()).device
    rewards = torch.tensor(batch_rewards, device=device)
    adv = (rewards - rewards.mean()) / (rewards.std() + eps)
    adv = adv.detach()

    policy_losses = []
    kl_losses = []
    ratio_means = []

    for i, seq in enumerate(batch_sequences):
        seq = torch.tensor(seq, device=device)
        if len(seq) < 2:
            continue

        input_ids = seq[:-1].unsqueeze(0)
        target_ids = seq[1:].unsqueeze(0)

        logits_new = agent(input_ids)
        logits_old = old_agent(input_ids).detach()
        logits_ref = ref_agent(input_ids).detach()

        logp_new = F.log_softmax(logits_new, dim=-1)
        logp_old = F.log_softmax(logits_old, dim=-1)
        logp_ref = F.log_softmax(logits_ref, dim=-1)

        token_logp_new = logp_new.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logp_old = logp_old.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logp_ref = logp_ref.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        # ===== GSPO sequence ratio =====
        log_ratio_seq = (token_logp_new - token_logp_old).mean()
        ratio_seq = torch.exp(log_ratio_seq)
        ratio_means.append(ratio_seq.detach())

        surr1 = ratio_seq * adv[i]
        surr2 = torch.clamp(ratio_seq, 1 - clip_eps, 1 + clip_eps) * adv[i]
        policy_losses.append(-torch.min(surr1, surr2))

        # ===== KL (token-mean) =====
        log_ratio_ref = token_logp_ref - token_logp_new
        kl = (torch.exp(log_ratio_ref) - log_ratio_ref - 1).mean()
        kl_losses.append(kl)

    policy_loss = torch.stack(policy_losses).mean()
    kl_loss = torch.stack(kl_losses).mean()
    ratio_mean = torch.stack(ratio_means).mean()

    total_loss = policy_loss + kl_beta * kl_loss
    return total_loss, policy_loss, kl_loss, ratio_mean




# ----------------- 4. 训练主循环（GRPO） -----------------
def train_gspo(
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
    log_path = os.path.join(log_dir, f"gspo_training_log_{timestamp}.csv")

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
                receptor_file="../feedback/8sc7.pdbqt",
                pdbqt_dir="../feedback/temp/pdbqt",
                output_dir="../feedback/vina_results/"
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
                total_loss, pol_loss, kl_loss, r_mean = compute_gspo_loss_batch(
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

    config = load_config("../config/decoder_only_tfm_config.yaml")
    data =   pd.read_csv('../data/htvs_molecules_with_selfies.csv')["selfies"].tolist()
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
            "../model/decoder_only_tfm_best/best_model_fold2.pt",
            map_location=device,
            weights_only=True
        ),
        strict=False
    )

    optimizer = optim.AdamW(agent.parameters(), lr=1e-5)

    train_gspo(
        agent,
        vocab,
        optimizer,
        device=device,
        iterations=500,  #
        M=2,  #
        batch_size=16,
        mu=4,
        clip_eps=0.2,
        kl_beta=0.01,
        temperature=1.0,
        top_k=10,
    )
