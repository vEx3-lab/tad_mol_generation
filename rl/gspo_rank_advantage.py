import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from datetime import datetime
from feedback.vina_scores import batch_scores_from_vina
from utils import make_reward_fn_from_vina,sample_selfies_batch_from_generate_selfies
import random
from utils import make_composite_reward

def compute_hybrid_advantage(rewards, alpha=0.5, eps=1e-8):
    rewards = np.array(rewards)

    # continuous
    A_cont = (rewards - rewards.mean()) / (rewards.std() + eps)

    # ranking
    ranks = np.argsort(np.argsort(rewards))
    ranks = ranks.astype(np.float32)
    A_rank = (ranks - ranks.mean()) / (ranks.std() + eps)

    # combine
    A = alpha * A_cont + (1 - alpha) * A_rank

    return torch.tensor(A, dtype=torch.float32)


def compute_gspo_loss_batch(
    agent,
    old_agent,
    ref_agent,
    batch_sequences,
    batch_rewards,
    clip_eps=0.2,
    kl_beta=0.05,
    eps=1e-8,
    pad_token_id=0,
    alpha=0.5,
):
    device = next(agent.parameters()).device

    # ⭐ Hybrid Advantage（只用current）
    adv = compute_hybrid_advantage(batch_rewards, alpha=alpha).to(device)
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

        mask = (target_ids != pad_token_id).float()
        valid_len = mask.sum() + eps

        logits_new = agent(input_ids)
        logits_old = old_agent(input_ids).detach()
        logits_ref = ref_agent(input_ids).detach()

        logp_new = F.log_softmax(logits_new, dim=-1)
        logp_old = F.log_softmax(logits_old, dim=-1)
        logp_ref = F.log_softmax(logits_ref, dim=-1)

        token_logp_new = logp_new.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logp_old = logp_old.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logp_ref = logp_ref.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        # PPO ratio
        log_ratio_policy = ((token_logp_new - token_logp_old) * mask).sum() / valid_len
        ratio_seq = torch.exp(log_ratio_policy)
        ratio_means.append(ratio_seq.detach())

        surr1 = ratio_seq * adv[i]
        surr2 = torch.clamp(ratio_seq, 1 - clip_eps, 1 + clip_eps) * adv[i]
        policy_losses.append(-torch.min(surr1, surr2))

        # KL
        log_ratio_kl = ((token_logp_ref - token_logp_new) * mask).sum() / valid_len
        kl = torch.exp(log_ratio_kl) - log_ratio_kl - 1
        kl_losses.append(kl)

    policy_loss = torch.stack(policy_losses).mean()
    kl_loss = torch.stack(kl_losses).mean()
    ratio_mean = torch.stack(ratio_means).mean()

    total_loss = policy_loss + kl_beta * kl_loss

    return total_loss, policy_loss, kl_loss, ratio_mean

def train_gspo(
    agent,
    vocab,
    optimizer,
    scheduler,
    device="cuda",
    iterations=100,
    M=4,
    batch_size=8,
    max_len=80,
    mu=4,
    clip_eps=0.1,
    kl_beta=0.1,
    temperature=1.0,
    top_k=10,
    log_dir="./feedback/logs",
    save_dir="./feedback/best_models",
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

    # ===== 初始化CSV =====
    if not os.path.exists(log_path):
        pd.DataFrame(columns=[
            "iteration", "step", "mean_reward",
            "mean_vina", "mean_qed", "mean_sa", "mean_logp",
            "top1_reward", "top5_mean_reward",
            "unique_ratio",
            "policy_loss", "kl_loss", "ratio_mean"
        ]).to_csv(log_path, index=False)

    best_reward = float("-inf")

    for it in range(1, iterations + 1):

        ref_agent.load_state_dict(agent.state_dict())
        for p in ref_agent.parameters():
            p.requires_grad_(False)

        for step in range(1, M + 1):

            old_agent.load_state_dict(agent.state_dict())
            for p in old_agent.parameters():
                p.requires_grad_(False)

            # ===== 采样 =====
            batch_token_ids, batch_smiles = sample_selfies_batch_from_generate_selfies(
                vocab=vocab,
                model=old_agent,
                batch_size=batch_size,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                device=device,
            )

            if len(batch_token_ids) < 2:
                continue

            # ===== Docking =====
            vina_results = batch_scores_from_vina(
                batch_smiles,
                receptor_file="../feedback/8sc7.pdbqt",
                pdbqt_dir="../feedback/temp/pdbqt",
                output_dir="../feedback/vina_results/"
            )

            reward_items = make_composite_reward(
                batch_smiles,
                vina_results,
                weights={'vina': 1, 'qed': 1, 'sa': 0.5, 'logp': 0.5},
                invalid_penalty=0
            )

            rewards = [item['reward'] for item in reward_items]

            # ===== 过滤无效 =====
            valid = [(s, r, smi) for s, r, smi in zip(batch_token_ids, rewards, batch_smiles) if r != 0]
            if len(valid) < 2:
                continue

            batch_token_ids, rewards, batch_smiles = zip(*valid)

            # ===== 指标统计 =====
            valid_items = [item for item in reward_items if item['reward'] != 0]

            mean_reward = float(np.mean(rewards))
            top1_reward = float(max(rewards))
            top5_mean_reward = float(np.mean(sorted(rewards, reverse=True)[:min(5, len(rewards))]))

            mean_vina = float(np.mean([item['vina'] for item in valid_items]))
            mean_qed = float(np.mean([item['qed'] for item in valid_items]))
            mean_sa = float(np.mean([item['sa'] for item in valid_items]))
            mean_logp = float(np.mean([item['logp'] for item in valid_items]))

            unique_ratio = len(set(batch_smiles)) / len(batch_smiles)

            # ===== RL更新 =====
            for _ in range(mu):
                total_loss, pol_loss, kl_loss, r_mean = compute_gspo_loss_batch(
                    agent,
                    old_agent,
                    ref_agent,
                    batch_token_ids,
                    rewards,
                    clip_eps,
                    kl_beta,
                    alpha=0.7,
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            # ===== 写日志 =====
            pd.DataFrame([{
                "iteration": it,
                "step": step,
                "mean_reward": mean_reward,
                "mean_vina": mean_vina,
                "mean_qed": mean_qed,
                "mean_sa": mean_sa,
                "mean_logp": mean_logp,
                "top1_reward": top1_reward,
                "top5_mean_reward": top5_mean_reward,
                "unique_ratio": unique_ratio,
                "policy_loss": pol_loss.item(),
                "kl_loss": kl_loss.item(),
                "ratio_mean": r_mean.item()
            }]).to_csv(log_path, mode="a", index=False, header=False)

            # ===== 打印 =====
            print(
                f"[Iter {it} Step {step}] "
                f"meanR={mean_reward:.3f} | top1={top1_reward:.3f} | top5={top5_mean_reward:.3f}\n"
                f"vina={mean_vina:.3f} | qed={mean_qed:.3f} | sa={mean_sa:.3f} | logp={mean_logp:.3f} | unique={unique_ratio:.3f}\n"
                f"policy={pol_loss.item():.4f} | kl={kl_loss.item():.4f} | ratio={r_mean.item():.3f}"
            )

            if mean_reward > best_reward:
                best_reward = mean_reward
                torch.save(
                    agent.state_dict(),
                    os.path.join(save_dir, f"best_reward_{timestamp}.pt")
                )
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ===== 冻结 agent 大部分参数，只微调最后一层和输出层 =====
def freeze_agent_partial(agent):
    # 1. 冻结全部参数
    for param in agent.parameters():
        param.requires_grad = False

    # 2. 解冻最后一层 Transformer Block
    for param in agent.layers[-1].parameters():
        param.requires_grad = True

    # 3. 解冻输出层 lm_head
    # for param in agent.lm_head.parameters():
    #     param.requires_grad = True
    for param in agent.fc_out.parameters():
        param.requires_grad = True

    # 4. 收集参数信息
    trainable_params = []
    frozen_params = []

    for name, p in agent.named_parameters():
        if p.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    # 5. 打印结果
    print("=" * 60)
    print(f"Trainable params ({len(trainable_params)}):")
    for n in trainable_params:
        print("  ✓", n)

    print("-" * 60)
    print(f"Frozen params ({len(frozen_params)}):")
    for n in frozen_params:
        print("  ❄", n)
    print("=" * 60)


if __name__ == "__main__":
    from data.data_utils import selfies_vocab
    from config.load_config import load_config
    # from model.v6 import decoder_only_lm
    from model.decoder_only_tfm import decoder_only_tfm
    set_seed(42)

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
            "../train/model/decoder_only_tfm/best_model_fold1.pt",
            map_location=device,
            weights_only=True
        ),
        strict=False
    )
    # ===== 冻结部分参数 =====
    freeze_agent_partial(agent)

    # optimizer 只会更新 requires_grad=True 的参数
    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, agent.parameters()), lr=5e-6)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, agent.parameters()),
        lr=5e-6,
        betas=(0.9, 0.999),
        weight_decay=0.001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1000,  # 总step数
        eta_min=1e-7  # 最低lr
    )
    # optimizer = optim.AdamW(agent.parameters(), lr=5e-5)

    train_gspo(
        agent,
        vocab,
        optimizer,
        scheduler,
        device=device,
        iterations=1,  #
        M=500,  #
        batch_size=8,
        mu=1,
        clip_eps=0.2,
        kl_beta=5e-2,
        temperature=1,
        top_k=20,
    )
