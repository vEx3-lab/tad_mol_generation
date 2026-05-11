# ===== GSPO 批量训练脚本 (Decoder-Only Transformer) =====
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import copy
import math
import heapq
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from feedback.vina_scores import batch_scores_from_vina
from utils import sample_selfies_batch_from_generate_selfies, make_composite_reward


# ----------------- 工具函数 -----------------
def compute_sequence_logprob_from_logits(logits, target_ids, pad_token_id=0, eps=1e-8):
    """
    logits: [1, T, V]
    target_ids: [1, T]
    return: 标量，mask 后的 sequence 平均 logprob
    """
    logp = F.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)   # [1, T]
    mask = (target_ids != pad_token_id).float()
    valid_len = mask.sum() + eps
    seq_logp = (token_logp * mask).sum() / valid_len
    return seq_logp


@torch.no_grad()
def compute_old_logprob_batch(model, batch_sequences, device, pad_token_id=0):
    """
    为 rollout 出来的每条序列记录其在行为策略(old policy)下的 sequence-level logprob
    返回 list[float]
    """
    model.eval()
    seq_logprobs = []

    for seq in batch_sequences:
        seq = torch.tensor(seq, device=device)
        if len(seq) < 2:
            seq_logprobs.append(None)
            continue

        input_ids = seq[:-1].unsqueeze(0)
        target_ids = seq[1:].unsqueeze(0)
        logits = model(input_ids)
        seq_logp = compute_sequence_logprob_from_logits(
            logits, target_ids, pad_token_id=pad_token_id
        )
        seq_logprobs.append(float(seq_logp.item()))

    return seq_logprobs


# ----------------- GSPO loss（sequence-level PPO with replay support）-----------------
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
    stored_old_logprobs=None,      # 新增：支持 replay 样本使用入池时记录的行为策略 logprob
    ratio_min=None,                # 新增：ratio filter
    ratio_max=None,
):
    """
    若 stored_old_logprobs is None，则退化为标准 on-policy GSPO；
    若提供，则优先使用其作为行为策略 logprob（更适合 replay sample）。
    """
    device = next(agent.parameters()).device
    rewards = torch.tensor(batch_rewards, device=device)
    adv = (rewards - rewards.mean()) / (rewards.std() + eps)
    adv = adv.detach()

    policy_losses = []
    kl_losses = []
    ratio_means = []
    kept_count = 0

    for i, seq in enumerate(batch_sequences):
        seq = torch.tensor(seq, device=device)
        if len(seq) < 2:
            continue

        input_ids = seq[:-1].unsqueeze(0)
        target_ids = seq[1:].unsqueeze(0)
        mask = (target_ids != pad_token_id).float()

        logits_new = agent(input_ids)
        logits_ref = ref_agent(input_ids).detach()

        logp_new = F.log_softmax(logits_new, dim=-1)
        logp_ref = F.log_softmax(logits_ref, dim=-1)

        token_logp_new = logp_new.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logp_ref = logp_ref.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        valid_len = mask.sum() + eps
        seq_logp_new = (token_logp_new * mask).sum() / valid_len
        seq_logp_ref = (token_logp_ref * mask).sum() / valid_len

        # ===== 行为策略 logprob：优先使用 rollout 时存下来的 old_logprob =====
        if stored_old_logprobs is not None and stored_old_logprobs[i] is not None:
            seq_logp_old = torch.tensor(stored_old_logprobs[i], device=device, dtype=seq_logp_new.dtype)
        else:
            logits_old = old_agent(input_ids).detach()
            logp_old = F.log_softmax(logits_old, dim=-1)
            token_logp_old = logp_old.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            seq_logp_old = (token_logp_old * mask).sum() / valid_len

        # ===== PPO/GSPO ratio =====
        log_ratio_policy = seq_logp_new - seq_logp_old
        ratio_seq = torch.exp(log_ratio_policy)

        # ===== ratio filter（尤其适合 replay 样本）=====
        if ratio_min is not None and ratio_seq.item() < ratio_min:
            continue
        if ratio_max is not None and ratio_seq.item() > ratio_max:
            continue

        kept_count += 1
        ratio_means.append(ratio_seq.detach())

        surr1 = ratio_seq * adv[i]
        surr2 = torch.clamp(ratio_seq, 1 - clip_eps, 1 + clip_eps) * adv[i]
        policy_losses.append(-torch.min(surr1, surr2))

        # ===== KL to ref =====
        log_ratio_kl = seq_logp_ref - seq_logp_new
        kl = torch.exp(log_ratio_kl) - log_ratio_kl - 1
        kl_losses.append(kl)

    if len(policy_losses) == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, zero.detach(), zero.detach(), torch.tensor(1.0, device=device), 0

    policy_loss = torch.stack(policy_losses).mean()
    kl_loss = torch.stack(kl_losses).mean()
    ratio_mean = torch.stack(ratio_means).mean()

    total_loss = policy_loss + kl_beta * kl_loss
    return total_loss, policy_loss, kl_loss, ratio_mean, kept_count


# ===== ReplayBuffer：保存高奖励、较新的样本，并记录 rollout 时行为策略 logprob =====
class ReplayBuffer:
    def __init__(self, max_size=100, max_age_iters=5):
        self.max_size = max_size
        self.max_age_iters = max_age_iters
        self.buffer = []
        self._counter = 0

    def add(self, token_ids, smiles, reward, old_logprob, iteration_id, step_id):
        """
        小顶堆：reward 最低的先被弹出
        """
        if old_logprob is None:
            return

        self._counter += 1
        item = {
            "reward": float(reward),
            "counter": self._counter,
            "token_ids": token_ids,
            "smiles": smiles,
            "old_logprob": float(old_logprob),
            "iteration_id": int(iteration_id),
            "step_id": int(step_id),
        }
        heapq.heappush(self.buffer, (item["reward"], item["counter"], item))
        if len(self.buffer) > self.max_size:
            heapq.heappop(self.buffer)

    def purge_old(self, current_iter):
        kept = []
        for reward, counter, item in self.buffer:
            if current_iter - item["iteration_id"] <= self.max_age_iters:
                kept.append((reward, counter, item))
        self.buffer = kept
        heapq.heapify(self.buffer)

    def sample(self, n):
        if len(self.buffer) == 0:
            return []

        n = min(n, len(self.buffer))
        rewards = np.array([r for r, _, _ in self.buffer], dtype=np.float32)
        probs = np.exp(rewards - rewards.max())
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.buffer), size=n, replace=False, p=probs)
        sampled = [self.buffer[i][2] for i in indices]
        return sampled

    def __len__(self):
        return len(self.buffer)


# ----------------- 训练主循环（GSPO + ReplayBuffer）-----------------
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
        log_dir=os.path.join(BASE_DIR, "rl/feedback/logs"),
        save_dir=os.path.join(BASE_DIR, "rl/feedback/best_models"),
        replay_buffer_size=300,
        replay_batch_size=0,
        replay_start=30,
        replay_max_age_iters=5,    # 新增：只重放最近若干 iteration 的样本
        ratio_min=0.5,             # 新增：ratio filter
        ratio_max=1.5,
        pad_token_id=0,
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

    if not os.path.exists(log_path):
        pd.DataFrame(columns=[
            "iteration", "step", "global_step", "mean_reward", "mean_vina", "mean_qed",
            "mean_sa", "mean_logp", "top1_reward", "top5_mean_reward",
            "policy_loss", "kl_loss", "total_loss", "ratio_mean",
            "lr", "lr_before_update", "lr_after_update",
            "kl_beta", "replay_size", "kept_samples", "updates_applied",
            "grad_norm", "weight_decay", "clip_eps", "mu", "ratio_min", "ratio_max"
        ]).to_csv(log_path, index=False)

    best_total_loss = float("inf")
    best_reward = float("-inf")
    best_total_loss_state = None
    best_reward_state = None

    replay_buffer = ReplayBuffer(
        max_size=replay_buffer_size,
        max_age_iters=replay_max_age_iters
    )

    for it in range(1, iterations + 1):
        print(f"\n{'=' * 70}")
        print(f" Iteration {it}/{iterations}")
        print(f"{'=' * 70}")

        ref_agent.load_state_dict(agent.state_dict())
        for p in ref_agent.parameters():
            p.requires_grad_(False)

        # 清掉太旧的 replay 样本
        replay_buffer.purge_old(it)

        for step in range(1, M + 1):
            print(f" [Iter {it}, Step {step}/{M}]")
            global_step = (it - 1) * M + step

            old_agent.load_state_dict(agent.state_dict())
            for p in old_agent.parameters():
                p.requires_grad_(False)

            # ===== Rollout（用 old_agent 采样）=====
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
                print("   Skip: no valid samples")
                continue

            # ===== 为当前 rollout 样本记录行为策略 old_logprob =====
            batch_old_logprobs = compute_old_logprob_batch(
                old_agent,
                batch_token_ids,
                device=device,
                pad_token_id=pad_token_id
            )

            # ===== Docking =====
            vina_results = batch_scores_from_vina(
                batch_smiles,
                receptor_file=os.path.join(BASE_DIR, "feedback/8sc7.pdbqt"),
                pdbqt_dir=os.path.join(BASE_DIR, "feedback/temp/pdbqt"),
                output_dir=os.path.join(BASE_DIR, "feedback/vina_results/")
            )

            reward_items = make_composite_reward(
                batch_smiles,
                vina_results,
                weights={'vina': 1.0, 'qed': 0.2, 'sa': 0.2, 'logp': 0.0},
                invalid_penalty=0
            )

            rewards = [item['reward'] for item in reward_items]

            # ===== Filter invalid，同时保留 smiles 和 old_logprob =====
            valid = [
                (tok, r, smi, old_lp)
                for tok, r, smi, old_lp in zip(batch_token_ids, rewards, batch_smiles, batch_old_logprobs)
                if r != 0 and old_lp is not None
            ]
            if len(valid) < 2:
                print("   Skip: invalid rewards")
                continue

            batch_token_ids, rewards, batch_smiles_valid, batch_old_logprobs = zip(*valid)

            valid_indices = [i for i, r in enumerate([item['reward'] for item in reward_items]) if r != 0]
            valid_items = [reward_items[i] for i in valid_indices]

            mean_reward = float(np.mean(rewards))
            top1_reward = max(rewards)
            top5_mean_reward = float(np.mean(sorted(rewards, reverse=True)[:min(5, len(rewards))]))
            mean_vina = float(np.mean([item['vina'] for item in valid_items]))
            mean_qed = float(np.mean([item['qed'] for item in valid_items]))
            mean_sa = float(np.mean([item['sa'] for item in valid_items]))
            mean_logp = float(np.mean([item['logp'] for item in valid_items]))

            # ===== 当前 rollout 样本写入 ReplayBuffer =====
            for token_ids, smiles, reward, old_lp in zip(
                batch_token_ids, batch_smiles_valid, rewards, batch_old_logprobs
            ):
                replay_buffer.add(
                    token_ids=token_ids,
                    smiles=smiles,
                    reward=reward,
                    old_logprob=old_lp,
                    iteration_id=it,
                    step_id=step,
                )

            # ===== fresh + replay 混合 =====
            mixed_token_ids = list(batch_token_ids)
            mixed_rewards = list(rewards)
            mixed_old_logprobs = list(batch_old_logprobs)

            if len(replay_buffer) >= replay_start:
                replay_samples = replay_buffer.sample(replay_batch_size)
                replay_ids = [x["token_ids"] for x in replay_samples]
                replay_rewards = [x["reward"] for x in replay_samples]
                replay_old_logprobs = [x["old_logprob"] for x in replay_samples]

                mixed_token_ids += replay_ids
                mixed_rewards += replay_rewards
                mixed_old_logprobs += replay_old_logprobs

                print(f"   Replay: +{len(replay_ids)} samples | buffer size={len(replay_buffer)}")
            else:
                print(f"   Replay: warming up ({len(replay_buffer)}/{replay_start})")

            lr_before_update = optimizer.param_groups[0]["lr"]
            current_lr = lr_before_update
            grad_norm_value = 0.0
            updates_applied = 0

            # ===== Policy updates（使用混合数据 + stored old logprob）=====
            for _ in range(mu):
                total_loss, pol_loss, kl_loss, r_mean, kept_count = compute_gspo_loss_batch(
                    agent=agent,
                    old_agent=old_agent,
                    ref_agent=ref_agent,
                    batch_sequences=mixed_token_ids,
                    batch_rewards=mixed_rewards,
                    clip_eps=clip_eps,
                    kl_beta=kl_beta,
                    pad_token_id=pad_token_id,
                    stored_old_logprobs=mixed_old_logprobs,
                    ratio_min=ratio_min,
                    ratio_max=ratio_max,
                )

                if kept_count == 0:
                    print("   Skip update: no samples passed ratio filter")
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                grad_norm_value = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
                updates_applied += 1

            lr_after_update = current_lr
            weight_decay = optimizer.param_groups[0].get("weight_decay", 0.0)

            pd.DataFrame([{
                "iteration": it,
                "step": step,
                "global_step": global_step,
                "mean_reward": mean_reward,
                "mean_vina": mean_vina,
                "mean_qed": mean_qed,
                "mean_sa": mean_sa,
                "mean_logp": mean_logp,
                "top1_reward": top1_reward,
                "top5_mean_reward": top5_mean_reward,
                "policy_loss": float(pol_loss.item()),
                "kl_loss": float(kl_loss.item()),
                "total_loss": float(total_loss.item()),
                "ratio_mean": float(r_mean.item()),
                "lr": current_lr,
                "lr_before_update": lr_before_update,
                "lr_after_update": lr_after_update,
                "kl_beta": kl_beta,
                "replay_size": len(replay_buffer),
                "kept_samples": kept_count,
                "updates_applied": updates_applied,
                "grad_norm": grad_norm_value,
                "weight_decay": weight_decay,
                "clip_eps": clip_eps,
                "mu": mu,
                "ratio_min": ratio_min,
                "ratio_max": ratio_max,
            }]).to_csv(log_path, mode="a", index=False, header=False)

            print(
                f"   meanR={mean_reward:.3f} | top1={top1_reward:.3f} | "
                f"mean_vina={mean_vina:.3f} | mean_qed={mean_qed:.3f} | "
                f"mean_sa={mean_sa:.3f} | mean_logp={mean_logp:.3f} | "
                f"top5={top5_mean_reward:.3f}\n"
                f"   policy={pol_loss.item():.4f} | kl={kl_loss.item():.4f} | "
                f"ratio={r_mean.item():.3f} | buffer={len(replay_buffer)} | kept={kept_count}"
            )

            if kl_loss.item() > 5.0:
                print(f"   Early stopping: KL loss too high ({kl_loss.item():.4f} > 1.0)")
                break

            if total_loss.item() < best_total_loss:
                best_total_loss = total_loss.item()
                best_total_loss_state = copy.deepcopy(agent.state_dict())
                print(f"   New best total loss: {best_total_loss:.4f}")

            if mean_reward > best_reward:
                best_reward = mean_reward
                best_reward_state = copy.deepcopy(agent.state_dict())
                print(f"   New best reward: {best_reward:.3f}")

    if best_total_loss_state is not None:
        save_path = os.path.join(save_dir, f"best_total_loss_{timestamp}.pt")
        torch.save(best_total_loss_state, save_path)
        print(f" Saved best loss model: {save_path}")

    if best_reward_state is not None:
        save_path = os.path.join(save_dir, f"best_reward_{timestamp}.pt")
        torch.save(best_reward_state, save_path)
        print(f" Saved best reward model: {save_path}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===== 冻结 agent 大部分参数，只微调最后一层和输出层 =====
def freeze_agent_partial(agent):
    for param in agent.parameters():
        param.requires_grad = False

    for param in agent.layers[-1].parameters():
        param.requires_grad = True

    for param in agent.fc_out.parameters():
        param.requires_grad = True

    trainable_params = []
    frozen_params = []

    for name, p in agent.named_parameters():
        if p.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print("=" * 60)
    print(f"Trainable params ({len(trainable_params)}):")
    for n in trainable_params:
        print("  +", n)

    print("-" * 60)
    print(f"Frozen params ({len(frozen_params)}):")
    for n in frozen_params:
        print("  -", n)
    print("=" * 60)


# ================== main ==================
if __name__ == "__main__":
    from data.data_utils import selfies_vocab
    from config.load_config import load_config
    from model.decoder_only_tfm import decoder_only_tfm

    set_seed(42)

    config = load_config(os.path.join(BASE_DIR, "config/decoder_only_tfm_config.yaml"))
    data = pd.read_csv(os.path.join(BASE_DIR, 'data/htvs_molecules_with_selfies.csv'))["selfies"].tolist()
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
            os.path.join(BASE_DIR, "train/model/decoder_only_tfm/best_model_fold1.pt"),
            map_location=device,
            weights_only=True
        ),
        strict=False
    )

    freeze_agent_partial(agent)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, agent.parameters()),
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=1000,
    #     eta_min=1e-6
    # )
    scheduler = None
    train_gspo(
        agent,
        vocab,
        optimizer,
        scheduler,
        device=device,
        iterations=1,
        M=1000,
        batch_size=16,
        mu=3,
        clip_eps=0.25,
        kl_beta=0.15,
        temperature=1.0,
        top_k=30,
        log_dir=os.path.join(BASE_DIR, "rl/feedback/logs"),
        save_dir=os.path.join(BASE_DIR, "rl/feedback/best_models"),
        replay_buffer_size=300,
        replay_batch_size=0,
        replay_start=50,
        replay_max_age_iters=5,
        ratio_min=0.5,
        ratio_max=1.5,
        pad_token_id=0,
    )
