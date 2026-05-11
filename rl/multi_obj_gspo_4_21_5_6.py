import copy
import heapq
import os
import random
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, RL_DIR)

import numpy as np
import pandas as pd
import torch

from feedback.utils import canonicalize_smiles
from feedback.vina_scores import batch_scores_from_vina
from rl.multi_obj_gspo_4_21 import set_seed
from rl.multi_obj_gspo_4_21_diagnostics import (
    _count_vina_status,
    _mean_or_none,
    _update_ema,
    _window_metrics,
    compute_gspo_loss_batch,
    compute_old_logprob_stats_batch,
    compute_sequence_ratio_diagnostics,
    compute_sampling_sequence_logprob,
)
from utils import make_composite_reward, sample_selfies_batch_from_generate_selfies


def freeze_agent_last_n_layers(agent, n_trainable_layers=2):
    """Freeze most of the decoder and train the last N transformer layers plus output head."""
    for param in agent.parameters():
        param.requires_grad = False

    layers = getattr(agent, "layers", None)
    if layers is not None and len(layers) > 0:
        for layer in layers[-n_trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    if hasattr(agent, "fc_out"):
        for param in agent.fc_out.parameters():
            param.requires_grad = True

    trainable = [name for name, p in agent.named_parameters() if p.requires_grad]
    frozen = [name for name, p in agent.named_parameters() if not p.requires_grad]

    print("=" * 60)
    print(f"Trainable params ({len(trainable)}):")
    for name in trainable:
        print("  +", name)
    print("-" * 60)
    print(f"Frozen params ({len(frozen)}):")
    for name in frozen:
        print("  -", name)
    print("=" * 60)


class EliteBuffer:
    """Keep high-reward samples for auxiliary likelihood learning."""

    def __init__(self, max_size=300):
        self.max_size = int(max_size)
        self._counter = 0
        self._heap = []
        self._by_smiles = {}

    def add(self, token_ids, smiles, reward, old_logprob=None, item=None, global_step=0):
        canonical = canonicalize_smiles(smiles) or smiles
        if canonical is None:
            return False

        record = {
            "token_ids": list(token_ids),
            "smiles": smiles,
            "canonical_smiles": canonical,
            "reward": float(reward),
            "old_logprob": None if old_logprob is None else float(old_logprob),
            "global_step": int(global_step),
            "vina_raw": None if item is None else item.get("vina_raw"),
            "vina_reward": None if item is None else item.get("vina_reward"),
            "qed": None if item is None else item.get("qed"),
            "sa_raw": None if item is None else item.get("sa_raw"),
            "sa_reward": None if item is None else item.get("sa_reward"),
            "logp_raw": None if item is None else item.get("logp_raw"),
            "logp_reward": None if item is None else item.get("logp_reward"),
        }

        old = self._by_smiles.get(canonical)
        if old is not None and old["reward"] >= record["reward"]:
            return False

        self._counter += 1
        record["counter"] = self._counter
        self._by_smiles[canonical] = record
        self._rebuild_heap()
        return True

    def _rebuild_heap(self):
        records = sorted(
            self._by_smiles.values(),
            key=lambda x: (x["reward"], x["counter"]),
            reverse=True,
        )[: self.max_size]
        self._by_smiles = {r["canonical_smiles"]: r for r in records}
        self._heap = [(r["reward"], r["counter"], r["canonical_smiles"]) for r in records]
        heapq.heapify(self._heap)

    def sample(self, n, reward_temperature=0.08):
        records = list(self._by_smiles.values())
        if not records or n <= 0:
            return []

        n = min(int(n), len(records))
        rewards = np.asarray([r["reward"] for r in records], dtype=np.float64)
        temp = max(float(reward_temperature), 1e-6)
        probs = np.exp((rewards - rewards.max()) / temp)
        probs = probs / probs.sum()
        indices = np.random.choice(len(records), size=n, replace=False, p=probs)
        return [records[i] for i in indices]

    def save_csv(self, path):
        records = sorted(self._by_smiles.values(), key=lambda x: x["reward"], reverse=True)
        pd.DataFrame(records).drop(columns=["token_ids"], errors="ignore").to_csv(path, index=False)

    def __len__(self):
        return len(self._by_smiles)


def elite_aux_likelihood_loss(
    agent,
    elite_samples,
    vocab,
    device,
    pad_token_id=0,
    temperature=1.0,
    reward_temperature=0.08,
):
    if not elite_samples:
        return torch.tensor(0.0, device=device)

    rewards = torch.tensor([s["reward"] for s in elite_samples], dtype=torch.float32, device=device)
    weights = torch.softmax((rewards - rewards.max()) / max(reward_temperature, 1e-6), dim=0).detach()

    losses = []
    was_training = agent.training
    agent.eval()
    try:
        for sample, weight in zip(elite_samples, weights):
            seq_logp = compute_sampling_sequence_logprob(
                agent,
                sample["token_ids"],
                vocab,
                device,
                pad_token_id=pad_token_id,
                temperature=temperature,
                top_k=None,
            )
            if seq_logp is not None:
                losses.append(-weight * seq_logp)
    finally:
        agent.train(was_training)

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).sum()


class RewardMovingBaseline:
    """ACEGEN-style running reward baseline for lower-variance advantages."""

    def __init__(self, epsilon=1e-3, device="cpu"):
        self.mean = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.std = torch.tensor(1.0, dtype=torch.float32, device=device)
        self.count = float(epsilon)

    def update(self, rewards):
        rewards = rewards.detach().float()
        if rewards.numel() == 0:
            return

        batch_mean = rewards.mean()
        batch_std = rewards.std(unbiased=False)
        batch_count = float(rewards.numel())
        total_count = self.count + batch_count
        self.mean = self.mean + (batch_mean - self.mean) * (batch_count / total_count)
        self.std = self.std + (batch_std - self.std) * (batch_count / total_count)
        self.count = total_count


def deduplicate_rollout_batch(batch_token_ids, batch_smiles, enabled=True):
    """Deduplicate valid molecules by canonical SMILES while keeping invalid samples visible to scoring."""
    raw_count = len(batch_smiles)
    stats = {
        "raw_sample_count": raw_count,
        "dedup_count": raw_count,
        "duplicate_removed": 0,
    }
    if not enabled:
        return list(batch_token_ids), list(batch_smiles), stats

    seen = set()
    dedup_token_ids = []
    dedup_smiles = []
    duplicate_removed = 0

    for token_ids, smiles in zip(batch_token_ids, batch_smiles):
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            if canonical in seen:
                duplicate_removed += 1
                continue
            seen.add(canonical)

        dedup_token_ids.append(token_ids)
        dedup_smiles.append(smiles)

    stats["dedup_count"] = len(dedup_smiles)
    stats["duplicate_removed"] = duplicate_removed
    return dedup_token_ids, dedup_smiles, stats


def prepare_advantages(
    rewards,
    device,
    mode="batch_zscore",
    baseline=None,
    clip_adv=3.0,
    eps=1e-8,
):
    """Create stable sequence-level advantages and return logging stats."""
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    mode = (mode or "batch_zscore").lower()

    if reward_tensor.numel() == 0:
        advantages = reward_tensor
    elif mode == "batch_zscore":
        mean = reward_tensor.mean()
        std = reward_tensor.std(unbiased=False)
        if reward_tensor.numel() < 2 or std.item() <= eps:
            advantages = torch.zeros_like(reward_tensor)
        else:
            advantages = (reward_tensor - mean) / (std + eps)
    elif mode == "moving_baseline":
        if baseline is None:
            raise ValueError("moving_baseline advantage mode requires a baseline object")
        baseline.update(reward_tensor)
        advantages = reward_tensor - baseline.mean
    elif mode == "raw":
        advantages = reward_tensor
    else:
        raise ValueError(f"Unknown advantage_mode: {mode}")

    advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_adv is not None:
        advantages = torch.clamp(advantages, -float(clip_adv), float(clip_adv))
    advantages = advantages.detach()

    if advantages.numel() == 0:
        stats = {"adv_mean": 0.0, "adv_std": 0.0, "adv_max_abs": 0.0}
    else:
        stats = {
            "adv_mean": float(advantages.mean().item()),
            "adv_std": float(advantages.std(unbiased=False).item()),
            "adv_max_abs": float(advantages.abs().max().item()),
        }
    return advantages, stats


def adjust_kl_beta(kl_beta, kl_value, target_kl, kl_beta_min, kl_beta_max):
    """Adapt KL pressure for the next update using the latest observed KL."""
    if target_kl is None or target_kl <= 0:
        return kl_beta, 0

    new_beta = float(kl_beta)
    if kl_value > target_kl * 1.5:
        new_beta = min(new_beta * 1.5, float(kl_beta_max))
    elif kl_value < target_kl / 3.0:
        new_beta = max(new_beta / 1.2, float(kl_beta_min))

    adjusted = int(abs(new_beta - float(kl_beta)) > 1e-12)
    return new_beta, adjusted


def compute_gspo_loss_batch(
    agent,
    old_agent,
    ref_agent,
    batch_sequences,
    batch_rewards,
    vocab,
    clip_eps=0.2,
    kl_beta=0.05,
    eps=1e-8,
    pad_token_id=0,
    stored_old_logprobs=None,
    ratio_min=None,
    ratio_max=None,
    temperature=1.0,
    top_k=None,
    batch_advantages=None,
):
    device = next(agent.parameters()).device
    if batch_advantages is None:
        rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
        std = rewards.std(unbiased=False)
        if rewards.numel() < 2 or std.item() <= eps:
            adv = torch.zeros_like(rewards)
        else:
            adv = (rewards - rewards.mean()) / (std + eps)
    else:
        adv = torch.as_tensor(batch_advantages, dtype=torch.float32, device=device)
        if adv.numel() != len(batch_sequences):
            raise ValueError("batch_advantages must have the same length as batch_sequences")
    adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0).detach()

    policy_losses = []
    kl_losses = []
    ratio_means = []
    kept_count = 0
    was_training = agent.training
    agent.eval()

    try:
        for i, seq in enumerate(batch_sequences):
            if len(seq) < 1:
                continue

            seq_logp_new = compute_sampling_sequence_logprob(
                agent,
                seq,
                vocab,
                device,
                pad_token_id=pad_token_id,
                temperature=temperature,
                top_k=top_k,
            )
            if seq_logp_new is None:
                continue

            with torch.no_grad():
                seq_logp_ref = compute_sampling_sequence_logprob(
                    ref_agent,
                    seq,
                    vocab,
                    device,
                    pad_token_id=pad_token_id,
                    temperature=temperature,
                    top_k=top_k,
                )
                if seq_logp_ref is None:
                    continue

                if stored_old_logprobs is not None and stored_old_logprobs[i] is not None:
                    seq_logp_old = torch.tensor(
                        stored_old_logprobs[i],
                        device=device,
                        dtype=seq_logp_new.dtype,
                    )
                else:
                    seq_logp_old = compute_sampling_sequence_logprob(
                        old_agent,
                        seq,
                        vocab,
                        device,
                        pad_token_id=pad_token_id,
                        temperature=temperature,
                        top_k=top_k,
                    )
                    if seq_logp_old is None:
                        continue

            log_ratio_policy = seq_logp_new - seq_logp_old
            ratio_seq = torch.exp(log_ratio_policy)

            if ratio_min is not None and ratio_seq.item() < ratio_min:
                continue
            if ratio_max is not None and ratio_seq.item() > ratio_max:
                continue

            kept_count += 1
            ratio_means.append(ratio_seq.detach())

            surr1 = ratio_seq * adv[i]
            surr2 = torch.clamp(ratio_seq, 1 - clip_eps, 1 + clip_eps) * adv[i]
            policy_losses.append(-torch.min(surr1, surr2))

            log_ratio_kl = seq_logp_ref - seq_logp_new
            kl = torch.exp(log_ratio_kl) - log_ratio_kl - 1
            kl_losses.append(kl)
    finally:
        agent.train(was_training)

    if len(policy_losses) == 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, zero.detach(), zero.detach(), torch.tensor(1.0, device=device), 0

    policy_loss = torch.stack(policy_losses).mean()
    kl_loss = torch.stack(kl_losses).mean()
    ratio_mean = torch.stack(ratio_means).mean()
    total_loss = policy_loss + kl_beta * kl_loss
    return total_loss, policy_loss, kl_loss, ratio_mean, kept_count


def train_gspo_5_6(
    agent,
    vocab,
    optimizer,
    scheduler,
    device="cuda",
    iterations=1,
    M=1000,
    batch_size=16,
    max_len=80,
    mu=1,
    clip_eps=0.25,
    kl_beta=0.10,
    temperature=1.15,
    top_k=80,
    reward_weights=None,
    log_dir=os.path.join(BASE_DIR, "rl/feedback/logs"),
    save_dir=os.path.join(BASE_DIR, "rl/feedback/best_models"),
    elite_buffer_size=300,
    aux_batch_size=8,
    aux_coef=0.05,
    aux_start=50,
    aux_reward_temperature=0.08,
    ratio_min=0.5,
    ratio_max=1.5,
    pad_token_id=0,
    ema_alpha=0.1,
    plateau_window=30,
    plateau_patience=8,
    plateau_tol=0.003,
    plateau_min_steps=150,
    kl_stop_threshold=0.8,
    kl_stop_patience=3,
    kl_stop_min_steps=80,
    deduplicate_batch=True,
    min_unique_samples=8,
    min_raw_unique_rate=0.5,
    collapse_stop_patience=12,
    advantage_mode="batch_zscore",
    clip_adv=3.0,
    target_kl=0.05,
    kl_beta_min=1e-3,
    kl_beta_max=2.0,
):
    if reward_weights is None:
        reward_weights = {"vina": 1.2, "qed": 0.2, "sa": 0.15, "logp": 0.0}

    min_unique_samples = max(2, min(int(min_unique_samples), int(batch_size)))
    min_raw_unique_rate = min(max(float(min_raw_unique_rate), 0.0), 1.0)

    agent.to(device)
    old_agent = copy.deepcopy(agent).to(device)
    ref_agent = copy.deepcopy(agent).to(device)
    old_agent.eval()
    ref_agent.eval()

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"gspo_5_6_training_log_{timestamp}.csv")
    elite_path = os.path.join(log_dir, f"gspo_5_6_elite_buffer_{timestamp}.csv")

    columns = [
        "iteration", "step", "global_step",
        "mean_reward", "mean_reward_ema", "top1_reward", "top5_mean_reward", "top5_reward_ema",
        "mean_qed",
        "mean_vina_raw", "mean_vina_reward", "mean_sa_raw", "mean_sa_reward",
        "mean_logp_raw", "mean_logp_reward",
        "raw_sample_count", "dedup_count", "raw_unique_rate", "valid_count",
        "seq_len_mean", "seq_len_max",
        "vina_success_count", "vina_failed_count",
        "policy_loss", "kl_loss", "aux_loss", "total_loss", "ratio_mean", "true_token_kl_mean",
        "adv_std", "adv_max_abs",
        "lr", "kl_beta", "target_kl", "kl_beta_adjusted",
        "kept_samples", "updates_applied", "grad_norm",
        "min_unique_samples", "min_raw_unique_rate",
        "elite_size", "elite_added", "aux_coef",
        "reward_ema_rel_improve", "top5_ema_rel_improve",
        "plateau_hits", "plateau_active", "kl_stop_hits", "collapse_skip_hits",
    ]
    pd.DataFrame(columns=columns).to_csv(log_path, index=False)

    elite_buffer = EliteBuffer(max_size=elite_buffer_size)
    advantage_baseline = (
        RewardMovingBaseline(device=device)
        if (advantage_mode or "").lower() == "moving_baseline"
        else None
    )
    best_total_loss = float("inf")
    best_reward = float("-inf")
    best_top5_ema = float("-inf")
    best_total_loss_state = None
    best_reward_state = None
    best_top5_ema_state = None

    reward_ema = None
    top5_ema = None
    reward_ema_history = []
    top5_ema_history = []
    plateau_hits = 0
    kl_stop_hits = 0
    collapse_skip_hits = 0
    stop_training = False

    for it in range(1, iterations + 1):
        print(f"\n{'=' * 70}")
        print(f" Iteration {it}/{iterations} (GSPO 5_6 + Elite Aux)")
        print(f"{'=' * 70}")

        ref_agent.load_state_dict(agent.state_dict())
        for p in ref_agent.parameters():
            p.requires_grad_(False)

        for step in range(1, M + 1):
            print(f" [Iter {it}, Step {step}/{M}]")
            global_step = (it - 1) * M + step

            old_agent.load_state_dict(agent.state_dict())
            for p in old_agent.parameters():
                p.requires_grad_(False)

            batch_token_ids, batch_smiles = sample_selfies_batch_from_generate_selfies(
                vocab=vocab,
                model=old_agent,
                batch_size=batch_size,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                device=device,
                include_eos=True,
            )
            if len(batch_token_ids) < 2:
                print("   Skip: no valid samples")
                continue

            batch_token_ids, batch_smiles, dedup_stats = deduplicate_rollout_batch(
                batch_token_ids,
                batch_smiles,
                enabled=deduplicate_batch,
            )
            raw_unique_rate = (
                float(dedup_stats["dedup_count"]) / max(float(dedup_stats["raw_sample_count"]), 1.0)
            )
            if (
                deduplicate_batch
                and (
                    len(batch_token_ids) < min_unique_samples
                    or raw_unique_rate < min_raw_unique_rate
                )
            ):
                collapse_skip_hits += 1
                print(
                    "   Skip: samples collapsed after canonical dedup "
                    f"({dedup_stats['dedup_count']}/{dedup_stats['raw_sample_count']}, "
                    f"raw_unique_rate={raw_unique_rate:.3f}, "
                    f"min_unique={min_unique_samples}, min_rate={min_raw_unique_rate:.3f})"
                )
                if collapse_skip_hits >= collapse_stop_patience:
                    print(
                        "   Early stopping: rollout diversity collapsed for "
                        f"{collapse_skip_hits} consecutive checks"
                    )
                    stop_training = True
                    break
                continue

            batch_old_logprob_stats = compute_old_logprob_stats_batch(
                old_agent,
                batch_token_ids,
                vocab,
                device=device,
                pad_token_id=pad_token_id,
                temperature=temperature,
                top_k=top_k,
            )
            batch_old_logprobs = [
                None if stats is None else stats["logp_mean"]
                for stats in batch_old_logprob_stats
            ]

            vina_results = batch_scores_from_vina(
                batch_smiles,
                receptor_file=os.path.join(BASE_DIR, "feedback/8sc7.pdbqt"),
                pdbqt_dir=os.path.join(BASE_DIR, "feedback/temp/pdbqt"),
                output_dir=os.path.join(BASE_DIR, "feedback/vina_results/"),
            )
            vina_counts = _count_vina_status(vina_results)

            reward_items = make_composite_reward(
                batch_smiles,
                vina_results,
                weights=reward_weights,
                invalid_penalty=0,
            )
            rewards = [item["reward"] for item in reward_items]

            valid = []
            for token_ids, reward, smiles, old_lp, old_stats, item in zip(
                batch_token_ids,
                rewards,
                batch_smiles,
                batch_old_logprobs,
                batch_old_logprob_stats,
                reward_items,
            ):
                if reward != 0 and old_lp is not None and old_stats is not None:
                    valid.append((token_ids, float(reward), smiles, old_lp, old_stats, item))

            if len(valid) < min_unique_samples:
                collapse_skip_hits += 1
                print(
                    "   Skip: not enough valid unique rewards "
                    f"({len(valid)}/{min_unique_samples})"
                )
                if collapse_skip_hits >= collapse_stop_patience:
                    print(
                        "   Early stopping: too few valid unique rewards for "
                        f"{collapse_skip_hits} consecutive checks"
                    )
                    stop_training = True
                    break
                continue

            batch_token_ids = [x[0] for x in valid]
            rewards = [x[1] for x in valid]
            batch_old_logprobs = [x[3] for x in valid]
            batch_old_logprob_stats = [x[4] for x in valid]
            valid_items = [x[5] for x in valid]
            preceding_collapse_skip_hits = collapse_skip_hits
            collapse_skip_hits = 0

            batch_advantages, adv_stats = prepare_advantages(
                rewards,
                device=device,
                mode=advantage_mode,
                baseline=advantage_baseline,
                clip_adv=clip_adv,
            )

            elite_added = 0
            for token_ids, reward, smiles, old_lp, _old_stats, item in valid:
                if elite_buffer.add(token_ids, smiles, reward, old_lp, item, global_step):
                    elite_added += 1

            mean_reward = float(np.mean(rewards))
            top1_reward = float(max(rewards))
            top5_mean_reward = float(np.mean(sorted(rewards, reverse=True)[:min(5, len(rewards))]))
            mean_qed = _mean_or_none([item.get("qed") for item in valid_items])
            mean_vina_raw = _mean_or_none([item.get("vina_raw") for item in valid_items])
            mean_vina_reward = _mean_or_none([item.get("vina_reward") for item in valid_items])
            mean_sa_raw = _mean_or_none([item.get("sa_raw") for item in valid_items])
            mean_sa_reward = _mean_or_none([item.get("sa_reward") for item in valid_items])
            mean_logp_raw = _mean_or_none([item.get("logp_raw") for item in valid_items])
            mean_logp_reward = _mean_or_none([item.get("logp_reward") for item in valid_items])
            valid_count = len(valid_items)

            reward_ema = _update_ema(reward_ema, mean_reward, ema_alpha)
            top5_ema = _update_ema(top5_ema, top5_mean_reward, ema_alpha)
            reward_ema_history.append(reward_ema)
            top5_ema_history.append(top5_ema)

            current_lr = optimizer.param_groups[0]["lr"]
            grad_norm_value = 0.0
            updates_applied = 0
            kept_count = 0
            pol_loss = torch.tensor(0.0, device=device)
            kl_loss = torch.tensor(0.0, device=device)
            gspo_loss = torch.tensor(0.0, device=device)
            aux_loss = torch.tensor(0.0, device=device)
            total_loss = torch.tensor(0.0, device=device)
            r_mean = torch.tensor(1.0, device=device)

            for _ in range(mu):
                gspo_loss, pol_loss, kl_loss, r_mean, kept_count = compute_gspo_loss_batch(
                    agent=agent,
                    old_agent=old_agent,
                    ref_agent=ref_agent,
                    batch_sequences=batch_token_ids,
                    batch_rewards=rewards,
                    vocab=vocab,
                    clip_eps=clip_eps,
                    kl_beta=kl_beta,
                    pad_token_id=pad_token_id,
                    stored_old_logprobs=batch_old_logprobs,
                    ratio_min=ratio_min,
                    ratio_max=ratio_max,
                    temperature=temperature,
                    top_k=top_k,
                    batch_advantages=batch_advantages,
                )

                elite_samples = []
                if global_step >= aux_start:
                    elite_samples = elite_buffer.sample(
                        aux_batch_size,
                        reward_temperature=aux_reward_temperature,
                    )
                aux_loss = elite_aux_likelihood_loss(
                    agent,
                    elite_samples,
                    vocab,
                    device,
                    pad_token_id=pad_token_id,
                    temperature=temperature,
                    reward_temperature=aux_reward_temperature,
                )

                if kept_count == 0 and not elite_samples:
                    print("   Skip update: no GSPO samples and no elite aux samples")
                    continue

                total_loss = gspo_loss + aux_coef * aux_loss
                optimizer.zero_grad()
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, agent.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]
                grad_norm_value = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)
                updates_applied += 1

            ratio_diagnostics = compute_sequence_ratio_diagnostics(
                agent=agent,
                ref_agent=ref_agent,
                batch_sequences=batch_token_ids,
                old_logprob_stats=batch_old_logprob_stats,
                vocab=vocab,
                device=device,
                pad_token_id=pad_token_id,
                temperature=temperature,
                top_k=top_k,
            )
            true_token_kl_mean = float(ratio_diagnostics["true_token_kl_mean"])

            _reward_ema_prev, _reward_ema_curr, reward_ema_improve = _window_metrics(
                reward_ema_history, plateau_window
            )
            _top5_ema_prev, _top5_ema_curr, top5_ema_improve = _window_metrics(
                top5_ema_history, plateau_window
            )

            plateau_active = 0
            if (
                reward_ema_improve is not None
                and top5_ema_improve is not None
                and global_step >= plateau_min_steps
            ):
                plateau_active = int(
                    reward_ema_improve < plateau_tol
                    and top5_ema_improve < plateau_tol
                    and updates_applied > 0
                )
                plateau_hits = plateau_hits + 1 if plateau_active else 0
            else:
                plateau_hits = 0

            if global_step >= kl_stop_min_steps and true_token_kl_mean > kl_stop_threshold:
                kl_stop_hits += 1
            else:
                kl_stop_hits = 0

            if updates_applied > 0:
                kl_beta, kl_beta_adjusted = adjust_kl_beta(
                    kl_beta,
                    true_token_kl_mean,
                    target_kl,
                    kl_beta_min,
                    kl_beta_max,
                )
            else:
                kl_beta_adjusted = 0

            row = {
                "iteration": it,
                "step": step,
                "global_step": global_step,
                "mean_reward": mean_reward,
                "mean_reward_ema": reward_ema,
                "top1_reward": top1_reward,
                "top5_mean_reward": top5_mean_reward,
                "top5_reward_ema": top5_ema,
                "mean_qed": mean_qed,
                "mean_vina_raw": mean_vina_raw,
                "mean_vina_reward": mean_vina_reward,
                "mean_sa_raw": mean_sa_raw,
                "mean_sa_reward": mean_sa_reward,
                "mean_logp_raw": mean_logp_raw,
                "mean_logp_reward": mean_logp_reward,
                "raw_sample_count": dedup_stats["raw_sample_count"],
                "dedup_count": dedup_stats["dedup_count"],
                "raw_unique_rate": raw_unique_rate,
                "valid_count": valid_count,
                "seq_len_mean": ratio_diagnostics["seq_len_mean"],
                "seq_len_max": ratio_diagnostics["seq_len_max"],
                "vina_success_count": vina_counts["success"],
                "vina_failed_count": vina_counts["failed"],
                "policy_loss": float(pol_loss.item()),
                "kl_loss": float(kl_loss.item()),
                "aux_loss": float(aux_loss.item()),
                "total_loss": float(total_loss.item()),
                "ratio_mean": float(r_mean.item()),
                "true_token_kl_mean": true_token_kl_mean,
                "adv_std": adv_stats["adv_std"],
                "adv_max_abs": adv_stats["adv_max_abs"],
                "lr": current_lr,
                "kl_beta": kl_beta,
                "target_kl": target_kl,
                "kl_beta_adjusted": kl_beta_adjusted,
                "kept_samples": kept_count,
                "updates_applied": updates_applied,
                "grad_norm": grad_norm_value,
                "min_unique_samples": min_unique_samples,
                "min_raw_unique_rate": min_raw_unique_rate,
                "elite_size": len(elite_buffer),
                "elite_added": elite_added,
                "aux_coef": aux_coef,
                "reward_ema_rel_improve": reward_ema_improve,
                "top5_ema_rel_improve": top5_ema_improve,
                "plateau_hits": plateau_hits,
                "plateau_active": plateau_active,
                "kl_stop_hits": kl_stop_hits,
                "collapse_skip_hits": preceding_collapse_skip_hits,
            }
            pd.DataFrame([row], columns=columns).to_csv(
                log_path, mode="a", index=False, header=False
            )

            print(
                f"   meanR={mean_reward:.3f} | ema={reward_ema:.3f} | "
                f"top1={top1_reward:.3f} | top5={top5_mean_reward:.3f} | top5_ema={top5_ema:.3f}\n"
                f"   vina={mean_vina_reward:.3f} raw={mean_vina_raw if mean_vina_raw is not None else 0:.3f} | "
                f"qed={mean_qed:.3f} | sa={mean_sa_reward:.3f} | logp={mean_logp_reward:.3f}\n"
                f"   policy={pol_loss.item():.4f} | kl={kl_loss.item():.4f} | "
                f"aux={aux_loss.item():.4f} | total={total_loss.item():.4f}\n"
                f"   ratio={r_mean.item():.3f} | lr={current_lr:.2e} | grad={grad_norm_value:.4f} | "
                f"elite={len(elite_buffer)}(+{elite_added}) | kept={kept_count}\n"
                f"   ratio_sum={ratio_diagnostics['ratio_sum_mean']:.3f} "
                f"[{ratio_diagnostics['ratio_sum_min']:.3f}, {ratio_diagnostics['ratio_sum_max']:.3f}] | "
                f"true_kl={ratio_diagnostics['true_token_kl_mean']:.4f} | "
                f"eos={ratio_diagnostics['eos_included_rate']:.2f}\n"
                f"   dedup={dedup_stats['dedup_count']}/{dedup_stats['raw_sample_count']} "
                f"(-{dedup_stats['duplicate_removed']}) | raw_unique={raw_unique_rate:.3f} | "
                f"adv_std={adv_stats['adv_std']:.3f} | kl_beta={kl_beta:.4f}"
            )

            if plateau_hits >= plateau_patience:
                print(
                    "   Early stopping: reward EMA and top5 EMA plateau detected "
                    f"for {plateau_hits} consecutive checks"
                )
                stop_training = True
                break

            if kl_stop_hits >= kl_stop_patience:
                print(
                    "   Early stopping: true-token KL exceeded threshold "
                    f"{kl_stop_threshold:.4f} for {kl_stop_hits} consecutive checks"
                )
                stop_training = True
                break

            if total_loss.item() < best_total_loss:
                best_total_loss = total_loss.item()
                best_total_loss_state = copy.deepcopy(agent.state_dict())
                print(f"   New best total loss: {best_total_loss:.4f}")

            if mean_reward > best_reward:
                best_reward = mean_reward
                best_reward_state = copy.deepcopy(agent.state_dict())
                print(f"   New best reward: {best_reward:.3f}")

            if top5_ema > best_top5_ema:
                best_top5_ema = top5_ema
                best_top5_ema_state = copy.deepcopy(agent.state_dict())
                print(f"   New best top5 EMA reward: {best_top5_ema:.3f}")

        if stop_training:
            break

    elite_buffer.save_csv(elite_path)
    print(f" Training log saved to: {log_path}")
    print(f" Elite buffer saved to: {elite_path}")

    if best_total_loss_state is not None:
        save_path = os.path.join(save_dir, f"best_total_loss_5_6_{timestamp}.pt")
        torch.save(best_total_loss_state, save_path)
        print(f" Saved best loss model: {save_path}")

    if best_reward_state is not None:
        save_path = os.path.join(save_dir, f"best_reward_5_6_{timestamp}.pt")
        torch.save(best_reward_state, save_path)
        print(f" Saved best reward model: {save_path}")

    if best_top5_ema_state is not None:
        save_path = os.path.join(save_dir, f"best_top5_ema_5_6_{timestamp}.pt")
        torch.save(best_top5_ema_state, save_path)
        print(f" Saved best top5 EMA model: {save_path}")


if __name__ == "__main__":
    from config.load_config import load_config
    from data.data_utils import selfies_vocab
    from model.decoder_only_tfm import decoder_only_tfm

    set_seed(42)
    random.seed(42)
    np.random.seed(42)

    config = load_config(os.path.join(BASE_DIR, "config/decoder_only_tfm_config.yaml"))
    data = pd.read_csv(os.path.join(BASE_DIR, "data/htvs_molecules_with_selfies.csv"))["selfies"].tolist()
    vocab = selfies_vocab(data)

    device = config["device"]
    model_cfg = config["model"]

    agent = decoder_only_tfm(
        vocab_size=len(vocab),
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        max_len=model_cfg["max_len"],
        dropout=model_cfg["dropout"],
    ).to(device)

    agent.load_state_dict(
        torch.load(
            os.path.join(BASE_DIR, "train/model/decoder_only_tfm/best_model_fold1.pt"),
            map_location=device,
            weights_only=True,
        ),
        strict=False,
    )

    freeze_agent_last_n_layers(agent, n_trainable_layers=2)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, agent.parameters()),
        lr=5e-5,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    scheduler = None

    train_gspo_5_6(
        agent,
        vocab,
        optimizer,
        scheduler,
        device=device,
        iterations=1,
        M=500,
        batch_size=16,
        mu=4,
        clip_eps=0.25,
        kl_beta=0.10,
        temperature=1.1,
        top_k=30,
        reward_weights={"vina": 1.0, "qed": 0.3, "sa": 0.2, "logp": 0.0},
        log_dir=os.path.join(BASE_DIR, "rl/feedback/logs"),
        save_dir=os.path.join(BASE_DIR, "rl/feedback/best_models"),
        elite_buffer_size=300,
        aux_batch_size=8,
        aux_coef=0.0,#0.01
        aux_start=50,
        aux_reward_temperature=0.08,
        ratio_min=0.5,
        ratio_max=1.5,
        pad_token_id=0,
        ema_alpha=0.1,
        plateau_window=30,
        plateau_patience=8,
        plateau_tol=0.003,
        plateau_min_steps=150,
        kl_stop_threshold=0.8,
        kl_stop_patience=3,
        kl_stop_min_steps=80,
        deduplicate_batch=True,
        min_unique_samples=8,
        min_raw_unique_rate=0.5,
        collapse_stop_patience=12,
        advantage_mode="batch_zscore",
        clip_adv=3.0,
        target_kl=0.05,
        kl_beta_min=1e-3,
        kl_beta_max=2.0,
    )
