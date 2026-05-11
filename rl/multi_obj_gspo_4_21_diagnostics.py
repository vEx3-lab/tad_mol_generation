import os
import sys
import copy
import math
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, RL_DIR)

import numpy as np
import pandas as pd
import selfies as sf
import torch
import torch.nn.functional as F

from feedback.utils import canonicalize_smiles
from feedback.vina_scores import batch_scores_from_vina
from rl.multi_obj_gspo_4_21 import (
    ReplayBuffer,
    freeze_agent_partial,
    set_seed,
)
from utils import make_composite_reward, sample_selfies_batch_from_generate_selfies


def _mean_or_none(values):
    values = [x for x in values if x is not None and not pd.isna(x)]
    if not values:
        return None
    return float(np.mean(values))


def _relative_improvement(prev_value, curr_value, eps=1e-8):
    if prev_value is None or curr_value is None:
        return None
    return float((curr_value - prev_value) / (abs(prev_value) + eps))


def _window_metrics(history, window):
    if len(history) < 2 * window:
        return None, None, None
    prev_mean = float(np.mean(history[-2 * window:-window]))
    curr_mean = float(np.mean(history[-window:]))
    return prev_mean, curr_mean, _relative_improvement(prev_mean, curr_mean)


def _update_ema(prev, value, alpha):
    if prev is None:
        return float(value)
    return float(alpha * value + (1.0 - alpha) * prev)


def _count_vina_status(vina_results):
    counts = {"cached": 0, "success": 0, "failed": 0}
    for item in vina_results.values():
        status = item.get("status", "unknown") if isinstance(item, dict) else "unknown"
        if status == "cached":
            counts["cached"] += 1
        elif status == "success":
            counts["success"] += 1
        else:
            counts["failed"] += 1
    return counts


def _unique_smiles_rate(smiles_list):
    canonical = [canonicalize_smiles(smi) for smi in smiles_list]
    canonical = [smi for smi in canonical if smi is not None]
    if not canonical:
        return 0.0
    return float(len(set(canonical)) / len(canonical))


def _mean_length(token_ids_list):
    if not token_ids_list:
        return 0.0
    return float(np.mean([len(x) for x in token_ids_list]))


def _id_to_token(vocab, token_id):
    id2token = getattr(vocab, "id2token", None)
    if isinstance(id2token, dict):
        return id2token[int(token_id)]
    return id2token[int(token_id)]


def _token_to_id(vocab, token):
    token2id = getattr(vocab, "token2id", None)
    if token2id is not None:
        return token2id[token]
    return vocab[token]


def _token_in_vocab(vocab, token):
    token2id = getattr(vocab, "token2id", None)
    if token2id is not None:
        return token in token2id
    try:
        vocab[token]
        return True
    except Exception:
        return False


def _vocab_tokens(vocab):
    if hasattr(vocab, "tokens") and callable(vocab.tokens):
        return list(vocab.tokens())
    token2id = getattr(vocab, "token2id", None)
    if token2id is not None:
        return list(token2id.keys())
    return [_id_to_token(vocab, i) for i in range(len(vocab))]


def _allowed_next_ids_from_prefix(vocab, prefix_token_ids, device):
    prefix_tokens = [_id_to_token(vocab, idx) for idx in prefix_token_ids]
    core_tokens = [tok for tok in prefix_tokens if tok not in ("<SOS>", "<EOS>", "<PAD>")]
    prefix_selfies = "".join(core_tokens)

    allowed_tokens = None
    try:
        next_tokens = sf.next_selfies_tokens(prefix_selfies)
        if next_tokens:
            allowed_tokens = set(next_tokens)
    except Exception:
        allowed_tokens = None

    if allowed_tokens is None:
        try:
            constraints = sf.get_semantic_constraints()
            allowed_tokens = set(constraints.keys())
        except Exception:
            allowed_tokens = set(_vocab_tokens(vocab))

    allowed_ids = [
        _token_to_id(vocab, token)
        for token in allowed_tokens
        if _token_in_vocab(vocab, token)
    ]
    if not allowed_ids:
        return None
    return torch.tensor(allowed_ids, dtype=torch.long, device=device)


def _sampling_logits_for_prefix(model, input_ids, vocab, prefix_ids, device, temperature):
    temp = temperature if temperature and temperature > 0 else 1.0
    logits = model(input_ids)[0, -1, :] / temp
    allowed_ids = _allowed_next_ids_from_prefix(vocab, prefix_ids, device)
    if allowed_ids is not None:
        mask = torch.zeros_like(logits, dtype=torch.bool, device=device)
        mask[allowed_ids] = True
        logits = logits.masked_fill(~mask, -1e9)
    return logits


def _compute_sampling_token_logps(
    model,
    sequence,
    vocab,
    device,
    pad_token_id=0,
    temperature=1.0,
    top_k=None,
    eps=1e-12,
):
    seq = [int(x) for x in sequence if int(x) != pad_token_id]
    if len(seq) == 0:
        return None, False

    if _token_in_vocab(vocab, "<SOS>"):
        prefix_ids = [_token_to_id(vocab, "<SOS>")]
    else:
        prefix_ids = []

    eos_id = _token_to_id(vocab, "<EOS>") if _token_in_vocab(vocab, "<EOS>") else None
    token_logps = []

    for target_id in seq:
        if not prefix_ids:
            input_ids = torch.tensor([[target_id]], dtype=torch.long, device=device)
        else:
            input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)

        logits = _sampling_logits_for_prefix(
            model,
            input_ids,
            vocab,
            prefix_ids,
            device,
            temperature,
        )

        if top_k is not None:
            k = min(int(top_k), logits.numel())
            top_logits, top_idx = torch.topk(logits, k)
            matches = (top_idx == int(target_id)).nonzero(as_tuple=False)
            if matches.numel() == 0:
                token_logp = logits.new_tensor(math.log(eps))
            else:
                token_logp = F.log_softmax(top_logits, dim=-1)[matches[0, 0]]
        else:
            token_logp = F.log_softmax(logits, dim=-1)[int(target_id)]

        token_logps.append(token_logp)
        prefix_ids.append(int(target_id))

    return torch.stack(token_logps), any(int(x) == eos_id for x in seq) if eos_id is not None else False


def compute_sampling_sequence_logprob(
    model,
    sequence,
    vocab,
    device,
    pad_token_id=0,
    temperature=1.0,
    top_k=None,
    eps=1e-12,
):
    """
    Sequence logprob under the same distribution used by generate_selfies:
    SELFIES mask + temperature + optional top-k renormalization.
    """
    token_logps, _ = _compute_sampling_token_logps(
        model,
        sequence,
        vocab,
        device,
        pad_token_id=pad_token_id,
        temperature=temperature,
        top_k=top_k,
        eps=eps,
    )
    if token_logps is None:
        return None
    return token_logps.mean()


def compute_sampling_sequence_logprob_stats(
    model,
    sequence,
    vocab,
    device,
    pad_token_id=0,
    temperature=1.0,
    top_k=None,
    eps=1e-12,
):
    token_logps, has_eos = _compute_sampling_token_logps(
        model,
        sequence,
        vocab,
        device,
        pad_token_id=pad_token_id,
        temperature=temperature,
        top_k=top_k,
        eps=eps,
    )
    if token_logps is None:
        return None
    return {
        "logp_mean": token_logps.mean(),
        "logp_sum": token_logps.sum(),
        "token_count": int(token_logps.numel()),
        "has_eos": bool(has_eos),
    }


@torch.no_grad()
def compute_old_logprob_batch(
    model,
    batch_sequences,
    vocab,
    device,
    pad_token_id=0,
    temperature=1.0,
    top_k=None,
):
    model.eval()
    seq_logprobs = []
    for seq in batch_sequences:
        seq_logp = compute_sampling_sequence_logprob(
            model,
            seq,
            vocab,
            device,
            pad_token_id=pad_token_id,
            temperature=temperature,
            top_k=top_k,
        )
        seq_logprobs.append(None if seq_logp is None else float(seq_logp.item()))
    return seq_logprobs


@torch.no_grad()
def compute_old_logprob_stats_batch(
    model,
    batch_sequences,
    vocab,
    device,
    pad_token_id=0,
    temperature=1.0,
    top_k=None,
):
    model.eval()
    batch_stats = []
    for seq in batch_sequences:
        stats = compute_sampling_sequence_logprob_stats(
            model,
            seq,
            vocab,
            device,
            pad_token_id=pad_token_id,
            temperature=temperature,
            top_k=top_k,
        )
        if stats is None:
            batch_stats.append(None)
        else:
            batch_stats.append(
                {
                    "logp_mean": float(stats["logp_mean"].item()),
                    "logp_sum": float(stats["logp_sum"].item()),
                    "token_count": int(stats["token_count"]),
                    "has_eos": bool(stats["has_eos"]),
                }
            )
    return batch_stats


def compute_sampled_prefix_token_kl(
    agent,
    ref_agent,
    sequence,
    vocab,
    device,
    pad_token_id=0,
    temperature=1.0,
):
    seq = [int(x) for x in sequence if int(x) != pad_token_id]
    if len(seq) == 0 or not _token_in_vocab(vocab, "<SOS>"):
        return None

    prefix_ids = [_token_to_id(vocab, "<SOS>")]
    kls = []
    for target_id in seq:
        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        logits_new = _sampling_logits_for_prefix(
            agent,
            input_ids,
            vocab,
            prefix_ids,
            device,
            temperature,
        )
        logits_ref = _sampling_logits_for_prefix(
            ref_agent,
            input_ids,
            vocab,
            prefix_ids,
            device,
            temperature,
        )
        logp_new = F.log_softmax(logits_new, dim=-1)
        logp_ref = F.log_softmax(logits_ref, dim=-1)
        p_new = logp_new.exp()
        kls.append(torch.sum(p_new * (logp_new - logp_ref)))
        prefix_ids.append(int(target_id))

    if not kls:
        return None
    return torch.stack(kls).mean()


@torch.no_grad()
def compute_sequence_ratio_diagnostics(
    agent,
    ref_agent,
    batch_sequences,
    old_logprob_stats,
    vocab,
    device,
    pad_token_id=0,
    temperature=1.0,
    top_k=None,
):
    ratio_sums = []
    log_ratio_sums = []
    seq_lens = []
    has_eos_values = []
    true_token_kls = []
    was_agent_training = agent.training
    was_ref_training = ref_agent.training
    agent.eval()
    ref_agent.eval()

    try:
        for seq, old_stats in zip(batch_sequences, old_logprob_stats):
            if old_stats is None:
                continue
            new_stats = compute_sampling_sequence_logprob_stats(
                agent,
                seq,
                vocab,
                device,
                pad_token_id=pad_token_id,
                temperature=temperature,
                top_k=top_k,
            )
            if new_stats is None:
                continue

            log_ratio_sum = float(new_stats["logp_sum"].item()) - float(old_stats["logp_sum"])
            log_ratio_sums.append(log_ratio_sum)
            ratio_sums.append(math.exp(max(min(log_ratio_sum, 60.0), -60.0)))
            seq_lens.append(int(new_stats["token_count"]))
            has_eos_values.append(float(new_stats["has_eos"]))

            token_kl = compute_sampled_prefix_token_kl(
                agent,
                ref_agent,
                seq,
                vocab,
                device,
                pad_token_id=pad_token_id,
                temperature=temperature,
            )
            if token_kl is not None:
                true_token_kls.append(float(token_kl.item()))
    finally:
        agent.train(was_agent_training)
        ref_agent.train(was_ref_training)

    if not ratio_sums:
        return {
            "ratio_sum_mean": 1.0,
            "ratio_sum_std": 0.0,
            "ratio_sum_min": 1.0,
            "ratio_sum_max": 1.0,
            "log_ratio_sum_mean": 0.0,
            "seq_len_mean": 0.0,
            "seq_len_max": 0,
            "eos_included_rate": 0.0,
            "true_token_kl_mean": 0.0,
        }

    return {
        "ratio_sum_mean": float(np.mean(ratio_sums)),
        "ratio_sum_std": float(np.std(ratio_sums)),
        "ratio_sum_min": float(np.min(ratio_sums)),
        "ratio_sum_max": float(np.max(ratio_sums)),
        "log_ratio_sum_mean": float(np.mean(log_ratio_sums)),
        "seq_len_mean": float(np.mean(seq_lens)),
        "seq_len_max": int(np.max(seq_lens)),
        "eos_included_rate": float(np.mean(has_eos_values)),
        "true_token_kl_mean": _mean_or_none(true_token_kls) or 0.0,
    }


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
):
    device = next(agent.parameters()).device
    rewards = torch.tensor(batch_rewards, device=device)
    adv = (rewards - rewards.mean()) / (rewards.std() + eps)
    adv = adv.detach()

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


def train_gspo_diagnostics(
    agent,
    vocab,
    optimizer,
    scheduler,
    device="cuda",
    iterations=100,
    M=4,
    batch_size=16,
    max_len=80,
    mu=1,
    clip_eps=0.2,
    kl_beta=0.2,
    temperature=1.0,
    top_k=None,
    log_dir=os.path.join(BASE_DIR, "rl/feedback/logs"),
    save_dir=os.path.join(BASE_DIR, "rl/feedback/best_models"),
    replay_buffer_size=300,
    replay_batch_size=0,
    replay_start=50,
    replay_max_age_iters=5,
    ratio_min=0.5,
    ratio_max=1.5,
    pad_token_id=0,
    ema_alpha=0.1,
    plateau_window=20,
    plateau_patience=5,
    plateau_tol=0.005,
    plateau_min_steps=80,
    kl_stop_threshold=5.0,
    kl_stop_patience=3,
    kl_stop_min_steps=50,
):
    agent.to(device)
    old_agent = copy.deepcopy(agent).to(device)
    ref_agent = copy.deepcopy(agent).to(device)

    old_agent.eval()
    ref_agent.eval()

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"gspo_diagnostics_log_{timestamp}.csv")

    pd.DataFrame(columns=[
        "iteration", "step", "global_step",
        "mean_reward", "mean_reward_ema", "top1_reward", "top5_mean_reward", "top5_reward_ema",
        "mean_vina", "mean_qed", "mean_sa", "mean_logp",
        "mean_vina_raw", "mean_vina_reward", "mean_sa_raw", "mean_sa_reward",
        "mean_logp_raw", "mean_logp_reward",
        "valid_count", "unique_smiles_rate", "mean_length",
        "vina_cached_count", "vina_success_count", "vina_failed_count",
        "policy_loss", "kl_loss", "total_loss", "ratio_mean",
        "lr", "lr_before_update", "lr_after_update",
        "kl_beta", "replay_size", "kept_samples", "updates_applied",
        "grad_norm", "weight_decay", "clip_eps", "mu", "ratio_min", "ratio_max",
        "reward_ema_prev_mean", "reward_ema_curr_mean", "reward_ema_rel_improve",
        "top5_ema_prev_mean", "top5_ema_curr_mean", "top5_ema_rel_improve",
        "plateau_hits", "plateau_active", "kl_stop_hits",
    ]).to_csv(log_path, index=False)

    best_total_loss = float("inf")
    best_reward = float("-inf")
    best_top5_reward_ema = float("-inf")
    best_total_loss_state = None
    best_reward_state = None
    best_top5_ema_state = None

    replay_buffer = ReplayBuffer(
        max_size=replay_buffer_size,
        max_age_iters=replay_max_age_iters,
    )

    reward_ema = None
    top5_reward_ema = None
    reward_ema_history = []
    top5_ema_history = []
    plateau_hits = 0
    kl_stop_hits = 0
    stop_training = False

    for it in range(1, iterations + 1):
        print(f"\n{'=' * 70}")
        print(f" Iteration {it}/{iterations} (GSPO Diagnostics)")
        print(f"{'=' * 70}")

        ref_agent.load_state_dict(agent.state_dict())
        for p in ref_agent.parameters():
            p.requires_grad_(False)

        replay_buffer.purge_old(it)

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
            )

            if len(batch_token_ids) < 2:
                print("   Skip: no valid samples")
                continue

            batch_old_logprobs = compute_old_logprob_batch(
                old_agent,
                batch_token_ids,
                vocab,
                device=device,
                pad_token_id=pad_token_id,
                temperature=temperature,
                top_k=top_k,
            )

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
                weights={"vina": 1.0, "qed": 0.2, "sa": 0.2, "logp": 0.0},
                invalid_penalty=0,
            )
            rewards = [item["reward"] for item in reward_items]

            valid = []
            for tok, reward, smi, old_lp, item in zip(
                batch_token_ids,
                rewards,
                batch_smiles,
                batch_old_logprobs,
                reward_items,
            ):
                if reward != 0 and old_lp is not None:
                    valid.append((tok, reward, smi, old_lp, item))

            if len(valid) < 2:
                print("   Skip: invalid rewards")
                continue

            batch_token_ids, rewards, batch_smiles_valid, batch_old_logprobs, valid_items = zip(*valid)
            batch_token_ids = list(batch_token_ids)
            rewards = list(rewards)
            batch_smiles_valid = list(batch_smiles_valid)
            batch_old_logprobs = list(batch_old_logprobs)
            valid_items = list(valid_items)

            mean_reward = float(np.mean(rewards))
            top1_reward = float(max(rewards))
            top5_mean_reward = float(np.mean(sorted(rewards, reverse=True)[:min(5, len(rewards))]))

            mean_vina = _mean_or_none([item.get("vina") for item in valid_items])
            mean_qed = _mean_or_none([item.get("qed") for item in valid_items])
            mean_sa = _mean_or_none([item.get("sa") for item in valid_items])
            mean_logp = _mean_or_none([item.get("logp") for item in valid_items])
            mean_vina_raw = _mean_or_none([item.get("vina_raw") for item in valid_items])
            mean_vina_reward = _mean_or_none([item.get("vina_reward") for item in valid_items])
            mean_sa_raw = _mean_or_none([item.get("sa_raw") for item in valid_items])
            mean_sa_reward = _mean_or_none([item.get("sa_reward") for item in valid_items])
            mean_logp_raw = _mean_or_none([item.get("logp_raw") for item in valid_items])
            mean_logp_reward = _mean_or_none([item.get("logp_reward") for item in valid_items])
            valid_count = len(valid_items)
            unique_rate = _unique_smiles_rate(batch_smiles_valid)
            mean_length = _mean_length(batch_token_ids)

            reward_ema = _update_ema(reward_ema, mean_reward, ema_alpha)
            top5_reward_ema = _update_ema(top5_reward_ema, top5_mean_reward, ema_alpha)
            reward_ema_history.append(reward_ema)
            top5_ema_history.append(top5_reward_ema)

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

            for _ in range(mu):
                total_loss, pol_loss, kl_loss, r_mean, kept_count = compute_gspo_loss_batch(
                    agent=agent,
                    old_agent=old_agent,
                    ref_agent=ref_agent,
                    batch_sequences=mixed_token_ids,
                    batch_rewards=mixed_rewards,
                    vocab=vocab,
                    clip_eps=clip_eps,
                    kl_beta=kl_beta,
                    pad_token_id=pad_token_id,
                    stored_old_logprobs=mixed_old_logprobs,
                    ratio_min=ratio_min,
                    ratio_max=ratio_max,
                    temperature=temperature,
                    top_k=top_k,
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

            reward_ema_prev, reward_ema_curr, reward_ema_improve = _window_metrics(
                reward_ema_history, plateau_window
            )
            top5_ema_prev, top5_ema_curr, top5_ema_improve = _window_metrics(
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

            if global_step >= kl_stop_min_steps and kl_loss.item() > kl_stop_threshold:
                kl_stop_hits += 1
            else:
                kl_stop_hits = 0

            pd.DataFrame([{
                "iteration": it,
                "step": step,
                "global_step": global_step,
                "mean_reward": mean_reward,
                "mean_reward_ema": reward_ema,
                "top1_reward": top1_reward,
                "top5_mean_reward": top5_mean_reward,
                "top5_reward_ema": top5_reward_ema,
                "mean_vina": mean_vina,
                "mean_qed": mean_qed,
                "mean_sa": mean_sa,
                "mean_logp": mean_logp,
                "mean_vina_raw": mean_vina_raw,
                "mean_vina_reward": mean_vina_reward,
                "mean_sa_raw": mean_sa_raw,
                "mean_sa_reward": mean_sa_reward,
                "mean_logp_raw": mean_logp_raw,
                "mean_logp_reward": mean_logp_reward,
                "valid_count": valid_count,
                "unique_smiles_rate": unique_rate,
                "mean_length": mean_length,
                "vina_cached_count": vina_counts["cached"],
                "vina_success_count": vina_counts["success"],
                "vina_failed_count": vina_counts["failed"],
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
                "reward_ema_prev_mean": reward_ema_prev,
                "reward_ema_curr_mean": reward_ema_curr,
                "reward_ema_rel_improve": reward_ema_improve,
                "top5_ema_prev_mean": top5_ema_prev,
                "top5_ema_curr_mean": top5_ema_curr,
                "top5_ema_rel_improve": top5_ema_improve,
                "plateau_hits": plateau_hits,
                "plateau_active": plateau_active,
                "kl_stop_hits": kl_stop_hits,
            }]).to_csv(log_path, mode="a", index=False, header=False)

            print(
                f"   meanR={mean_reward:.3f} | ema={reward_ema:.3f} | "
                f"top1={top1_reward:.3f} | top5={top5_mean_reward:.3f} | "
                f"top5_ema={top5_reward_ema:.3f}\n"
                f"   vina={mean_vina_reward:.3f} raw={mean_vina_raw if mean_vina_raw is not None else 0:.3f} | "
                f"qed={mean_qed:.3f} | sa={mean_sa_reward:.3f} | logp={mean_logp_reward:.3f}\n"
                f"   policy={pol_loss.item():.4f} | kl={kl_loss.item():.4f} | "
                f"ratio={r_mean.item():.3f} | lr={current_lr:.2e} | grad={grad_norm_value:.4f}\n"
                f"   valid={valid_count} | unique={unique_rate:.3f} | len={mean_length:.1f} | "
                f"vina cache/success/fail={vina_counts['cached']}/{vina_counts['success']}/{vina_counts['failed']} | "
                f"plateau={plateau_hits}/{plateau_patience}"
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
                    "   Early stopping: KL loss exceeded threshold "
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

            if top5_reward_ema > best_top5_reward_ema:
                best_top5_reward_ema = top5_reward_ema
                best_top5_ema_state = copy.deepcopy(agent.state_dict())
                print(f"   New best top5 EMA reward: {best_top5_reward_ema:.3f}")

        if stop_training:
            break

    if best_total_loss_state is not None:
        save_path = os.path.join(save_dir, f"best_total_loss_diag_{timestamp}.pt")
        torch.save(best_total_loss_state, save_path)
        print(f" Saved best loss model: {save_path}")

    if best_reward_state is not None:
        save_path = os.path.join(save_dir, f"best_reward_diag_{timestamp}.pt")
        torch.save(best_reward_state, save_path)
        print(f" Saved best reward model: {save_path}")

    if best_top5_ema_state is not None:
        save_path = os.path.join(save_dir, f"best_top5_ema_diag_{timestamp}.pt")
        torch.save(best_top5_ema_state, save_path)
        print(f" Saved best top5 EMA model: {save_path}")

    print(f" Training log saved to: {log_path}")


if __name__ == "__main__":
    from config.load_config import load_config
    from data.data_utils import selfies_vocab
    from model.decoder_only_tfm import decoder_only_tfm

    set_seed(42)

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

    freeze_agent_partial(agent)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, agent.parameters()),
        lr=3e-6,
        betas=(0.9, 0.999),
        weight_decay=0.001,
    )

    scheduler = None

    train_gspo_diagnostics(
        agent,
        vocab,
        optimizer,
        scheduler,
        device=device,
        iterations=1,
        M=2000,
        batch_size=16,
        mu=1,
        clip_eps=0.2,
        kl_beta=0.2,
        temperature=1.0,
        top_k=None,
        log_dir=os.path.join(BASE_DIR, "rl/feedback/logs"),
        save_dir=os.path.join(BASE_DIR, "rl/feedback/best_models"),
        replay_buffer_size=300,
        replay_batch_size=0,
        replay_start=50,
        replay_max_age_iters=5,
        ratio_min=0.5,
        ratio_max=1.5,
        pad_token_id=0,
        ema_alpha=0.1,
        plateau_window=20,
        plateau_patience=5,
        plateau_tol=0.005,
        plateau_min_steps=80,
        kl_stop_threshold=5.0,
        kl_stop_patience=3,
        kl_stop_min_steps=50,
    )
