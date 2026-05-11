import os
import sys
import copy
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import torch

from feedback.vina_scores import batch_scores_from_vina
from rl.multi_obj_gspo_4_21 import (
    ReplayBuffer,
    compute_gspo_loss_batch,
    compute_old_logprob_batch,
    freeze_agent_partial,
    set_seed,
)
from utils import sample_selfies_batch_from_generate_selfies, make_composite_reward


def relative_improvement(prev_value, curr_value, eps=1e-8):
    return (curr_value - prev_value) / (abs(prev_value) + eps)


def compute_plateau_metrics(history, window):
    if len(history) < 2 * window:
        return None, None, None

    prev_mean = float(np.mean(history[-2 * window:-window]))
    curr_mean = float(np.mean(history[-window:]))
    improve = float(relative_improvement(prev_mean, curr_mean))
    return prev_mean, curr_mean, improve


def train_gspo_reward_plateau(
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
    replay_max_age_iters=5,
    ratio_min=0.5,
    ratio_max=1.5,
    pad_token_id=0,
    reward_plateau_window=20,
    reward_plateau_patience=3,
    reward_plateau_tol=0.015,
    reward_plateau_min_steps=80,
    reward_plateau_ratio_floor=0.65,
    reward_plateau_require_vina=True,
    kl_stop_threshold=2.0,
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
    log_path = os.path.join(log_dir, f"gspo_plateau_training_log_{timestamp}.csv")

    pd.DataFrame(columns=[
        "iteration", "step", "global_step",
        "mean_reward", "mean_vina", "mean_qed", "mean_sa", "mean_logp",
        "top1_reward", "top5_mean_reward",
        "policy_loss", "kl_loss", "total_loss", "ratio_mean",
        "lr", "lr_before_update", "lr_after_update",
        "kl_beta", "replay_size", "kept_samples", "updates_applied",
        "grad_norm", "weight_decay", "clip_eps", "mu", "ratio_min", "ratio_max",
        "reward_prev_mean", "reward_curr_mean", "reward_rel_improve",
        "top5_prev_mean", "top5_curr_mean", "top5_rel_improve",
        "vina_prev_mean", "vina_curr_mean", "vina_rel_improve",
        "reward_plateau_hits", "reward_plateau_active", "kl_stop_hits",
    ]).to_csv(log_path, index=False)

    best_total_loss = float("inf")
    best_reward = float("-inf")
    best_total_loss_state = None
    best_reward_state = None

    replay_buffer = ReplayBuffer(
        max_size=replay_buffer_size,
        max_age_iters=replay_max_age_iters,
    )

    reward_history = []
    top5_history = []
    vina_history = []
    ratio_history = []
    updates_history = []
    reward_plateau_hits = 0
    kl_stop_hits = 0
    stop_training = False

    for it in range(1, iterations + 1):
        print(f"\n{'=' * 70}")
        print(f" Iteration {it}/{iterations} (Reward Plateau)")
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
                device=device,
                pad_token_id=pad_token_id,
            )

            vina_results = batch_scores_from_vina(
                batch_smiles,
                receptor_file=os.path.join(BASE_DIR, "feedback/8sc7.pdbqt"),
                pdbqt_dir=os.path.join(BASE_DIR, "feedback/temp/pdbqt"),
                output_dir=os.path.join(BASE_DIR, "feedback/vina_results/"),
            )

            reward_items = make_composite_reward(
                batch_smiles,
                vina_results,
                weights={"vina": 1.0, "qed": 0.2, "sa": 0.2, "logp": 0.0},
                invalid_penalty=0,
            )
            rewards = [item["reward"] for item in reward_items]

            valid = [
                (tok, r, smi, old_lp)
                for tok, r, smi, old_lp in zip(batch_token_ids, rewards, batch_smiles, batch_old_logprobs)
                if r != 0 and old_lp is not None
            ]
            if len(valid) < 2:
                print("   Skip: invalid rewards")
                continue

            batch_token_ids, rewards, batch_smiles_valid, batch_old_logprobs = zip(*valid)
            valid_indices = [i for i, item in enumerate(reward_items) if item["reward"] != 0]
            valid_items = [reward_items[i] for i in valid_indices]

            mean_reward = float(np.mean(rewards))
            top1_reward = float(max(rewards))
            top5_mean_reward = float(np.mean(sorted(rewards, reverse=True)[:min(5, len(rewards))]))
            mean_vina = float(np.mean([item["vina"] for item in valid_items]))
            mean_qed = float(np.mean([item["qed"] for item in valid_items]))
            mean_sa = float(np.mean([item["sa"] for item in valid_items]))
            mean_logp = float(np.mean([item["logp"] for item in valid_items]))

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

            reward_history.append(mean_reward)
            top5_history.append(top5_mean_reward)
            vina_history.append(mean_vina)
            ratio_history.append(float(r_mean.item()))
            updates_history.append(updates_applied)

            reward_prev_mean, reward_curr_mean, reward_rel_improve = compute_plateau_metrics(
                reward_history, reward_plateau_window
            )
            top5_prev_mean, top5_curr_mean, top5_rel_improve = compute_plateau_metrics(
                top5_history, reward_plateau_window
            )
            vina_prev_mean, vina_curr_mean, vina_rel_improve = compute_plateau_metrics(
                vina_history, reward_plateau_window
            )

            reward_plateau_active = 0
            if (
                reward_rel_improve is not None
                and top5_rel_improve is not None
                and global_step >= reward_plateau_min_steps
            ):
                vina_condition = True
                if reward_plateau_require_vina:
                    vina_condition = (
                        vina_rel_improve is not None
                        and vina_rel_improve < reward_plateau_tol
                    )

                ratio_condition = float(np.mean(ratio_history[-reward_plateau_window:])) >= reward_plateau_ratio_floor
                update_condition = float(np.mean(updates_history[-reward_plateau_window:])) > 0.0

                reward_plateau_active = int(
                    reward_rel_improve < reward_plateau_tol
                    and top5_rel_improve < reward_plateau_tol
                    and vina_condition
                    and ratio_condition
                    and update_condition
                )

                if reward_plateau_active:
                    reward_plateau_hits += 1
                else:
                    reward_plateau_hits = 0
            else:
                reward_plateau_hits = 0

            if global_step >= kl_stop_min_steps and kl_loss.item() > kl_stop_threshold:
                kl_stop_hits += 1
            else:
                kl_stop_hits = 0

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
                "reward_prev_mean": reward_prev_mean,
                "reward_curr_mean": reward_curr_mean,
                "reward_rel_improve": reward_rel_improve,
                "top5_prev_mean": top5_prev_mean,
                "top5_curr_mean": top5_curr_mean,
                "top5_rel_improve": top5_rel_improve,
                "vina_prev_mean": vina_prev_mean,
                "vina_curr_mean": vina_curr_mean,
                "vina_rel_improve": vina_rel_improve,
                "reward_plateau_hits": reward_plateau_hits,
                "reward_plateau_active": reward_plateau_active,
                "kl_stop_hits": kl_stop_hits,
            }]).to_csv(log_path, mode="a", index=False, header=False)

            print(
                f"   meanR={mean_reward:.3f} | top1={top1_reward:.3f} | "
                f"mean_vina={mean_vina:.3f} | mean_qed={mean_qed:.3f} | "
                f"mean_sa={mean_sa:.3f} | mean_logp={mean_logp:.3f} | "
                f"top5={top5_mean_reward:.3f}\n"
                f"   policy={pol_loss.item():.4f} | kl={kl_loss.item():.4f} | "
                f"ratio={r_mean.item():.3f} | buffer={len(replay_buffer)} | kept={kept_count}\n"
                f"   lr={current_lr:.2e} | grad={grad_norm_value:.4f} | updates={updates_applied} | "
                f"plateau_hits={reward_plateau_hits}/{reward_plateau_patience} | "
                f"kl_hits={kl_stop_hits}/{kl_stop_patience}"
            )

            if reward_rel_improve is not None and top5_rel_improve is not None:
                vina_text = "n/a" if vina_rel_improve is None else f"{vina_rel_improve:.4f}"
                print(
                    f"   plateau check | d_reward={reward_rel_improve:.4f} | "
                    f"d_top5={top5_rel_improve:.4f} | d_vina={vina_text}"
                )

            if reward_plateau_hits >= reward_plateau_patience:
                print(
                    "   Early stopping: reward plateau detected "
                    f"for {reward_plateau_hits} consecutive checks"
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

        if stop_training:
            break

    if best_total_loss_state is not None:
        save_path = os.path.join(save_dir, f"best_total_loss_plateau_{timestamp}.pt")
        torch.save(best_total_loss_state, save_path)
        print(f" Saved best loss model: {save_path}")

    if best_reward_state is not None:
        save_path = os.path.join(save_dir, f"best_reward_plateau_{timestamp}.pt")
        torch.save(best_reward_state, save_path)
        print(f" Saved best reward model: {save_path}")


if __name__ == "__main__":
    from data.data_utils import selfies_vocab
    from config.load_config import load_config
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
        lr=5e-5,
        betas=(0.9, 0.999),
        weight_decay=0.001,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1000,
        eta_min=1e-7,
    )

    train_gspo_reward_plateau(
        agent,
        vocab,
        optimizer,
        scheduler,
        device=device,
        iterations=1,
        M=5000,
        batch_size=8,
        mu=3,
        clip_eps=0.2,
        kl_beta=0.1,
        temperature=1.0,
        top_k=20,
        log_dir=os.path.join(BASE_DIR, "rl/feedback/logs"),
        save_dir=os.path.join(BASE_DIR, "rl/feedback/best_models"),
        replay_buffer_size=300,
        replay_batch_size=0,
        replay_start=50,
        replay_max_age_iters=5,
        ratio_min=0.5,
        ratio_max=1.5,
        pad_token_id=0,
        reward_plateau_window=20,
        reward_plateau_patience=3,
        reward_plateau_tol=0.015,
        reward_plateau_min_steps=80,
        reward_plateau_ratio_floor=0.65,
        reward_plateau_require_vina=True,
        kl_stop_threshold=2.0,
        kl_stop_patience=3,
        kl_stop_min_steps=50,
    )
