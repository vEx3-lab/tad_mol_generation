# ===== GSPO 批量训练脚本 (Beam Search) =====
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from datetime import datetime
from feedback.vina_scores import batch_scores_from_vina
from utils import make_reward_fn_from_vina, make_composite_reward
from sample.sample_beam import sample_selfies_batch_beam
import random


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

        mask = (target_ids != pad_token_id).float()

        logits_new = agent(input_ids)
        logits_old = old_agent(input_ids).detach()
        logits_ref = ref_agent(input_ids).detach()

        logp_new = F.log_softmax(logits_new, dim=-1)
        logp_old = F.log_softmax(logits_old, dim=-1)
        logp_ref = F.log_softmax(logits_ref, dim=-1)

        token_logp_new = logp_new.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logp_old = logp_old.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logp_ref = logp_ref.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        valid_len = mask.sum() + eps

        log_ratio_policy = ((token_logp_new - token_logp_old) * mask).sum() / valid_len
        ratio_seq = torch.exp(log_ratio_policy)
        ratio_means.append(ratio_seq.detach())

        surr1 = ratio_seq * adv[i]
        surr2 = torch.clamp(ratio_seq, 1 - clip_eps, 1 + clip_eps) * adv[i]
        policy_losses.append(-torch.min(surr1, surr2))

        log_ratio_kl = ((token_logp_ref - token_logp_new) * mask).sum() / valid_len
        kl = torch.exp(log_ratio_kl) - log_ratio_kl - 1
        kl_losses.append(kl)

    policy_loss = torch.stack(policy_losses).mean()
    kl_loss = torch.stack(kl_losses).mean()
    ratio_mean = torch.stack(ratio_means).mean()

    total_loss = policy_loss + kl_beta * kl_loss
    return total_loss, policy_loss, kl_loss, ratio_mean


class ReplayBuffer:
    def __init__(self, max_size=300):
        self.buffer = []
        self.max_size = max_size

    def add(self, token_ids, smiles, reward):
        self.buffer.append((token_ids, smiles, reward))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return [], [], []
        indices = random.sample(range(len(self.buffer)), batch_size)
        token_ids = [self.buffer[i][0] for i in indices]
        smiles = [self.buffer[i][1] for i in indices]
        rewards = [self.buffer[i][2] for i in indices]
        return token_ids, smiles, rewards

    def __len__(self):
        return len(self.buffer)


def train_gspo_beam(
        agent,
        vocab,
        optimizer,
        scheduler,
        device="cuda",
        iterations=100,
        M=4,
        batch_size=8,
        max_len=80,
        beam_width=5,
        mu=4,
        clip_eps=0.1,
        kl_beta=0.1,
        temperature=1.0,
        log_dir=os.path.join(BASE_DIR, "rl/feedback/logs"),
        save_dir=os.path.join(BASE_DIR, "rl/feedback/best_models"),
        replay_buffer_size=300,
        replay_batch_size=4,
        replay_start=30,
):
    agent.to(device)
    old_agent = copy.deepcopy(agent).to(device)
    ref_agent = copy.deepcopy(agent).to(device)

    old_agent.eval()
    ref_agent.eval()

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"gspo_beam_training_log_{timestamp}.csv")

    if not os.path.exists(log_path):
        pd.DataFrame(columns=[
            "iteration", "step", "mean_reward", "mean_vina", "mean_qed",
            "mean_sa", "mean_logp", "top1_reward", "top5_mean_reward",
            "policy_loss", "kl_loss", "ratio_mean", "replay_size"
        ]).to_csv(log_path, index=False)

    best_total_loss = float("inf")
    best_reward = float("-inf")
    best_total_loss_state = None
    best_reward_state = None

    replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

    for it in range(1, iterations + 1):
        print(f"\n{'=' * 70}")
        print(f" Iteration {it}/{iterations} (Beam Search)")
        print(f"{'=' * 70}")

        ref_agent.load_state_dict(agent.state_dict())
        for p in ref_agent.parameters():
            p.requires_grad_(False)

        for step in range(1, M + 1):
            print(f" [Iter {it}, Step {step}/{M}]")

            old_agent.load_state_dict(agent.state_dict())
            for p in old_agent.parameters():
                p.requires_grad_(False)

            batch_token_ids, batch_smiles = sample_selfies_batch_beam(
                vocab=vocab,
                model=old_agent,
                batch_size=batch_size,
                max_len=max_len,
                beam_width=beam_width,
                temperature=temperature,
                beam_temperature=1.0,
                noise_std=0.5,
                device=device,
            )

            if len(batch_token_ids) < 2:
                print(f"   Skip: no valid samples")
                continue

            vina_results = batch_scores_from_vina(
                batch_smiles,
                receptor_file=os.path.join(BASE_DIR, "feedback/8sc7.pdbqt"),
                pdbqt_dir=os.path.join(BASE_DIR, "feedback/temp/pdbqt"),
                output_dir=os.path.join(BASE_DIR, "feedback/vina_results/")
            )

            reward_items = make_composite_reward(
                batch_smiles,
                vina_results,
                weights={'vina': 3.0, 'qed': 1.0, 'sa': 0.5, 'logp': 0},
                invalid_penalty=0
            )

            rewards = [item['reward'] for item in reward_items]

            valid = [
                (s, r, smi) for s, r, smi in zip(batch_token_ids, rewards, batch_smiles)
                if r != 0
            ]
            if len(valid) < 2:
                print(f"   Skip: invalid rewards")
                continue

            batch_token_ids, rewards, batch_smiles_valid = zip(*valid)

            valid_indices = [i for i, r in enumerate(
                [item['reward'] for item in reward_items]) if r != 0]
            valid_items = [reward_items[i] for i in valid_indices]

            mean_reward = float(np.mean(rewards))
            top1_reward = max(rewards)
            top5_mean_reward = float(np.mean(sorted(rewards, reverse=True)[:min(5, len(rewards))]))
            mean_vina = float(np.mean([item['vina'] for item in valid_items]))
            mean_qed = float(np.mean([item['qed'] for item in valid_items]))
            mean_sa = float(np.mean([item['sa'] for item in valid_items]))
            mean_logp = float(np.mean([item['logp'] for item in valid_items]))

            for token_ids, smiles, reward in zip(batch_token_ids, batch_smiles_valid, rewards):
                replay_buffer.add(token_ids, smiles, reward)

            if len(replay_buffer) >= replay_start:
                replay_ids, _, replay_rewards = replay_buffer.sample(replay_batch_size)
                mixed_token_ids = list(batch_token_ids) + replay_ids
                mixed_rewards = list(rewards) + replay_rewards
                print(f"   Replay: +{len(replay_ids)} samples | buffer size={len(replay_buffer)}")
            else:
                mixed_token_ids = list(batch_token_ids)
                mixed_rewards = list(rewards)
                print(f"   Replay: warming up ({len(replay_buffer)}/{replay_start})")

            for update_idx in range(mu):
                total_loss, pol_loss, kl_loss, r_mean = compute_gspo_loss_batch(
                    agent,
                    old_agent,
                    ref_agent,
                    mixed_token_ids,
                    mixed_rewards,
                    clip_eps,
                    kl_beta,
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

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
                "policy_loss": pol_loss.item(),
                "kl_loss": kl_loss.item(),
                "ratio_mean": r_mean.item(),
                "replay_size": len(replay_buffer)
            }]).to_csv(log_path, mode="a", index=False, header=False)

            print(
                f"   meanR={mean_reward:.3f} | top1={top1_reward:.3f} | "
                f"mean_vina={mean_vina:.3f} | mean_qed={mean_qed:.3f} | "
                f"mean_sa={mean_sa:.3f} | mean_logp={mean_logp:.3f} | "
                f"top5={top5_mean_reward:.3f}\n"
                f"   policy={pol_loss.item():.4f} | kl={kl_loss.item():.4f} | "
                f"ratio={r_mean.item():.3f} | buffer={len(replay_buffer)}"
            )

            if kl_loss.item() > 0.8:
                print(f"   Early stopping: KL loss too high ({kl_loss.item():.4f} > 0.8)")
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


def freeze_agent_partial(agent):
    for param in agent.parameters():
        param.requires_grad = False

    for param in agent.layers[-1].parameters():
        param.requires_grad = True

    for param in agent.fc_out.parameters():
        param.requires_grad = True


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
        weight_decay=0.001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1000,
        eta_min=1e-6
    )

    train_gspo_beam(
        agent,
        vocab,
        optimizer,
        scheduler,
        device=device,
        iterations=1,
        M=500,
        batch_size=8,
        beam_width=8,
        mu=2,
        clip_eps=0.2,
        kl_beta=0.1,
        temperature=1,
        log_dir=os.path.join(BASE_DIR, "rl/feedback/logs"),
        save_dir=os.path.join(BASE_DIR, "rl/feedback/best_models"),
        replay_start=50,
    )