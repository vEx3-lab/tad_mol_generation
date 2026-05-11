# Stable route:
#   on-policy GSPO
#   + elite replay memory as auxiliary likelihood loss only
#   + Vina cache
#   + scaffold diversity control
#
# Replay samples are NOT mixed into the GSPO/PPO ratio loss. They only enter
# the supervised auxiliary loss, so they still backpropagate, but do not break
# the on-policy assumption of GSPO.

import copy
import math
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
import selfies as sf
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from feedback.utils import canonicalize_smiles
from feedback.vina_scores import batch_scores_from_vina
from utils import make_composite_reward


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def special_id(vocab, token, fallback=0):
    return vocab.token2id.get(token, fallback)


def smiles_to_scaffold(smiles):
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None

    mol = Chem.MolFromSmiles(canonical)
    if mol is None:
        return None

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""

    if not scaffold:
        return "acyclic"
    return scaffold


def tokenize_selfies_result(result, vocab, max_len):
    if result.get("selfies") is None or result.get("smiles") is None:
        return None

    try:
        core_tokens = list(sf.split_selfies(result["selfies"]))
    except Exception:
        return None

    if not core_tokens:
        return None

    core_tokens = core_tokens[: max(1, max_len - 2)]
    tokens = ["<SOS>"] + core_tokens + ["<EOS>"]
    return [vocab[token] for token in tokens]


@torch.no_grad()
def sample_rollout_batch(
    model,
    vocab,
    batch_size=8,
    max_len=80,
    temperature=1.0,
    top_k=20,
    device="cuda",
):
    from sample.sample import generate_selfies

    model.eval()
    token_ids = []
    smiles = []

    attempts = 0
    max_attempts = batch_size * 4
    while len(token_ids) < batch_size and attempts < max_attempts:
        attempts += 1
        result = generate_selfies(
            model=model,
            vocab=vocab,
            device=device,
            max_len=max(1, max_len - 2),
            temperature=temperature,
            top_k=top_k,
        )
        ids = tokenize_selfies_result(result, vocab, max_len=max_len)
        smi = result.get("smiles")
        if ids is None or smi is None:
            continue
        if len(ids) < 3:
            continue
        token_ids.append(ids)
        smiles.append(smi)

    return token_ids, smiles


def sequence_logprob(agent, seq, pad_token_id=0, eps=1e-8):
    device = next(agent.parameters()).device
    seq = torch.tensor(seq, dtype=torch.long, device=device)
    if seq.numel() < 2:
        return None

    input_ids = seq[:-1].unsqueeze(0)
    target_ids = seq[1:].unsqueeze(0)
    mask = (target_ids != pad_token_id).float()
    valid_len = mask.sum() + eps

    logits = agent(input_ids)
    logp = F.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum() / valid_len


def compute_gspo_on_policy_loss(
    agent,
    old_agent,
    ref_agent,
    batch_sequences,
    batch_rewards,
    clip_eps=0.1,
    kl_beta=0.05,
    pad_token_id=0,
    eps=1e-8,
):
    device = next(agent.parameters()).device
    rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
    if rewards.numel() < 2:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        one = torch.tensor(1.0, device=device)
        return zero, zero.detach(), zero.detach(), one

    adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)
    adv = adv.detach()

    policy_losses = []
    kl_losses = []
    ratio_values = []

    for i, seq in enumerate(batch_sequences):
        seq = torch.tensor(seq, dtype=torch.long, device=device)
        if seq.numel() < 2:
            continue

        input_ids = seq[:-1].unsqueeze(0)
        target_ids = seq[1:].unsqueeze(0)
        mask = (target_ids != pad_token_id).float()
        valid_len = mask.sum() + eps

        logits_new = agent(input_ids)
        with torch.no_grad():
            logits_old = old_agent(input_ids)
            logits_ref = ref_agent(input_ids)

        logp_new = F.log_softmax(logits_new, dim=-1)
        logp_old = F.log_softmax(logits_old, dim=-1)
        logp_ref = F.log_softmax(logits_ref, dim=-1)

        token_new = logp_new.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_old = logp_old.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_ref = logp_ref.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        log_ratio_policy = ((token_new - token_old) * mask).sum() / valid_len
        ratio = torch.exp(log_ratio_policy)
        ratio_values.append(ratio.detach())

        surr1 = ratio * adv[i]
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv[i]
        policy_losses.append(-torch.min(surr1, surr2))

        log_ratio_kl = ((token_ref - token_new) * mask).sum() / valid_len
        kl = torch.exp(log_ratio_kl) - log_ratio_kl - 1.0
        kl_losses.append(kl)

    if not policy_losses:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        one = torch.tensor(1.0, device=device)
        return zero, zero.detach(), zero.detach(), one

    policy_loss = torch.stack(policy_losses).mean()
    kl_loss = torch.stack(kl_losses).mean()
    ratio_mean = torch.stack(ratio_values).mean()
    total_loss = policy_loss + kl_beta * kl_loss
    return total_loss, policy_loss, kl_loss, ratio_mean


def compute_aux_likelihood_loss(
    agent,
    replay_samples,
    pad_token_id=0,
    reward_temperature=0.08,
    eps=1e-8,
):
    device = next(agent.parameters()).device
    if not replay_samples:
        return torch.tensor(0.0, device=device), 0

    rewards = torch.tensor(
        [sample["reward"] for sample in replay_samples],
        dtype=torch.float32,
        device=device,
    )
    scaled = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)
    weights = F.softmax(scaled / max(reward_temperature, eps), dim=0).detach()
    weights = weights * len(replay_samples)

    losses = []
    used = 0
    for weight, sample in zip(weights, replay_samples):
        seq = torch.tensor(sample["token_ids"], dtype=torch.long, device=device)
        if seq.numel() < 2:
            continue

        input_ids = seq[:-1].unsqueeze(0)
        target_ids = seq[1:].unsqueeze(0)
        mask = (target_ids != pad_token_id).float()
        valid_len = mask.sum() + eps

        logits = agent(input_ids)
        logp = F.log_softmax(logits, dim=-1)
        token_logp = logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        seq_nll = -(token_logp * mask).sum() / valid_len
        losses.append(weight * seq_nll)
        used += 1

    if not losses:
        return torch.tensor(0.0, device=device), 0

    return torch.stack(losses).mean(), used


class EliteReplayMemory:
    def __init__(
        self,
        max_size=300,
        max_per_scaffold=12,
        sample_top_k=120,
        min_reward=None,
    ):
        self.max_size = int(max_size)
        self.max_per_scaffold = int(max_per_scaffold)
        self.sample_top_k = int(sample_top_k)
        self.min_reward = min_reward
        self.items_by_smiles = {}

    def add(self, token_ids, smiles, reward, step_id, reward_info=None):
        canonical = canonicalize_smiles(smiles)
        if canonical is None:
            return False
        if self.min_reward is not None and reward < self.min_reward:
            return False

        scaffold = smiles_to_scaffold(canonical)
        if scaffold is None:
            return False

        old = self.items_by_smiles.get(canonical)
        if old is not None and reward <= old["reward"]:
            return False

        self.items_by_smiles[canonical] = {
            "token_ids": list(token_ids),
            "smiles": smiles,
            "canonical_smiles": canonical,
            "scaffold": scaffold,
            "reward": float(reward),
            "step_id": int(step_id),
            "reward_info": reward_info or {},
        }

        self._trim_scaffold(scaffold)
        self._trim_total()
        return True

    def _trim_scaffold(self, scaffold):
        if self.max_per_scaffold <= 0:
            return

        members = [
            item for item in self.items_by_smiles.values()
            if item["scaffold"] == scaffold
        ]
        if len(members) <= self.max_per_scaffold:
            return

        members = sorted(members, key=lambda x: x["reward"], reverse=True)
        remove = members[self.max_per_scaffold:]
        for item in remove:
            self.items_by_smiles.pop(item["canonical_smiles"], None)

    def _trim_total(self):
        if len(self.items_by_smiles) <= self.max_size:
            return

        members = sorted(
            self.items_by_smiles.values(),
            key=lambda x: x["reward"],
            reverse=True,
        )
        keep = members[: self.max_size]
        self.items_by_smiles = {item["canonical_smiles"]: item for item in keep}

    def sample(self, n):
        items = sorted(
            self.items_by_smiles.values(),
            key=lambda x: x["reward"],
            reverse=True,
        )[: self.sample_top_k]
        if not items:
            return []

        selected = []
        used_scaffolds = set()

        # First pass: prefer different scaffolds.
        for item in items:
            if len(selected) >= n:
                break
            if item["scaffold"] in used_scaffolds:
                continue
            selected.append(item)
            used_scaffolds.add(item["scaffold"])

        # Second pass: fill remaining slots by reward.
        if len(selected) < n:
            selected_ids = {item["canonical_smiles"] for item in selected}
            for item in items:
                if len(selected) >= n:
                    break
                if item["canonical_smiles"] in selected_ids:
                    continue
                selected.append(item)

        return selected

    def known_scaffolds(self):
        return {item["scaffold"] for item in self.items_by_smiles.values()}

    def scaffold_count(self):
        return len(self.known_scaffolds())

    def __len__(self):
        return len(self.items_by_smiles)


def apply_scaffold_diversity_bonus(
    reward_items,
    memory,
    new_scaffold_bonus=0.03,
    duplicate_scaffold_penalty=0.01,
):
    known = memory.known_scaffolds()
    seen_in_batch = set()
    adjusted = []

    for item in reward_items:
        item = dict(item)
        scaffold = smiles_to_scaffold(item.get("smiles"))
        item["scaffold"] = scaffold
        item["scaffold_bonus"] = 0.0

        if scaffold is not None:
            if scaffold not in known and scaffold not in seen_in_batch:
                item["scaffold_bonus"] += new_scaffold_bonus
            elif scaffold in seen_in_batch:
                item["scaffold_bonus"] -= duplicate_scaffold_penalty
            seen_in_batch.add(scaffold)

        item["base_reward"] = float(item.get("reward", 0.0))
        item["reward"] = item["base_reward"] + item["scaffold_bonus"]
        item["total_reward"] = item["reward"]
        adjusted.append(item)

    return adjusted


def summarize_vina_status(vina_results):
    counts = {}
    for value in vina_results.values():
        status = value.get("status", "unknown") if isinstance(value, dict) else "unknown"
        counts[status] = counts.get(status, 0) + 1
    cached = counts.get("cached", 0)
    success = counts.get("success", 0)
    failed = sum(v for k, v in counts.items() if k not in {"cached", "success"})
    return cached, success, failed


def freeze_agent_partial(agent):
    for param in agent.parameters():
        param.requires_grad = False

    for param in agent.layers[-1].parameters():
        param.requires_grad = True

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


def train_gspo_elite_aux(
    agent,
    vocab,
    optimizer,
    scheduler,
    device="cuda",
    iterations=1,
    steps_per_iter=700,
    batch_size=8,
    max_len=80,
    updates_per_step=1,
    clip_eps=0.1,
    kl_beta=0.05,
    target_kl=0.12,
    aux_coef=0.02,
    aux_start=32,
    aux_batch_size=8,
    temperature=1.0,
    top_k=20,
    reward_weights=None,
    replay_size=300,
    max_per_scaffold=12,
    scaffold_bonus=0.03,
    duplicate_scaffold_penalty=0.01,
    pad_token_id=0,
    log_dir=os.path.join(BASE_DIR, "rl/feedback/logs"),
    save_dir=os.path.join(BASE_DIR, "rl/feedback/best_models"),
    vina_cache_file=os.path.join(BASE_DIR, "feedback/vina_results/vina_cache.csv"),
):
    agent.to(device)
    old_agent = copy.deepcopy(agent).to(device)
    ref_agent = copy.deepcopy(agent).to(device)
    old_agent.eval()
    ref_agent.eval()

    for p in old_agent.parameters():
        p.requires_grad_(False)
    for p in ref_agent.parameters():
        p.requires_grad_(False)

    memory = EliteReplayMemory(
        max_size=replay_size,
        max_per_scaffold=max_per_scaffold,
        sample_top_k=max(4 * aux_batch_size, 64),
    )

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"gspo_elite_aux_log_{timestamp}.csv")

    pd.DataFrame(columns=[
        "iteration", "step", "global_step",
        "mean_reward", "mean_base_reward", "top1_reward", "top5_mean_reward",
        "mean_vina", "mean_qed", "mean_sa", "mean_logp",
        "mean_scaffold_bonus", "unique_scaffolds_batch",
        "policy_loss", "kl_loss", "aux_loss", "total_loss", "ratio_mean",
        "lr", "kl_beta", "aux_coef",
        "memory_size", "memory_scaffolds", "aux_used",
        "vina_cached", "vina_success", "vina_failed",
    ]).to_csv(log_path, index=False)

    if reward_weights is None:
        reward_weights = {"vina": 1.0, "qed": 0.2, "sa": 0.2, "logp": 0.0}

    best_reward = float("-inf")
    best_loss = float("inf")
    best_reward_state = None
    best_loss_state = None
    global_step = 0

    for iteration in range(1, iterations + 1):
        print("\n" + "=" * 70)
        print(f" Iteration {iteration}/{iterations} (on-policy GSPO + elite aux)")
        print("=" * 70)

        ref_agent.load_state_dict(agent.state_dict())
        ref_agent.eval()

        for step in range(1, steps_per_iter + 1):
            global_step += 1
            print(f" [Iter {iteration}, Step {step}/{steps_per_iter}]")

            old_agent.load_state_dict(agent.state_dict())
            old_agent.eval()

            batch_token_ids, batch_smiles = sample_rollout_batch(
                model=old_agent,
                vocab=vocab,
                batch_size=batch_size,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k,
                device=device,
            )

            if len(batch_token_ids) < 2:
                print("   Skip: no valid rollout samples")
                continue

            vina_results = batch_scores_from_vina(
                batch_smiles,
                receptor_file=os.path.join(BASE_DIR, "feedback/8sc7.pdbqt"),
                pdbqt_dir=os.path.join(BASE_DIR, "feedback/temp/pdbqt"),
                output_dir=os.path.join(BASE_DIR, "feedback/vina_results"),
                cache_file=vina_cache_file,
                use_cache=True,
                keep_ligand_pdbqt=False,
                save_poses=False,
            )
            vina_cached, vina_success, vina_failed = summarize_vina_status(vina_results)

            reward_items = make_composite_reward(
                batch_smiles,
                vina_results,
                weights=reward_weights,
                invalid_penalty=0,
            )
            reward_items = apply_scaffold_diversity_bonus(
                reward_items,
                memory=memory,
                new_scaffold_bonus=scaffold_bonus,
                duplicate_scaffold_penalty=duplicate_scaffold_penalty,
            )

            valid = []
            for token_ids, smi, item in zip(batch_token_ids, batch_smiles, reward_items):
                if item["reward"] <= 0:
                    continue
                if item.get("scaffold") is None:
                    continue
                valid.append((token_ids, smi, item))

            if len(valid) < 2:
                print("   Skip: valid reward samples < 2")
                continue

            batch_token_ids = [x[0] for x in valid]
            batch_smiles_valid = [x[1] for x in valid]
            reward_items = [x[2] for x in valid]
            rewards = [float(item["reward"]) for item in reward_items]
            base_rewards = [float(item["base_reward"]) for item in reward_items]

            for token_ids, smi, item in zip(batch_token_ids, batch_smiles_valid, reward_items):
                memory.add(
                    token_ids=token_ids,
                    smiles=smi,
                    reward=item["reward"],
                    step_id=global_step,
                    reward_info=item,
                )

            mean_reward = float(np.mean(rewards))
            mean_base_reward = float(np.mean(base_rewards))
            top1_reward = float(np.max(rewards))
            top5_mean_reward = float(np.mean(sorted(rewards, reverse=True)[: min(5, len(rewards))]))
            mean_vina = float(np.mean([item["vina"] for item in reward_items]))
            mean_qed = float(np.mean([item["qed"] for item in reward_items]))
            mean_sa = float(np.mean([item["sa"] for item in reward_items]))
            mean_logp = float(np.mean([item["logp"] for item in reward_items]))
            mean_scaffold_bonus = float(np.mean([item["scaffold_bonus"] for item in reward_items]))
            unique_scaffolds_batch = len({item["scaffold"] for item in reward_items})

            last_total_loss = None
            last_policy_loss = None
            last_kl_loss = None
            last_aux_loss = None
            last_ratio = None
            last_aux_used = 0

            for update_idx in range(updates_per_step):
                gspo_loss, policy_loss, kl_loss, ratio_mean = compute_gspo_on_policy_loss(
                    agent=agent,
                    old_agent=old_agent,
                    ref_agent=ref_agent,
                    batch_sequences=batch_token_ids,
                    batch_rewards=rewards,
                    clip_eps=clip_eps,
                    kl_beta=kl_beta,
                    pad_token_id=pad_token_id,
                )

                aux_samples = []
                if len(memory) >= aux_start:
                    aux_samples = memory.sample(aux_batch_size)
                aux_loss, aux_used = compute_aux_likelihood_loss(
                    agent=agent,
                    replay_samples=aux_samples,
                    pad_token_id=pad_token_id,
                )

                total_loss = gspo_loss + aux_coef * aux_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                last_total_loss = total_loss.detach()
                last_policy_loss = policy_loss.detach()
                last_kl_loss = kl_loss.detach()
                last_aux_loss = aux_loss.detach()
                last_ratio = ratio_mean.detach()
                last_aux_used = aux_used

            if last_kl_loss is not None:
                if last_kl_loss.item() > target_kl * 1.5:
                    kl_beta = min(kl_beta * 1.5, 1.0)
                elif last_kl_loss.item() < target_kl / 3.0:
                    kl_beta = max(kl_beta / 1.2, 1e-4)

            current_lr = optimizer.param_groups[0]["lr"]
            pd.DataFrame([{
                "iteration": iteration,
                "step": step,
                "global_step": global_step,
                "mean_reward": mean_reward,
                "mean_base_reward": mean_base_reward,
                "top1_reward": top1_reward,
                "top5_mean_reward": top5_mean_reward,
                "mean_vina": mean_vina,
                "mean_qed": mean_qed,
                "mean_sa": mean_sa,
                "mean_logp": mean_logp,
                "mean_scaffold_bonus": mean_scaffold_bonus,
                "unique_scaffolds_batch": unique_scaffolds_batch,
                "policy_loss": float(last_policy_loss.item()),
                "kl_loss": float(last_kl_loss.item()),
                "aux_loss": float(last_aux_loss.item()),
                "total_loss": float(last_total_loss.item()),
                "ratio_mean": float(last_ratio.item()),
                "lr": current_lr,
                "kl_beta": kl_beta,
                "aux_coef": aux_coef,
                "memory_size": len(memory),
                "memory_scaffolds": memory.scaffold_count(),
                "aux_used": last_aux_used,
                "vina_cached": vina_cached,
                "vina_success": vina_success,
                "vina_failed": vina_failed,
            }]).to_csv(log_path, mode="a", index=False, header=False)

            print(
                f"   meanR={mean_reward:.3f} | baseR={mean_base_reward:.3f} | "
                f"top1={top1_reward:.3f} | top5={top5_mean_reward:.3f}\n"
                f"   vina={mean_vina:.3f} | qed={mean_qed:.3f} | "
                f"sa={mean_sa:.3f} | logp={mean_logp:.3f} | "
                f"scaffold+={mean_scaffold_bonus:.3f} | scaffolds={unique_scaffolds_batch}\n"
                f"   policy={last_policy_loss.item():.4f} | kl={last_kl_loss.item():.4f} | "
                f"ratio={last_ratio.item():.3f} | aux={last_aux_loss.item():.4f} "
                f"(used={last_aux_used}) | total={last_total_loss.item():.4f}\n"
                f"   lr={current_lr:.2e} | kl_beta={kl_beta:.4f} | "
                f"memory={len(memory)}/{memory.scaffold_count()}scaf | "
                f"vina cache/success/fail={vina_cached}/{vina_success}/{vina_failed}"
            )

            if last_total_loss.item() < best_loss:
                best_loss = last_total_loss.item()
                best_loss_state = copy.deepcopy(agent.state_dict())
                print(f"   New best total loss: {best_loss:.4f}")

            if mean_reward > best_reward:
                best_reward = mean_reward
                best_reward_state = copy.deepcopy(agent.state_dict())
                print(f"   New best reward: {best_reward:.3f}")

            if last_kl_loss.item() > 1.0:
                print(f"   Early stop: KL too high ({last_kl_loss.item():.4f})")
                break

    if best_loss_state is not None:
        path = os.path.join(save_dir, f"best_total_loss_elite_aux_{timestamp}.pt")
        torch.save(best_loss_state, path)
        print(f"Saved best loss model: {path}")

    if best_reward_state is not None:
        path = os.path.join(save_dir, f"best_reward_elite_aux_{timestamp}.pt")
        torch.save(best_reward_state, path)
        print(f"Saved best reward model: {path}")

    print(f"Training log saved to: {log_path}")


def load_vocab_for_rl(data_file):
    from data.data_utils import load_selfies_vocab, selfies_vocab

    vocab_path = os.path.join(BASE_DIR, "model/decoder_only_tfm/vocab.json")
    if os.path.exists(vocab_path):
        return load_selfies_vocab(vocab_path)

    data = pd.read_csv(data_file)["selfies"].dropna().tolist()
    return selfies_vocab(data)


if __name__ == "__main__":
    from config.load_config import load_config
    from model.decoder_only_tfm import decoder_only_tfm

    set_seed(42)

    config = load_config(os.path.join(BASE_DIR, "config/decoder_only_tfm_config.yaml"))
    device = config["device"]
    model_cfg = config["model"]
    data_file = os.path.join(BASE_DIR, "data/htvs_molecules_with_selfies.csv")
    vocab = load_vocab_for_rl(data_file)

    agent = decoder_only_tfm(
        vocab_size=len(vocab),
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        max_len=model_cfg["max_len"],
        dropout=model_cfg["dropout"],
        pad_token_id=special_id(vocab, "<PAD>", 0),
    ).to(device)

    start_model = os.path.join(BASE_DIR, "train/model/decoder_only_tfm/best_model_fold1.pt")
    agent.load_state_dict(
        torch.load(start_model, map_location=device, weights_only=True),
        strict=False,
    )

    freeze_agent_partial(agent)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, agent.parameters()),
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=0.001,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1000,
        eta_min=1e-7,
    )

    train_gspo_elite_aux(
        agent=agent,
        vocab=vocab,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        iterations=1,
        steps_per_iter=700,
        batch_size=8,
        max_len=model_cfg["max_len"],
        updates_per_step=1,
        clip_eps=0.1,
        kl_beta=0.05,
        target_kl=0.12,
        aux_coef=0.02,
        aux_start=32,
        aux_batch_size=8,
        temperature=1.0,
        top_k=20,
        reward_weights={"vina": 1.0, "qed": 0.2, "sa": 0.2, "logp": 0.0},
        replay_size=300,
        max_per_scaffold=12,
        scaffold_bonus=0.03,
        duplicate_scaffold_penalty=0.01,
        pad_token_id=special_id(vocab, "<PAD>", 0),
        log_dir=os.path.join(BASE_DIR, "rl/feedback/logs"),
        save_dir=os.path.join(BASE_DIR, "rl/feedback/best_models"),
        vina_cache_file=os.path.join(BASE_DIR, "feedback/vina_results/vina_cache.csv"),
    )
