import copy
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
from rl.multi_obj_gspo_4_21 import freeze_agent_partial, set_seed
from rl.multi_obj_gspo_4_21_diagnostics import (
    compute_gspo_loss_batch,
    compute_old_logprob_batch,
)
from utils import make_composite_reward, sample_selfies_batch_from_generate_selfies


OBJECTIVE_KEYS = ("vina_reward", "qed", "sa_reward", "logp_reward")
OBJECTIVE_NAMES = ("vina", "qed", "sa", "logp")


def dominates(a, b, eps=1e-9):
    """Return True when objective vector a Pareto-dominates b. Higher is better."""
    return np.all(a >= b - eps) and np.any(a > b + eps)


def fast_non_dominated_sort(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    n = len(vectors)
    dominates_list = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=np.int64)
    ranks = np.full(n, -1, dtype=np.int64)
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(vectors[i], vectors[j]):
                dominates_list[i].append(j)
            elif dominates(vectors[j], vectors[i]):
                dominated_count[i] += 1

        if dominated_count[i] == 0:
            ranks[i] = 0
            fronts[0].append(i)

    current_rank = 0
    while fronts[current_rank]:
        next_front = []
        for i in fronts[current_rank]:
            for j in dominates_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    ranks[j] = current_rank + 1
                    next_front.append(j)
        current_rank += 1
        fronts.append(next_front)

    return ranks, fronts[:-1]


def crowding_distance(vectors, fronts):
    vectors = np.asarray(vectors, dtype=np.float64)
    n, dim = vectors.shape
    distances = np.zeros(n, dtype=np.float64)

    for front in fronts:
        if len(front) == 0:
            continue
        if len(front) <= 2:
            distances[front] = np.inf
            continue

        front_vectors = vectors[front]
        for m in range(dim):
            order = np.argsort(front_vectors[:, m])
            sorted_front = [front[idx] for idx in order]
            distances[sorted_front[0]] = np.inf
            distances[sorted_front[-1]] = np.inf

            min_v = front_vectors[order[0], m]
            max_v = front_vectors[order[-1], m]
            denom = max(max_v - min_v, 1e-12)
            for k in range(1, len(sorted_front) - 1):
                prev_v = front_vectors[order[k - 1], m]
                next_v = front_vectors[order[k + 1], m]
                distances[sorted_front[k]] += (next_v - prev_v) / denom

    return distances


def minmax_normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    mins = vectors.min(axis=0)
    maxs = vectors.max(axis=0)
    denom = np.maximum(maxs - mins, 1e-12)
    return (vectors - mins) / denom


def normalize_crowding(distances):
    distances = np.asarray(distances, dtype=np.float64)
    finite = np.isfinite(distances)
    if not finite.any():
        return np.ones_like(distances)

    normalized = distances.copy()
    max_finite = max(float(distances[finite].max()), 1e-12)
    normalized[~finite] = max_finite
    return normalized / max_finite


def pareto_training_scores(
    vectors,
    ranks,
    distances,
    preference_weights=None,
    random_preference=True,
    crowding_coef=0.05,
    scalar_hint_coef=0.10,
):
    """
    Convert a Pareto relation into a scalar signal for policy gradient.

    rank score keeps non-dominated molecules ahead, crowding rewards frontier
    diversity, and a small scalar hint avoids zero signal when all samples are
    non-dominated in a small batch.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    dim = vectors.shape[1]

    if preference_weights is None:
        if random_preference:
            weights = np.random.dirichlet(np.ones(dim))
        else:
            weights = np.ones(dim, dtype=np.float64) / dim
    else:
        weights = np.asarray(preference_weights, dtype=np.float64)
        weights = weights / max(weights.sum(), 1e-12)

    normalized_vectors = minmax_normalize(vectors)
    rank_score = 1.0 / (1.0 + ranks.astype(np.float64))
    crowding_score = normalize_crowding(distances)
    scalar_hint = normalized_vectors @ weights

    scores = rank_score + crowding_coef * crowding_score + scalar_hint_coef * scalar_hint
    return scores.astype(np.float64), weights


def objective_vector_from_item(item, objective_keys=OBJECTIVE_KEYS):
    values = []
    for key in objective_keys:
        value = item.get(key)
        if value is None or pd.isna(value):
            return None
        values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def mean_or_none(values):
    values = [x for x in values if x is not None and not pd.isna(x)]
    if not values:
        return None
    return float(np.mean(values))


def unique_smiles_rate(smiles_list):
    canonical = [canonicalize_smiles(smi) for smi in smiles_list]
    canonical = [smi for smi in canonical if smi is not None]
    if not canonical:
        return 0.0
    return float(len(set(canonical)) / len(canonical))


class ParetoArchive:
    def __init__(self, objective_keys=OBJECTIVE_KEYS, max_size=500):
        self.objective_keys = tuple(objective_keys)
        self.max_size = int(max_size)
        self.records = []

    def add_many(self, records):
        self.records.extend(records)
        self._filter()

    def _filter(self):
        if not self.records:
            return

        dedup = {}
        for record in self.records:
            key = record.get("canonical_smiles") or record.get("smiles")
            if key is None:
                continue
            old = dedup.get(key)
            if old is None or record["weighted_reward"] > old["weighted_reward"]:
                dedup[key] = record

        records = list(dedup.values())
        vectors = np.asarray([r["objectives"] for r in records], dtype=np.float64)
        ranks, fronts = fast_non_dominated_sort(vectors)
        keep = [idx for idx, rank in enumerate(ranks) if rank == 0]
        records = [records[idx] for idx in keep]

        if len(records) > self.max_size:
            vectors = np.asarray([r["objectives"] for r in records], dtype=np.float64)
            ranks, fronts = fast_non_dominated_sort(vectors)
            distances = crowding_distance(vectors, fronts)
            crowding = normalize_crowding(distances)
            scalar = minmax_normalize(vectors).mean(axis=1)
            order = np.argsort(-(crowding + 0.1 * scalar))
            records = [records[idx] for idx in order[: self.max_size]]

        self.records = records

    def sample(self, n):
        if not self.records or n <= 0:
            return []

        n = min(int(n), len(self.records))
        rewards = np.asarray([r["weighted_reward"] for r in self.records], dtype=np.float64)
        probs = np.exp(rewards - rewards.max())
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.records), size=n, replace=False, p=probs)
        return [self.records[idx] for idx in indices]

    def save_csv(self, path):
        rows = []
        for record in self.records:
            row = {
                "smiles": record.get("smiles"),
                "canonical_smiles": record.get("canonical_smiles"),
                "weighted_reward": record.get("weighted_reward"),
                "pareto_score": record.get("pareto_score"),
                "pareto_rank": record.get("pareto_rank"),
            }
            for name, value in zip(OBJECTIVE_NAMES, record["objectives"]):
                row[name] = value
            rows.append(row)

        pd.DataFrame(rows).to_csv(path, index=False)

    def __len__(self):
        return len(self.records)


def train_pareto_gspo(
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
    clip_eps=0.2,
    kl_beta=0.15,
    temperature=1.0,
    top_k=None,
    log_dir=os.path.join(BASE_DIR, "rl/feedback/logs"),
    save_dir=os.path.join(BASE_DIR, "rl/feedback/best_models"),
    archive_max_size=500,
    archive_replay_batch_size=0,
    ratio_min=0.5,
    ratio_max=1.5,
    pad_token_id=0,
    objective_keys=OBJECTIVE_KEYS,
    objective_names=OBJECTIVE_NAMES,
    logging_weights=(1.0, 0.2, 0.2, 0.0),
    preference_weights=None,
    random_preference=True,
    crowding_coef=0.05,
    scalar_hint_coef=0.10,
):
    agent.to(device)
    old_agent = copy.deepcopy(agent).to(device)
    ref_agent = copy.deepcopy(agent).to(device)
    old_agent.eval()
    ref_agent.eval()

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"pareto_gspo_training_log_{timestamp}.csv")
    archive_path = os.path.join(log_dir, f"pareto_archive_{timestamp}.csv")

    objective_keys = tuple(objective_keys)
    objective_names = tuple(objective_names)
    if len(objective_keys) != len(objective_names):
        raise ValueError("objective_keys and objective_names must have the same length")

    logging_weights = np.asarray(logging_weights, dtype=np.float64)
    if len(logging_weights) != len(objective_keys):
        raise ValueError("logging_weights must match the number of objectives")

    archive = ParetoArchive(objective_keys=objective_keys, max_size=archive_max_size)
    pref_columns = [f"pref_{name}" for name in objective_names]

    columns = [
        "iteration", "step", "global_step",
        "mean_pareto_score", "top1_pareto_score", "top5_pareto_score",
        "mean_weighted_reward", "top1_weighted_reward", "top5_weighted_reward",
        "front0_count", "front0_rate", "mean_pareto_rank",
        "archive_size", "policy_loss", "kl_loss", "total_loss", "ratio_mean",
        "lr", "kl_beta", "kept_samples", "updates_applied", "grad_norm",
        "clip_eps", "mu", "ratio_min", "ratio_max",
        "valid_count", "unique_smiles_rate",
    ]
    columns += pref_columns
    columns += [f"mean_{name}" for name in objective_names]
    columns += [f"top5_{name}" for name in objective_names]
    pd.DataFrame(columns=columns).to_csv(log_path, index=False)

    best_pareto_score = float("-inf")
    best_weighted_reward = float("-inf")
    best_pareto_state = None
    best_weighted_state = None

    for it in range(1, iterations + 1):
        print(f"\n{'=' * 70}")
        print(f" Iteration {it}/{iterations} (Pareto GSPO)")
        print(f"{'=' * 70}")

        ref_agent.load_state_dict(agent.state_dict())
        for p in ref_agent.parameters():
            p.requires_grad_(False)

        for step in range(1, M + 1):
            global_step = (it - 1) * M + step
            print(f" [Iter {it}, Step {step}/{M}]")

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

            reward_items = make_composite_reward(
                batch_smiles,
                vina_results,
                weights={"vina": 1.0, "qed": 0.2, "sa": 0.2, "logp": 0.0},
                invalid_penalty=0,
            )

            valid = []
            for token_ids, smiles, old_lp, item in zip(
                batch_token_ids, batch_smiles, batch_old_logprobs, reward_items
            ):
                vector = objective_vector_from_item(item, objective_keys)
                if vector is None or old_lp is None:
                    continue
                if np.allclose(vector, 0.0):
                    continue
                valid.append((token_ids, smiles, old_lp, item, vector))

            if len(valid) < 2:
                print("   Skip: insufficient valid Pareto samples")
                continue

            batch_token_ids = [x[0] for x in valid]
            batch_smiles_valid = [x[1] for x in valid]
            batch_old_logprobs = [x[2] for x in valid]
            valid_items = [x[3] for x in valid]
            vectors = np.asarray([x[4] for x in valid], dtype=np.float64)

            ranks, fronts = fast_non_dominated_sort(vectors)
            distances = crowding_distance(vectors, fronts)
            pareto_scores, preference = pareto_training_scores(
                vectors,
                ranks,
                distances,
                preference_weights=preference_weights,
                random_preference=random_preference,
                crowding_coef=crowding_coef,
                scalar_hint_coef=scalar_hint_coef,
            )
            weighted_rewards = vectors @ logging_weights

            records = []
            for token_ids, smiles, old_lp, item, vector, rank, score, weighted in zip(
                batch_token_ids,
                batch_smiles_valid,
                batch_old_logprobs,
                valid_items,
                vectors,
                ranks,
                pareto_scores,
                weighted_rewards,
            ):
                records.append({
                    "token_ids": token_ids,
                    "smiles": smiles,
                    "canonical_smiles": item.get("canonical_smiles") or canonicalize_smiles(smiles),
                    "old_logprob": float(old_lp),
                    "objectives": vector,
                    "pareto_rank": int(rank),
                    "pareto_score": float(score),
                    "weighted_reward": float(weighted),
                })
            archive.add_many(records)

            mixed_token_ids = list(batch_token_ids)
            mixed_rewards = list(pareto_scores)
            mixed_old_logprobs = list(batch_old_logprobs)

            archive_samples = archive.sample(archive_replay_batch_size)
            if archive_samples:
                mixed_token_ids += [x["token_ids"] for x in archive_samples]
                mixed_rewards += [x["pareto_score"] for x in archive_samples]
                mixed_old_logprobs += [x["old_logprob"] for x in archive_samples]
                print(f"   Pareto archive replay: +{len(archive_samples)}")

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

            mean_pareto_score = float(np.mean(pareto_scores))
            top1_pareto_score = float(np.max(pareto_scores))
            top5_pareto_score = float(np.mean(sorted(pareto_scores, reverse=True)[:min(5, len(pareto_scores))]))
            mean_weighted_reward = float(np.mean(weighted_rewards))
            top1_weighted_reward = float(np.max(weighted_rewards))
            top5_weighted_reward = float(np.mean(sorted(weighted_rewards, reverse=True)[:min(5, len(weighted_rewards))]))
            front0_count = int(np.sum(ranks == 0))
            front0_rate = float(front0_count / len(ranks))

            row = {
                "iteration": it,
                "step": step,
                "global_step": global_step,
                "mean_pareto_score": mean_pareto_score,
                "top1_pareto_score": top1_pareto_score,
                "top5_pareto_score": top5_pareto_score,
                "mean_weighted_reward": mean_weighted_reward,
                "top1_weighted_reward": top1_weighted_reward,
                "top5_weighted_reward": top5_weighted_reward,
                "front0_count": front0_count,
                "front0_rate": front0_rate,
                "mean_pareto_rank": float(np.mean(ranks)),
                "archive_size": len(archive),
                "policy_loss": float(pol_loss.item()),
                "kl_loss": float(kl_loss.item()),
                "total_loss": float(total_loss.item()),
                "ratio_mean": float(r_mean.item()),
                "lr": current_lr,
                "kl_beta": kl_beta,
                "kept_samples": kept_count,
                "updates_applied": updates_applied,
                "grad_norm": grad_norm_value,
                "clip_eps": clip_eps,
                "mu": mu,
                "ratio_min": ratio_min,
                "ratio_max": ratio_max,
                "valid_count": len(valid),
                "unique_smiles_rate": unique_smiles_rate(batch_smiles_valid),
            }
            for name, value in zip(objective_names, preference):
                row[f"pref_{name}"] = float(value)

            top5_idx = np.argsort(-weighted_rewards)[:min(5, len(weighted_rewards))]
            for idx, name in enumerate(objective_names):
                row[f"mean_{name}"] = float(np.mean(vectors[:, idx]))
                row[f"top5_{name}"] = float(np.mean(vectors[top5_idx, idx]))

            pd.DataFrame([row]).to_csv(log_path, mode="a", index=False, header=False)

            objective_summary = " | ".join(
                f"{name}={row[f'mean_{name}']:.3f}" for name in objective_names
            )
            print(
                f"   pareto={mean_pareto_score:.3f} | weighted={mean_weighted_reward:.3f} | "
                f"front0={front0_count}/{len(ranks)} | archive={len(archive)}\n"
                f"   {objective_summary}\n"
                f"   policy={pol_loss.item():.4f} | kl={kl_loss.item():.4f} | "
                f"ratio={r_mean.item():.3f} | lr={current_lr:.2e} | kept={kept_count}"
            )

            if mean_pareto_score > best_pareto_score:
                best_pareto_score = mean_pareto_score
                best_pareto_state = copy.deepcopy(agent.state_dict())
                print(f"   New best Pareto score: {best_pareto_score:.4f}")

            if mean_weighted_reward > best_weighted_reward:
                best_weighted_reward = mean_weighted_reward
                best_weighted_state = copy.deepcopy(agent.state_dict())
                print(f"   New best weighted reward: {best_weighted_reward:.4f}")

    archive.save_csv(archive_path)
    print(f" Pareto training log saved to: {log_path}")
    print(f" Pareto archive saved to: {archive_path}")

    if best_pareto_state is not None:
        save_path = os.path.join(save_dir, f"best_pareto_gspo_{timestamp}.pt")
        torch.save(best_pareto_state, save_path)
        print(f" Saved best Pareto model: {save_path}")

    if best_weighted_state is not None:
        save_path = os.path.join(save_dir, f"best_weighted_pareto_gspo_{timestamp}.pt")
        torch.save(best_weighted_state, save_path)
        print(f" Saved best weighted model: {save_path}")


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

    train_pareto_gspo(
        agent,
        vocab,
        optimizer,
        scheduler,
        device=device,
        iterations=1,
        M=1000,
        batch_size=8,
        mu=1,
        clip_eps=0.2,
        kl_beta=0.15,
        temperature=1.0,
        top_k=None,
        log_dir=os.path.join(BASE_DIR, "rl/feedback/logs"),
        save_dir=os.path.join(BASE_DIR, "rl/feedback/best_models"),
        archive_max_size=500,
        archive_replay_batch_size=0,
        ratio_min=0.5,
        ratio_max=1.5,
        pad_token_id=0,
        logging_weights=(1.0, 0.2, 0.2, 0.0),
        preference_weights=None,
        random_preference=True,
        crowding_coef=0.05,
        scalar_hint_coef=0.10,
    )
