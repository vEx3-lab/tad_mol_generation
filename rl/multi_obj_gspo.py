# ===== GSPO 批量训练脚本 (Decoder-Only Transformer) =====
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
from utils import make_reward_fn_from_vina,sample_selfies_batch_from_generate_selfies
import random
from utils import make_composite_reward


# -----------------  GSPO loss（sequence-level PPO）-----------------
# def compute_gspo_loss_batch(
#     agent,
#     old_agent,
#     ref_agent,
#     batch_sequences,
#     batch_rewards,
#     clip_eps=0.2,
#     kl_beta=0.05,
#     eps=1e-8,
# ):
#     device = next(agent.parameters()).device
#     rewards = torch.tensor(batch_rewards, device=device)
#     adv = (rewards - rewards.mean()) / (rewards.std() + eps)
#     adv = adv.detach()
#
#     policy_losses = []
#     kl_losses = []
#     ratio_means = []
#
#     for i, seq in enumerate(batch_sequences):
#         seq = torch.tensor(seq, device=device)
#         if len(seq) < 2:
#             continue
#
#         input_ids = seq[:-1].unsqueeze(0)
#         target_ids = seq[1:].unsqueeze(0)
#
#         logits_new = agent(input_ids)
#         logits_old = old_agent(input_ids).detach()
#         logits_ref = ref_agent(input_ids).detach()
#
#         logp_new = F.log_softmax(logits_new, dim=-1)
#         logp_old = F.log_softmax(logits_old, dim=-1)
#         logp_ref = F.log_softmax(logits_ref, dim=-1)
#
#         token_logp_new = logp_new.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
#         token_logp_old = logp_old.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
#         token_logp_ref = logp_ref.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
#
#         # ===== GSPO sequence ratio =====
#         log_ratio_seq = (token_logp_new - token_logp_old).mean()
#
#         ratio_seq = torch.exp(log_ratio_seq)
#         ratio_means.append(ratio_seq.detach())
#
#         surr1 = ratio_seq * adv[i]
#         surr2 = torch.clamp(ratio_seq, 1 - clip_eps, 1 + clip_eps) * adv[i]
#         policy_losses.append(-torch.min(surr1, surr2))
#
#         # ===== KL  =====
#         log_ratio_seq = (token_logp_ref - token_logp_new).mean()
#
#         # f-KL on sequence
#         kl = torch.exp(log_ratio_seq) - log_ratio_seq - 1
#         kl_losses.append(kl)
#
#
#     policy_loss = torch.stack(policy_losses).mean()
#     kl_loss = torch.stack(kl_losses).mean()
#     ratio_mean = torch.stack(ratio_means).mean()
#
#     total_loss = policy_loss + kl_beta * kl_loss
#     return total_loss, policy_loss, kl_loss, ratio_mean

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

        # ===== 新增：构建mask，标记padding位置 =====
        mask = (target_ids != pad_token_id).float()  # [1, seq_len]

        logits_new = agent(input_ids)
        logits_old = old_agent(input_ids).detach()
        logits_ref = ref_agent(input_ids).detach()

        logp_new = F.log_softmax(logits_new, dim=-1)
        logp_old = F.log_softmax(logits_old, dim=-1)
        logp_ref = F.log_softmax(logits_ref, dim=-1)

        token_logp_new = logp_new.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logp_old = logp_old.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        token_logp_ref = logp_ref.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        # ===== mask加权平均，替换原来的.mean() =====
        valid_len = mask.sum() + eps

        log_ratio_policy = ((token_logp_new - token_logp_old) * mask).sum() / valid_len
        ratio_seq = torch.exp(log_ratio_policy)
        ratio_means.append(ratio_seq.detach())

        surr1 = ratio_seq * adv[i]
        surr2 = torch.clamp(ratio_seq, 1 - clip_eps, 1 + clip_eps) * adv[i]
        policy_losses.append(-torch.min(surr1, surr2))

        # ===== KL =====
        log_ratio_kl = ((token_logp_ref - token_logp_new) * mask).sum() / valid_len
        kl = torch.exp(log_ratio_kl) - log_ratio_kl - 1
        kl_losses.append(kl)

    policy_loss = torch.stack(policy_losses).mean()
    kl_loss = torch.stack(kl_losses).mean()
    ratio_mean = torch.stack(ratio_means).mean()

    total_loss = policy_loss + kl_beta * kl_loss
    return total_loss, policy_loss, kl_loss, ratio_mean


# ----------------- 4. 训练主循环（GRPO）-----------------
# def train_gspo(
#         agent,
#         vocab,
#         optimizer,
#         device="cuda",
#         iterations=100,  # 减少外循环次数
#         M=4,  #  新增：内循环次数
#         batch_size=8,
#         max_len=80,
#         mu=4,
#         clip_eps=0.1,
#         kl_beta=0.1,
#         temperature=1.0,
#         top_k=10,
#         log_dir="./feedback/logs",
#         save_dir="./feedback/best_models"
# ):
#     agent.to(device)
#     old_agent = copy.deepcopy(agent).to(device)
#     ref_agent = copy.deepcopy(agent).to(device)
#
#     old_agent.eval()
#     ref_agent.eval()
#
#     os.makedirs(log_dir, exist_ok=True)
#     os.makedirs(save_dir, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_path = os.path.join(log_dir, f"gspo_training_log_{timestamp}.csv")
#
#     # CSV列：添加step到
#     if not os.path.exists(log_path):
#         pd.DataFrame(columns=[
#             "iteration", "step", "mean_reward", "mean_vina","mean_qed",
#                 "mean_sa","mean_logp","top1_reward", "top5_mean_reward",
#             "policy_loss", "kl_loss", "ratio_mean"
#         ]).to_csv(log_path, index=False)
#
#     best_total_loss = float("inf")
#     best_reward = float("-inf")
#     best_total_loss_state = None
#     best_reward_state = None
#
#     # ==================== 澶栧惊鐜細Iterations ====================
#     for it in range(1, iterations + 1):
#         print(f"\n{'=' * 70}")
#         print(f" Iteration {it}/{iterations}")
#         print(f"{'=' * 70}")
#
#         # =====  更新 reference model（iteration级别，只在外循环更新）=====
#         ref_agent.load_state_dict(agent.state_dict())
#         for p in ref_agent.parameters():
#             p.requires_grad_(False)
#
#         # ==================== 涓惊鐜細M Steps ====================
#         for step in range(1, M + 1):
#             print(f" [Iter {it}, Step {step}/{M}]")
#
#             # =====  更新 old policy（每个step都更新）=====
#             old_agent.load_state_dict(agent.state_dict())
#             for p in old_agent.parameters():
#                 p.requires_grad_(False)
#
#             # ===== Rollout（使用old_agent 采样）====
#             batch_token_ids, batch_smiles = sample_selfies_batch_from_generate_selfies(
#                 vocab=vocab,
#                 model=old_agent,  # 浣跨敤old_agent閲囨牱
#                 batch_size=batch_size,
#                 max_len=max_len,
#                 temperature=temperature,
#                 top_k=top_k,
#                 device=device,
#             )
#
#             if len(batch_token_ids) < 2:
#                 print(f"   Skip: no valid samples")
#                 continue
#
#             # ===== Docking =====
#             vina_results = batch_scores_from_vina(
#                 batch_smiles,
#                 receptor_file="../feedback/8sc7.pdbqt",
#                 pdbqt_dir="../feedback/temp/pdbqt",
#                 output_dir="../feedback/vina_results/"
#             )
#
#             reward_items = make_composite_reward(
#                 batch_smiles,
#                 vina_results,
#                 weights={'vina': 0.5, 'qed': 0.2, 'sa': 0.2, 'logp': 0.1},
#                 invalid_penalty=0
#             )
#
#             rewards = [item['reward'] for item in reward_items]
#
#             # ===== Filter invalid =====
#             valid = [(s, r) for s, r in zip(batch_token_ids, rewards) if r != 0]
#             if len(valid) < 2:
#                 print(f"   Skip: invalid rewards")
#                 continue
#
#             batch_token_ids, rewards = zip(*valid)
#
#             mean_reward = float(np.mean(rewards))
#             top1_reward = max(rewards)
#             top5_mean_reward = float(np.mean(sorted(rewards, reverse=True)[:min(5, len(rewards))]))
#             mean_vina = float(np.mean([item['vina'] for item in reward_items]))
#             mean_qed = float(np.mean([item['qed'] for item in reward_items]))
#             mean_sa = float(np.mean([item['sa'] for item in reward_items]))
#             mean_logp = float(np.mean([item['logp'] for item in reward_items]))
#
#             # ==================== 内循环：Policy Updates ====================
#             for update_idx in range(mu):
#                 total_loss, pol_loss, kl_loss, r_mean = compute_gspo_loss_batch(
#                     agent,
#                     old_agent,
#                     ref_agent,
#                     batch_token_ids,
#                     rewards,
#                     clip_eps,
#                     kl_beta,
#                 )
#
#                 optimizer.zero_grad()
#                 total_loss.backward()
#
#                 #  娣诲姞姊害瑁佸壀
#                 torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
#
#                 optimizer.step()
#
# ===== 记录到CSV =====
#             pd.DataFrame([{
#                 "iteration": it,
#                 "step": step,
#                 "mean_reward": mean_reward,
#                 "mean_vina":mean_vina,
#                 "mean_qed":mean_qed,
#                 "mean_sa":mean_sa,
#                 "mean_logp":mean_logp,
#                 "top1_reward": top1_reward,
#                 "top5_mean_reward": top5_mean_reward,
#                 "policy_loss": pol_loss.item(),
#                 "kl_loss": kl_loss.item(),
#                 "ratio_mean": r_mean.item()
#             }]).to_csv(log_path, mode="a", index=False, header=False)
#
#             print(
#                 f"   meanR={mean_reward:.3f} | top1={top1_reward:.3f} | "
#                 f"mean_vina = {mean_vina:.3f} | mean_qed = {mean_qed:.3f}|"
#                 f"mean_sa = {mean_sa:.3f} | mean_logp = {mean_logp:.3f}|"
#                 f"top5={top5_mean_reward:.3f}\n"
#                 f"   policy={pol_loss.item():.4f} | kl={kl_loss.item():.4f} | "
#                 f"ratio={r_mean.item():.3f}"
#             )
#
#             # ===== 淇濆瓨鏈€浼樻ā鍨?=====
#             if total_loss.item() < best_total_loss:
#                 best_total_loss = total_loss.item()
#                 best_total_loss_state = copy.deepcopy(agent.state_dict())
#                 print(f"   New best total loss: {best_total_loss:.4f}")
#
#             if mean_reward > best_reward:
#                 best_reward = mean_reward
#                 best_reward_state = copy.deepcopy(agent.state_dict())
#                 print(f"   New best reward: {best_reward:.3f}")
#
#     # ===== 鏈€缁堜繚瀛樻ā鍨?=====
#     if best_total_loss_state is not None:
#         save_path = os.path.join(save_dir, f"best_total_loss_{timestamp}.pt")
#         torch.save(best_total_loss_state, save_path)
#         print(f" Saved best loss model: {save_path}")
#
#     if best_reward_state is not None:
#         save_path = os.path.join(save_dir, f"best_reward_{timestamp}.pt")
#         torch.save(best_reward_state, save_path)
#         print(f" Saved best reward model: {save_path}")
# ===== ReplayBuffer 类（加在文件顶部，import之后）?====
import heapq

class ReplayBuffer:
    def __init__(self, max_size=100):
        self.buffer = []  # (reward, idx, token_ids, smiles) idx用于打破reward相等时的比较
        self.buffer = []  # (reward, idx, token_ids, smiles) idx用于打破reward相等时的比较
        self._counter = 0  # 唯一计数器，用于避免tuple比较时比token_ids
        self._counter = 0  # 唯一计数器，用于避免tuple比较时比token_ids

    def add(self, token_ids, smiles, reward):
        self._counter += 1
        heapq.heappush(self.buffer, (reward, self._counter, token_ids, smiles))
        if len(self.buffer) > self.max_size:
            heapq.heappop(self.buffer)  # 寮瑰嚭reward鏈€灏忕殑

    def sample(self, n):
        if len(self.buffer) == 0:
            return [], [], []
        n = min(n, len(self.buffer))
        rewards = np.array([r for r, _, _, _ in self.buffer])
        # softmax采样，reward越高越容易被选中
        probs = np.exp(rewards - rewards.max())
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.buffer), size=n, replace=False, p=probs)
        sampled = [self.buffer[i] for i in indices]
        token_ids_list = [s[2] for s in sampled]
        smiles_list = [s[3] for s in sampled]
        rewards_list = [s[0] for s in sampled]
        return token_ids_list, smiles_list, rewards_list

    def __len__(self):
        return len(self.buffer)


# ===== 淇敼鍚庣殑 train_gspo =====
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
        replay_buffer_size=300,    # 新增
        replay_batch_size=4,       # 新增：每step从buffer取多少条
        replay_start=30,           # 新增：buffer积累多少步后开始回放
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
            "iteration", "step", "mean_reward", "mean_vina", "mean_qed",
            "mean_sa", "mean_logp", "top1_reward", "top5_mean_reward",
            "policy_loss", "kl_loss", "total_loss", "ratio_mean",
            "lr", "kl_beta", "replay_size"
        ]).to_csv(log_path, index=False)

    best_total_loss = float("inf")
    best_reward = float("-inf")
    best_total_loss_state = None
    best_reward_state = None

    # ===== 鍒濆鍖?ReplayBuffer =====
    replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

    for it in range(1, iterations + 1):
        print(f"\n{'=' * 70}")
        print(f" Iteration {it}/{iterations}")
        print(f"{'=' * 70}")

        ref_agent.load_state_dict(agent.state_dict())
        for p in ref_agent.parameters():
            p.requires_grad_(False)

        for step in range(1, M + 1):
            print(f" [Iter {it}, Step {step}/{M}]")

            old_agent.load_state_dict(agent.state_dict())
            for p in old_agent.parameters():
                p.requires_grad_(False)

            # ===== Rollout =====
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
                print(f"   Skip: no valid samples")
                continue

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
                weights={'vina': 1.0, 'qed': 0.2, 'sa': 0.2, 'logp': 0},
                invalid_penalty=0
            )

            rewards = [item['reward'] for item in reward_items]

            # ===== Filter invalid =====
            # 修复：同时保存smiles用于Buffer
            valid = [
                (s, r, smi) for s, r, smi in zip(batch_token_ids, rewards, batch_smiles)
                if r != 0
            ]
            if len(valid) < 2:
                print(f"   Skip: invalid rewards")
                continue

            batch_token_ids, rewards, batch_smiles_valid = zip(*valid)

            # 修复：只统计valid样本的指标?
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

            # ===== 新分子存到ReplayBuffer =====
            for token_ids, smiles, reward in zip(batch_token_ids, batch_smiles_valid, rewards):
                replay_buffer.add(token_ids, smiles, reward)

            # ===== 娣峰悎鏂版暟鎹?+ 鍥炴斁鏁版嵁 =====
            if len(replay_buffer) >= replay_start:
                replay_ids, _, replay_rewards = replay_buffer.sample(replay_batch_size)
                mixed_token_ids = list(batch_token_ids) + replay_ids
                mixed_rewards = list(rewards) + replay_rewards
                print(f"   Replay: +{len(replay_ids)} samples | buffer size={len(replay_buffer)}")
            else:
                mixed_token_ids = list(batch_token_ids)
                mixed_rewards = list(rewards)
                print(f"   Replay: warming up ({len(replay_buffer)}/{replay_start})")

            # ===== 内循环：Policy Updates（使用混合数据）=====
            current_lr = optimizer.param_groups[0]["lr"]
            for update_idx in range(mu):
                total_loss, pol_loss, kl_loss, r_mean = compute_gspo_loss_batch(
                    agent,
                    old_agent,
                    ref_agent,
                    mixed_token_ids,   # 娣峰悎鏁版嵁
                    mixed_rewards,     # 娣峰悎rewards
                    clip_eps,
                    kl_beta,
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                optimizer.step()
                current_lr = optimizer.param_groups[0]["lr"]
                scheduler.step() # 浣欏鸡閫€鐏?
            # ===== 记录到CSV =====
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
                "total_loss": total_loss.item(),
                "ratio_mean": r_mean.item(),
                "lr": current_lr,
                "kl_beta": kl_beta,
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

            # ===== Early Stopping =====
            if kl_loss.item() >1:
                print(f"   Early stopping: KL loss too high ({kl_loss.item():.4f} > 0.5)")
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

# ===== 鍐荤粨 agent 澶ч儴鍒嗗弬鏁帮紝鍙井璋冩渶鍚庝竴灞傚拰杈撳嚭灞?=====
def freeze_agent_partial(agent):
    # 1. 鍐荤粨鍏ㄩ儴鍙傛暟
    for param in agent.parameters():
        param.requires_grad = False

    # 2. 瑙ｅ喕鏈€鍚庝竴灞?Transformer Block
    for param in agent.layers[-1].parameters():
        param.requires_grad = True

    # 3. 瑙ｅ喕杈撳嚭灞?lm_head
    # for param in agent.lm_head.parameters():
    #     param.requires_grad = True
    for param in agent.fc_out.parameters():
        param.requires_grad = True

    # 4. 鏀堕泦鍙傛暟淇℃伅
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
    # from model.v6 import decoder_only_lm
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
    # ===== 鍐荤粨閮ㄥ垎鍙傛暟 =====
    freeze_agent_partial(agent)

    # optimizer 鍙細鏇存柊 requires_grad=True 鐨勫弬鏁?
    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, agent.parameters()), lr=5e-6)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, agent.parameters()),
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=0.001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1000,  # 鎬籹tep鏁?
        eta_min=1e-7  # 鏈€浣巐r
    )
    # optimizer = optim.AdamW(agent.parameters(), lr=5e-5)

    train_gspo(
        agent,
        vocab,
        optimizer,
        scheduler,
        device=device,
        iterations=1,
        M=1000,
        batch_size=8,
        mu=2,
        clip_eps=0.2,
        kl_beta=0.1,
        temperature=1,
        top_k=20,
        log_dir=os.path.join(BASE_DIR, "rl/feedback/logs"),
        save_dir=os.path.join(BASE_DIR, "rl/feedback/best_models"),
        replay_start=50,
    )
