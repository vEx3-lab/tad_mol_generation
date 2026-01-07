## 此版本代码为尚未按照伪代码实现的grpo，其实现过程主要在更新参数环节上存在部分问题#

# ===== GRPO 批量训练脚本 (Decoder-Only Transformer) =====
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nn_utils
import pandas as pd
import numpy as np
from datetime import datetime
import selfies as sf

from feedback.vina_scores import batch_scores_from_vina
from generate_selfies import generate_selfies

# ----------------- 1. 生成 batch -----------------
def sample_selfies_batch_from_generate_selfies(
    model_name, vocab, model, batch_size=16, max_len=80, temperature=1.0, top_k=None,
    device="cuda", save_dir=None, epoch=None
):
    batch_token_ids = []
    batch_selfies = []
    batch_smiles = []

    model_mode_backup = model.training
    model.eval()  # eval 保证生成时 dropout 不影响
    with torch.no_grad():
        for _ in range(batch_size):
            result = generate_selfies(
                model_name=model_name,
                vocab=vocab,
                model=model,
                device=device,
                max_len=max_len,
                temperature=temperature,
                top_k=top_k
            )
            token_ids = [vocab[t] for t in sf.split_selfies(result["selfies"])]
            batch_token_ids.append(token_ids)
            batch_selfies.append(result["selfies"])
            batch_smiles.append(result["smiles"])

    # 恢复原来的模式
    if model_mode_backup:
        model.train()

    # 保存 CSV
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"generated_epoch{epoch}.csv" if epoch is not None else "generated.csv"
        df = pd.DataFrame({"selfies": batch_selfies, "smiles": batch_smiles})
        df.to_csv(os.path.join(save_dir, file_name), index=False)

    return batch_token_ids, batch_selfies, batch_smiles

# ----------------- 2. 批量 GRPO loss -----------------
def compute_grpo_loss_batch(
        agent,
        old_agent,
        ref_agent,
        batch_sequences,
        batch_rewards,
        clip_eps=0.2,
        kl_beta=0.05,
        pad_token_id=0,
        eps=1e-8
):
    device = next(agent.parameters()).device
    batch_size = len(batch_sequences)
    max_len = max(len(seq) for seq in batch_sequences)

    # ===== 构建 batch tensor =====
    input_ids = torch.full((batch_size, max_len-1), pad_token_id, dtype=torch.long, device=device)
    target_ids = torch.full((batch_size, max_len-1), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(input_ids, dtype=torch.float32)

    for i, seq in enumerate(batch_sequences):
        if len(seq) < 2: continue
        seq_tensor = torch.tensor(seq, dtype=torch.long, device=device)
        input_ids[i, :len(seq)-1] = seq_tensor[:-1]
        target_ids[i, :len(seq)-1] = seq_tensor[1:]
        mask[i, :len(seq)-1] = 1.0

    # ===== Advantage 归一化 =====
    rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
    mean_r = rewards_tensor.mean()
    std_r = rewards_tensor.std().clamp(min=eps)
    advantages = ((rewards_tensor - mean_r) / std_r).unsqueeze(1)  # [B,1]

    # ===== 批量 forward =====
    logits_new = agent(input_ids)
    with torch.no_grad():
        logits_old = old_agent(input_ids)
        logits_ref = ref_agent(input_ids)

    logp_new = F.log_softmax(logits_new, dim=-1)
    logp_old = F.log_softmax(logits_old, dim=-1)
    logp_ref = F.log_softmax(logits_ref, dim=-1)

    # ===== 取生成 token 的 logp =====
    token_logp_new = logp_new.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    token_logp_old = logp_old.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    # ===== Policy loss =====
    ratio = torch.exp(token_logp_new - token_logp_old)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -(torch.min(surr1, surr2) * mask).sum() / mask.sum().clamp(min=1e-6)

    # ===== KL loss (严格对整个词表) =====
    kl_per_pos = (logp_ref.exp() * (logp_ref - logp_new)).sum(-1)  # [B,T]
    kl_loss = (kl_per_pos * mask).sum() / mask.sum()

    # ===== 总 loss =====
    total_loss = policy_loss + kl_beta * kl_loss
    return total_loss, policy_loss, kl_loss

# ----------------- 3. reward 函数 -----------------
def make_reward_fn_from_vina(vina_results, invalid_penalty=0):
    def reward_fn(smiles):
        res = vina_results.get(smiles, None)
        if res is None: return invalid_penalty
        score = res.get("score", None)
        if score is None: return invalid_penalty
        return -score  # Vina 越小越好 → reward 越大
    return reward_fn

# ----------------- 4. 训练函数 -----------------
def train_grpo(
    agent,
    vocab,
    optimizer,
    device="cuda",
    epochs=100,
    batch_size=16,
    max_len=80,
    clip_eps=0.1,
    kl_beta=0.05,
    temperature=1.0,
    top_k=None,
    sync_old_every=1,
    sync_ref_every =5,
    log_dir="./feedback/logs",
    save_dir="./feedback/best_models",
    pad_token_id=0
):
    agent.to(device)
    old_agent = copy.deepcopy(agent).to(device)
    old_agent.eval()
    ref_agent = copy.deepcopy(agent).to(device)
    ref_agent.eval()

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"grpo_training_log_{timestamp}.csv")

    # CSV 表头
    pd.DataFrame(columns=[
        "epoch","valid_sequences","mean_reward","top1_reward",
        "top5_mean_reward","policy_loss","kl_loss","adv_mean","kl_beta"
    ]).to_csv(log_path, index=False)

    best_total_loss = float("inf")
    best_reward = float("-inf")
    best_total_loss_state = None
    best_reward_state = None

    for epoch in range(1, epochs+1):
        # ===== Rollout =====
        batch_token_ids, batch_selfies, batch_smiles = sample_selfies_batch_from_generate_selfies(
            model_name="decoder_only_tfm",
            vocab=vocab,
            model=old_agent,
            batch_size=batch_size,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            device=device,
            save_dir="./logs/generated_smiles",
            epoch=epoch
        )

        # ===== Docking =====
        vina_results = batch_scores_from_vina(
            batch_smiles,
            receptor_file='./feedback/8sc7.pdbqt',
            pdbqt_dir='./feedback/temp/pdbqt',
            output_dir='./feedback/vina_results/'
        )
        reward_fn = make_reward_fn_from_vina(vina_results)

        # ===== Reward 计算 =====
        batch_rewards = [reward_fn(smi) for smi in batch_smiles]
        valid_sequences = batch_token_ids
        valid_smiles = batch_smiles

        if len(valid_sequences) < 2:
            print(f"[Epoch {epoch}] skip (too few valid samples)")
            continue

        # top5 / top1
        reward_smi_pairs = sorted(zip(batch_rewards, valid_smiles), key=lambda x: x[0], reverse=True)
        top5_rewards = [x[0] for x in reward_smi_pairs[:5]]
        top1_reward = top5_rewards[0]
        top5_mean_reward = float(np.mean(top5_rewards))
        mean_reward = float(np.mean(batch_rewards))

        # ===== GRPO loss =====
        total_loss, policy_loss, kl_loss = compute_grpo_loss_batch(
            agent, old_agent, ref_agent,
            batch_sequences=valid_sequences,
            batch_rewards=batch_rewards,
            clip_eps=clip_eps,
            kl_beta=kl_beta,
            pad_token_id=pad_token_id
        )

        # ===== 更新 =====
        optimizer.zero_grad()
        total_loss.backward()
        nn_utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        # ===== 同步 old_agent =====
        if epoch % sync_old_every == 0:
            old_agent.load_state_dict(agent.state_dict())

        if epoch % sync_ref_every == 0:
            ref_agent.load_state_dict(agent.state_dict())

        # ===== Log =====
        print(
            f"[Epoch {epoch:03d}] valid={len(valid_sequences):02d} | "
            f"meanR={mean_reward:.3f} | top5={top5_mean_reward:.3f} | "
            f"policy={policy_loss.item():.4f} | kl={kl_loss.item():.4f} | "
        )
        df_epoch = pd.DataFrame([{
            "epoch": epoch,
            "valid_sequences": len(valid_sequences),
            "mean_reward": mean_reward,
            "top1_reward": top1_reward,
            "top5_mean_reward": top5_mean_reward,
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "kl_beta": kl_beta
        }])
        df_epoch.to_csv(log_path, mode='a', index=False, header=False)

        # ===== 保存最优模型 =====
        if total_loss.item() < best_total_loss:
            best_total_loss = total_loss.item()
            best_total_loss_state = copy.deepcopy(agent.state_dict())
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_reward_state = copy.deepcopy(agent.state_dict())

    # ===== 保存模型 =====
    if best_total_loss_state is not None:
        torch.save(best_total_loss_state, os.path.join(save_dir, f"best_total_loss_{timestamp}.pt"))
    if best_reward_state is not None:
        torch.save(best_reward_state, os.path.join(save_dir, f"best_reward_{timestamp}.pt"))

if __name__ == '__main__':
    import torch
    import pandas as pd
    from data.data_utils import selfies_vocab
    from config.load_config import load_config
    import os
    from model.decoder_only_tfm import decoder_only_tfm
    import torch.optim as optim
    # ===== 读取配置文件 =====
    config_path = "./config/decoder_only_tfm_config.yaml"
    # config_path = "config/bi_lstm_config.yaml"
    config = load_config(config_path)

    model_name = config["model_name"]
    device = config["device"]
    # ===== 载入数据 =====
    data_file = config["paths"]["data_file"]
    data = pd.read_csv(data_file)["selfies"].tolist()

    vocab = selfies_vocab(data)
    model_cfg = config['model']
    # ===== SMILES 生成 =====
    # 这里载入最后一个 fold 的最优模型
    # best_model_path = './model/decoder_only_tfm/20251229_140125/best.pt'
    best_model_path = './model/decoder_only_tfm_best/best_model_fold2.pt'
    model_path = best_model_path
    print(model_path)
    num_samples = config["generate"]["num_samples"]
    temperature = config["generate"]["temperature"]
    top_k = config["generate"]["top_k"]
    max_len = config["generate"]["max_len"]
    agent = decoder_only_tfm(
        vocab_size=len(vocab),
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        max_len=model_cfg["max_len"],
        dropout=model_cfg["dropout"]
    ).to(device)
    state_dict = torch.load(best_model_path, map_location=device,weights_only=True)
    # 尽量兼容加载（允许少量不匹配）
    agent.load_state_dict(state_dict, strict=False)

    optimizer = optim.AdamW(agent.parameters(), lr=1e-4)
    train_grpo(
            agent,
            vocab,
            optimizer,
            device="cuda",
            epochs=300,
            batch_size=8,
            max_len=80,
            clip_eps=0.1,
            kl_beta=0.1,
            temperature=1.0,
            top_k=10,
            sync_old_every=1,
            sync_ref_every=10
    )