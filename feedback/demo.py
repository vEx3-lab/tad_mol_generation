import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import selfies as sf
import subprocess
import os

# =========================
# 假设你已有生成模型 model
# =========================
# model: decoder-only transformer
# tokenizer: 包含 token_to_idx 和 idx_to_token
# device = "cuda" or "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=1e-5)

# =========================
# RL 微调函数
# =========================
def rlp_step(model, tokenizer, batch_selfies, reward_func, gamma=0.99, clip_epsilon=0.2):
    log_probs_list = []
    rewards_list = []

    for sf_str in batch_selfies:
        smi = sf.decoder(sf_str)
        tokens = [tokenizer['token_to_idx'][c] for c in smi if c in tokenizer['token_to_idx']]
        if len(tokens) < 2:
            continue

        x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs=probs)

        log_prob = dist.log_prob(y)
        log_probs_list.append(log_prob.sum())

        # ====== 调用外部奖励函数 ======
        reward = reward_func(smi)
        rewards_list.append(torch.tensor([reward], dtype=torch.float, device=device))

    if len(log_probs_list) == 0:
        return 0.0

    log_probs = torch.stack(log_probs_list)
    rewards = torch.stack(rewards_list).squeeze()

    # discounted rewards
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.stack(discounted_rewards)

    advantages = discounted_rewards - discounted_rewards.mean()
    ratio = torch.exp(log_probs - log_probs.detach())
    loss = -torch.min(ratio * advantages,
                      torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# =========================
# 示例奖励函数：调用 vina 打分
# =========================
def vina_reward(smiles):
    """
    smiles -> pdbqt -> vina 打分 -> 返回结合能
    这里返回负能量作为 reward（越低越好）
    """
    output_dir = "./temp_pdbqt/"
    os.makedirs(output_dir, exist_ok=True)
    pdbqt_file = os.path.join(output_dir, "ligand.pdbqt")

    # 假设你已有 from_smi_2_pdbqt() 函数
    status = from_smi_2_pdbqt(smiles, pdbqt_file)
    if status != "success":
        return 0.0  # 转换失败，reward 为 0

    # 调用 vina 计算打分
    v = Vina(sf_name='vina')
    v.set_receptor("./receptor/8SC7.pdbqt")
    v.set_ligand_from_file(pdbqt_file)
    v.compute_vina_maps(center=[24.7,7.5,58.6], box_size=[79.5,76.8,61.5])
    energy = v.score()[0]
    return -energy  # reward 越大越好

# =========================
# 训练循环
# =========================
batch_selfies = ["[C][O][C]", "[C][C][O][H]"]  # 可以用你生成的 SELFIES 批次
for epoch in range(10):
    loss = rlp_step(model, tokenizer, batch_selfies, vina_reward)
    print(f"Epoch {epoch} RL loss: {loss:.4f}")
