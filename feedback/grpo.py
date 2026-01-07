
import torch
from torch.distributions import Categorical
from feedback.utils import from_smi_2_pdbqt
from feedback.vina_scores import batch_scores_from_vina

import torch
from torch.distributions import Categorical


def reward_func(smiles):
    if smiles is None:
        return -10.0  # 非法分子惩罚

    pdbqt_file = f"./temp_ligand.pdbqt"
    status = from_smi_2_pdbqt(smiles, pdbqt_file)
    if status != "success":
        return -10.0

    score = batch_scores_from_vina(pdbqt_file)  # 你之前封装的 vina_score
    return -score  # kcal/mol 越低越好



import torch
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence

import os
import torch
import pandas as pd
import selfies as sf
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# 假设这些是你已经写好的模块
from generate_selfies import generate_selfies
from data.data_utils import selfies_vocab
from config.load_config import load_config
from model.decoder_only_tfm import decoder_only_tfm
from model.bi_lstm import bi_lstm
from feedback.vina_scores import batch_scores_from_vina


# ==========================================
# 1. 优化的 Log Prob 计算
# ==========================================
def compute_log_prob_batch(model, batch_tokens, pad_idx=0):
    """
    计算序列的 Log Probability
    注意：batch_tokens 必须包含 <sos> 起始符
    """
    device = next(model.parameters()).device

    # 转换为 Tensor 并 Padding
    # 假设 batch_tokens 是 list of list
    seq_tensors = [torch.tensor(seq, dtype=torch.long, device=device) for seq in batch_tokens]
    padded = pad_sequence(seq_tensors, batch_first=True, padding_value=pad_idx)  # [B, L]

    # 构建 Input (x) 和 Target (y)
    # x: [<sos>, A, B, C] -> 预测 -> A, B, C, <eos>
    x = padded[:, :-1]
    y = padded[:, 1:]

    # 创建 Mask (忽略 padding 部分)
    mask = (y != pad_idx).float()

    # 前向传播
    logits = model(x)  # [B, L-1, Vocab]

    # 获取目标 token 的 log_prob
    # 使用 gather 提取对应位置的 logit 可能会更省显存，但 Categorical 更直观
    dist = Categorical(logits=logits)
    token_log_probs = dist.log_prob(y)  # [B, L-1]

    # 应用 Mask
    token_log_probs = token_log_probs * mask

    # 求和得到整条序列的 log_prob
    seq_log_probs = token_log_probs.sum(dim=1)  # [B]

    return seq_log_probs


# ==========================================
# 2. 优化的 GRPO Loss (含归一化)
# ==========================================
def compute_grpo_loss(agent, old_model, batch_tokens, batch_rewards,
                      clip_eps=0.2, kl_beta=0.01, pad_idx=0):  # kl_beta 通常很小
    """
    GRPO Loss 计算核心
    """
    # 1. 计算新旧模型的 Log Prob
    # agent 需要梯度，old_model 不需要
    logp_new = compute_log_prob_batch(agent, batch_tokens, pad_idx=pad_idx)

    with torch.no_grad():
        logp_old = compute_log_prob_batch(old_model, batch_tokens, pad_idx=pad_idx)

    # 2. 计算 Advantage (关键修改：增加归一化)
    rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=logp_new.device)

    # GRPO 核心：组内归一化 (Group Normalization)
    # 这里假设整个 batch 就是一个 group (或者 batch 内包含多个 group)
    # 如果 batch_size=16 且都是同一个 prompt 生成的，直接算 mean/std
    mean_reward = rewards.mean()
    std_reward = rewards.std() + 1e-8
    advantages = (rewards - mean_reward) / std_reward

    # 3. 计算 Ratio
    # ratio = exp(log_new - log_old)
    ratio = torch.exp(logp_new - logp_old)

    # 4. PPO Clipped Loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    ppo_loss = -torch.min(surr1, surr2).mean()

    # 5. KL Divergence Penalty (近似)
    # 我们希望 logp_new 不要偏离 logp_old 太多
    # approx_kl = (ratio - 1) - log(ratio)  # 更精确的近似
    # 或者简单的: logp_old - logp_new (如果是正数，说明 new 的概率变小了)
    # 这里使用简单的 MSE 风格或者 ref_log_prob 约束
    # 常用写法：loss = policy_loss + beta * KL
    # 这里用 logp_new - logp_old 来观测，实际 Loss 加上 KL 惩罚
    # KL(new || old) approx = logp_new - logp_old (expectation)
    # 但为了防止模型崩坏，通常惩罚 KL
    kl_val = (logp_old - logp_new).mean()  # 这是一个观测值

    # 这里的 kl_loss 写法取决于你想怎么约束。
    # 标准 PPO 通常把 KL 放在 reward 里 (Reward - beta * KL)，或者作为额外 Loss项
    # 这里作为额外 Loss 项：
    # 简单的 KL 惩罚： (logp_new - logp_old)^2 * 0.5 (Schulman approx)
    # 或者直接用 logp_new 和 logp_old 的距离
    kl_loss = torch.mean((logp_new - logp_old) ** 2)

    total_loss = ppo_loss + kl_beta * kl_loss

    return total_loss, ppo_loss, kl_val


# ==========================================
# 3. 训练循环 (Training Loop)
# ==========================================
def train_grpo():
    config = load_config("./config/decoder_only_tfm_config.yaml")
    model_name = config["model_name"]
    device = config["device"]

    # ... (加载数据和 Vocab 代码不变) ...
    data_file = config["paths"]["data_file"]
    data = pd.read_csv(data_file)["selfies"].tolist()
    vocab = selfies_vocab(data)
    save_dir = config["paths"]["save_dir"].format(model_name=model_name)
    os.makedirs(save_dir, exist_ok=True)
    model_cfg = config["model"]

    # 初始化 Agent
    if model_name == "decoder_only_tfm":
        agent = decoder_only_tfm(vocab_size=len(vocab), **model_cfg).to(device)
    elif model_name == "bi_lstm":
        agent = bi_lstm(vocab_size=len(vocab), **model_cfg).to(device)

    # 加载预训练权重
    best_model_path = os.path.join(save_dir, 'best_model_fold2')
    if os.path.exists(best_model_path):
        print(f"Loading pretrained model from {best_model_path}")
        agent.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)

    # 初始化 Old Model (Reference Model)
    old_model = type(agent)(**{"vocab_size": len(vocab), **model_cfg}).to(device)
    old_model.load_state_dict(agent.state_dict())
    old_model.eval()  # 永远是 eval 模式
    for param in old_model.parameters():
        param.requires_grad = False  # 冻结参数

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-5)  # RL 学习率通常要比预训练小

    # 超参数
    batch_size = 16  # 建议设大一点，比如 64，GRPO 需要样本多样性
    epochs = 10
    max_len = config["generate"]["max_len"]
    temperature = 1.0  # 采样温度，RL 需要一定的随机性来探索

    print("Start GRPO training...")

    for epoch in range(epochs):
        agent.eval()  # 生成阶段使用 eval 模式

        batch_selfies = []
        batch_sequences = []

        # --- 阶段 1: 采样 (Rollout) ---
        # 建议：如果 generate_selfies 支持 batch，请移出循环
        print(f"Sampling batch {epoch}...")
        with torch.no_grad():
            for _ in range(batch_size):
                # 注意：这里生成的 token list 必须包含 <sos> 对应的 index
                # 如果 generate_selfies 返回的不含 sos，你需要手动 insert
                g = generate_selfies(model_name=model_name, vocab=vocab, model=agent,
                                     device=device, max_len=max_len, temperature=temperature, top_k=50)

                s_str = g["selfies"]
                # 检查是否为空
                if not s_str: continue

                try:
                    s_tokens = list(sf.split_selfies(s_str))
                    # 转换 token ID
                    token_ids = [vocab[t] for t in s_tokens if t in vocab]

                    # !!! 关键：确保开头有 <sos> !!!
                    # 假设 vocab['<sos>'] 是起始符 ID
                    if '<sos>' in vocab and (not token_ids or token_ids[0] != vocab['<sos>']):
                        token_ids.insert(0, vocab['<sos>'])

                    batch_selfies.append(s_str)
                    batch_sequences.append(token_ids)
                except Exception as e:
                    print(f"Parse error: {e}")
                    continue

        if len(batch_sequences) < 2:  # 样本太少无法计算 std
            continue

        # --- 阶段 2: 评估 (Reward Calculation) ---
        batch_smiles = []
        for s in batch_selfies:
            try:
                smi = sf.decoder(s)
                batch_smiles.append(smi)
            except:
                batch_smiles.append(None)

        # 批量调用 Vina
        # 假设 batch_scores_from_vina 处理了 None 的情况或者我们在下面处理
        valid_smiles_for_scoring = [s for s in batch_smiles if s is not None]

        if not valid_smiles_for_scoring:
            continue

        # 这里假设 batch_scores_from_vina 返回一个 dict: {smiles: {'score': -8.5}}
        reward_map = batch_scores_from_vina(
            valid_smiles_for_scoring,
            receptor_file='./feedback/8sc7.pdbqt',
            pdbqt_dir='./feedback/temp/temp_pdbqt',
            output_dir='./feedback/vina_results'
        )

        batch_rewards = []
        for smi in batch_smiles:
            if smi is None:
                batch_rewards.append(-10.0)  # 非法分子惩罚
            else:
                res = reward_map.get(smi)
                if res and res['score'] is not None:
                    # Vina 分数越低越好 (-10 优于 -5)
                    # 我们需要 Reward 越高越好
                    # 策略：Reward = -1 * Vina_Score
                    # 例如：-12 -> +12, -5 -> +5
                    batch_rewards.append(-1.0 * res['score'])
                else:
                    batch_rewards.append(-10.0)  # 对接失败惩罚

        # --- 阶段 3: 更新 (Update) ---
        agent.train()  # 切换回训练模式

        # 过滤掉异常短的序列
        final_seqs = []
        final_rewards = []
        for seq, r in zip(batch_sequences, batch_rewards):
            if len(seq) > 1:
                final_seqs.append(seq)
                final_rewards.append(r)

        if not final_seqs: continue

        # 计算 Loss 并更新
        loss, ppo_loss, kl_val = compute_grpo_loss(
            agent, old_model, final_seqs, final_rewards,
            clip_eps=0.2, kl_beta=0.05
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)  # 梯度裁剪防止爆炸
        optimizer.step()

        # 日志
        mean_r = sum(final_rewards) / len(final_rewards)
        print(f"[Epoch {epoch}] Loss: {loss.item():.4f} | Mean Reward: {mean_r:.4f} | KL: {kl_val.item():.4f}")

        # 定期同步 Old Model
        if (epoch + 1) % 5 == 0:
            old_model.load_state_dict(agent.state_dict())
            # 保存检查点
            torch.save(agent.state_dict(), os.path.join(save_dir, f"grpo_epoch_{epoch}.pt"))


if __name__ == "__main__":
    train_grpo()
