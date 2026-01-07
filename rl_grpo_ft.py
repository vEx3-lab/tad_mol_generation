
import os
import torch
import pandas as pd
import selfies as sf
from tqdm import tqdm
import torch.functional as F
from generate_selfies import generate_selfies
from data.data_utils import selfies_vocab
from config.load_config import load_config
from model.decoder_only_tfm import decoder_only_tfm
from model.bi_lstm import bi_lstm
from feedback.grpo import compute_grpo_loss
from feedback.vina_scores import batch_scores_from_vina

def sample_selfies_train(model, vocab, model_name,
                         device="cuda", max_len=80, temperature=1.0):
    """
    训练用 SELFIES 采样（GRPO/PPO 用），返回：
    - tokens：生成的 token 列表
    - log_probs：每步 log prob（用于 advantage / policy loss）
    """

    model.train()      # <-- 训练态
    sequences = ["<SOS>"]
    log_probs = []

    for step in range(max_len):

        input_ids = torch.tensor(
            [[vocab[t] for t in sequences]],
            device=device
        )

        # 不要 no_grad
        if model_name == "bi_lstm":
            logits = model(input_ids, use_forward_only=True)
        else:
            logits = model(input_ids)

        logits = logits[0, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        next_id = dist.sample()

        log_probs.append(dist.log_prob(next_id))
        next_token = vocab.id2token[next_id.item()]

        if next_token == "<EOS>":
            sequences.append(next_token)
            break

        sequences.append(next_token)

    return sequences, log_probs


def train_grpo():
    # ===== 读取配置 =====
    config = load_config("./config/decoder_only_tfm_config.yaml")
    model_name = config["model_name"]
    device = config["device"]

    # 载入训练数据构建 vocab
    data_file = config["paths"]["data_file"]
    data = pd.read_csv(data_file)["selfies"].tolist()
    vocab = selfies_vocab(data)

    max_len = config["generate"]["max_len"]
    temperature = config["generate"]["temperature"]
    top_k = config["generate"]["top_k"]

    save_dir = config["paths"]["save_dir"].format(model_name=model_name)
    os.makedirs(save_dir, exist_ok=True)

    model_cfg = config["model"]

    # ===== 创建 agent =====
    if model_name == "decoder_only_tfm":
        agent = decoder_only_tfm(vocab_size=len(vocab), **model_cfg).to(device)
    elif model_name == "bi_lstm":
        agent = bi_lstm(vocab_size=len(vocab), **model_cfg).to(device)
    else:
        raise ValueError("Unknown model_name")

    # 加载预训练模型
    best_model_path = os.path.join(save_dir, 'best_model_fold2.pt')
    if os.path.exists(best_model_path):
        print("Loading pretrained:", best_model_path)
        agent.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)

    # 创建 old_model（冻结）
    old_model = type(agent)(**{"vocab_size": len(vocab), **model_cfg}).to(device)
    old_model.load_state_dict(agent.state_dict())
    old_model.eval()

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)

    batch_size = 16
    epochs = 10
    target_update_freq = 5

    print("Start GRPO training...")

    for epoch in range(epochs):
        batch_selfies = []
        batch_sequences = []

        # ===== 生成 batch =====
        for _ in range(batch_size):
            g = generate_selfies(model_name=model_name,
                                 vocab=vocab,
                                 model_path=best_model_path,
                                 device=device,
                                 max_len=max_len,
                                 temperature=temperature,
                                 top_k=top_k)
            s_tokens = list(sf.split_selfies(g["selfies"]))
            if len(s_tokens) == 0:
                continue  # 跳过空序列

            batch_selfies.append(g["selfies"])
            batch_sequences.append([vocab[token] for token in s_tokens])

        if len(batch_selfies) == 0:
            print(f"[Epoch {epoch}] All generated sequences are empty, skipping this epoch.")
            continue

        # ===== 转 SMILES 并批量打分 =====
        batch_smiles = []
        for s in batch_selfies:
            try:
                smi = sf.decoder(s)
                batch_smiles.append(smi)
            except:
                batch_smiles.append(None)

        # 只对有效 SMILES 批量打分
        valid_smiles = [smi for smi in batch_smiles if smi is not None]
        reward_dict = batch_scores_from_vina(
            valid_smiles,
            receptor_file='./feedback/8sc7.pdbqt',
            pdbqt_dir='./feedback/temp/temp_pdbqt',
            output_dir='./feedback/vina_results'
        )

        # ===== 构建 batch_rewards =====
        batch_rewards = []
        for smi in batch_smiles:
            if smi is None:
                batch_rewards.append(-10.0)
            else:
                r = reward_dict.get(smi, {"score": None})
                batch_rewards.append(r["score"] if r["score"] is not None else -10.0)

        # ===== 过滤空序列 =====
        batch_sequences_filtered = []
        batch_rewards_filtered = []

        for seq, r in zip(batch_sequences, batch_rewards):
            # 过滤掉长度 <=1 的序列
            if len(seq) <= 1:
                continue
            batch_sequences_filtered.append(seq)
            batch_rewards_filtered.append(r)

        # 如果没有有效序列就跳过
        if len(batch_sequences_filtered) == 0:
            print(f"[Epoch {epoch}] No valid sequences after filtering, skipping this epoch.")
            continue

        # ===== 打印调试信息 =====
        seq_lengths = [len(seq) for seq in batch_sequences_filtered]
        print(f"[Epoch {epoch}] valid sequences: {len(batch_sequences_filtered)}, "
              f"lengths: {seq_lengths}, mean_reward: {sum(batch_rewards_filtered)/len(batch_rewards_filtered):.3f}")

        # ===== 计算 GRPO Loss =====
        loss, reward_loss, kl_loss = compute_grpo_loss(
            agent, old_model, batch_sequences_filtered, batch_rewards_filtered,
            clip_eps=0.2, kl_beta=0.1
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===== old_model 同步 =====
        if (epoch + 1) % target_update_freq == 0:
            old_model.load_state_dict(agent.state_dict())
        log_file = os.path.join(save_dir, "training_log.csv")

        if epoch == 0 and not os.path.exists(log_file):
            with open(log_file, "w") as f:
                f.write("epoch,total_loss,reward_loss,kl_loss,mean_reward,valid_sequences\n")

        mean_reward = sum(batch_rewards_filtered) / len(batch_rewards_filtered)
        valid_seq_count = len(batch_sequences_filtered)
        print(f"[Epoch {epoch}] total_loss={loss.item():.4f}, reward_loss={reward_loss.item():.4f}, "
              f"kl_loss={kl_loss.item():.4f}, mean_reward={mean_reward:.3f}, valid_seq={valid_seq_count}")

        with open(log_file, "a") as f:
            f.write(f"{epoch},{loss.item()},{reward_loss.item()},{kl_loss.item()},{mean_reward},{valid_seq_count}\n")

        # ===== 保存模型 =====
        torch.save(agent.state_dict(), os.path.join(save_dir, f"{model_name}_grpo_epoch_{epoch}.pt"))


if __name__ == "__main__":
    train_grpo()

