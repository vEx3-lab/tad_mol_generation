import pandas as pd
import matplotlib.pyplot as plt

# 读取 loss 文件
df = pd.read_csv("../decoder_only_tfm/loss_fold_1.csv")

# 假设 loss_fold_1.csv 中包含两列：train_loss 和 val_loss
epochs = range(1, len(df) + 1)

# 绘制训练集与验证集损失
plt.figure(figsize=(8, 6))
plt.plot(epochs, df["train_loss"], label="Train Loss", linewidth=2)
plt.plot(epochs, df["val_loss"], label="Validation Loss", linewidth=2, linestyle='--')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存与展示
plt.savefig("./loss_curve.png", dpi=300)
plt.show()
