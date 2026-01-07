import pandas as pd
import matplotlib.pyplot as plt

# 读取 loss 文件
df = pd.read_csv("../model/decoder_only_tfm/finetune_loss_1.csv")

# 假设 loss_fold_1.csv 中包含两列：train_loss 和 val_loss
epochs = range(1, len(df) + 1)

# 绘制训练集与验证集损失
plt.figure(figsize=(8, 6))
plt.plot(epochs, df["train_loss"], label="Train Loss", linewidth=2)
# plt.plot(epochs, df["val_loss"], label="Validation Loss", linewidth=2, linestyle='--')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存与展示
plt.savefig("../plot/fine_loss_curve.png", dpi=300)
plt.show()
#
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # ======= 读取两个模型的 loss 文件 =======
# df_tfm = pd.read_csv("../model/decoder_only_tfm/loss_fold_1.csv")
# df_lstm = pd.read_csv("../model/bi_lstm/loss_fold_1.csv")
#
# epochs_tfm = range(1, len(df_tfm) + 1)
# epochs_lstm = range(1, len(df_lstm) + 1)
#
# plt.figure(figsize=(9, 7))
#
# # ======== Transformer Loss ========
# plt.plot(epochs_tfm, df_tfm["train_loss"],
#          label="TFM Train Loss", linewidth=2, color="C0")
# plt.plot(epochs_tfm, df_tfm["val_loss"],
#          label="TFM Val Loss", linewidth=2, linestyle='--', color="C0")
#
# # ======== Bi-LSTM Loss ========
# plt.plot(epochs_lstm, df_lstm["train_loss"],
#          label="Bi-LSTM Train Loss", linewidth=2, color="C1")
# plt.plot(epochs_lstm, df_lstm["val_loss"],
#          label="Bi-LSTM Val Loss", linewidth=2, linestyle='--', color="C1")
#
# # ======== 图形设置 ========
# plt.xlabel("Epoch", fontsize=12)
# plt.ylabel("Loss", fontsize=12)
# plt.title("TFM vs Bi-LSTM Training & Validation Loss", fontsize=14)
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
#
# # 保存
# plt.savefig("../plot/tfm_vs_lstm_loss_curve.png", dpi=300)
#
# plt.show()
