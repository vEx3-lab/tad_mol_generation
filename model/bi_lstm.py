import torch.nn as nn
class bi_lstm(nn.Module):
    """
    BiLSTM 生成模型（训练/生成一致：只用 forward hidden）
    """
    def __init__(
        self,
        vocab_size,
        embed_dims=64,
        hidden_dims=128,
        n_layers=2,
        dropout=0.2,
        max_len=80
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dims = hidden_dims
        self.max_len = max_len

        # Embedding
        self.token_emb = nn.Embedding(vocab_size, embed_dims)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embed_dims,
            hidden_size=hidden_dims,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # LayerNorm + 输出
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-5)  # 只用 forward hidden
        self.fc_out = nn.Linear(hidden_dims, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, x):
        """
        x: [B, T]
        return: logits [B, T, vocab_size] (只用 forward hidden)
        """
        emb = self.token_emb(x)           # [B, T, E]
        lstm_out, _ = self.lstm(emb)     # [B, T, 2H]
        # 只取 forward hidden
        lstm_out = lstm_out[:, :, :self.hidden_dims]
        out = self.norm(lstm_out)
        logits = self.fc_out(out)         # [B, T, V]
        return logits
