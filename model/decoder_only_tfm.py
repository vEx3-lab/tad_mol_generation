# ===== Stable DecoderOnlyTransformer =====
import torch
import torch.nn as nn
class decoder_only_tfm(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=2, max_len=80, dropout=0.2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4*d_model,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

        #  初始化权重
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
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(pos)

        # causal mask: 上三角 -inf，防止未来信息泄露
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)

        for layer in self.layers:
            x = layer(x, src_mask=mask)

        x = self.norm(x)
        logits = self.fc_out(x)
        # clamp logits 防止 NaN / Inf
        logits = torch.clamp(logits, min=-1e2, max=1e2)
        return logits