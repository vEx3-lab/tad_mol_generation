import torch
import torch.nn as nn
import math

class rope(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def rotate_half(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def apply_rope(self, q, k):
        seq_len = q.size(-2)
        cos = self.cos[..., :seq_len, :]
        sin = self.sin[..., :seq_len, :]
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        return q, k

    def forward(self, q, k):
        return self.apply_rope(q, k)


class decoder_only_lm(nn.Module):
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

        # 初始化权重
        self._init_weights()

        # RoPE
        self.rope = rope(d_model // n_heads, max_len)
        self.n_heads = n_heads

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

        # causal mask: 上三角 -inf
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)

        # --------------- 应用 RoPE ----------------
        # 将 x reshape 为多头形式 [B, L, H, d_k] -> [B*H, L, d_k]
        d_model = x.size(-1)
        d_k = d_model // self.n_heads
        x_ = x.view(B, T, self.n_heads, d_k).transpose(1, 2)  # [B, H, L, d_k]
        q = k = x_
        q, k = self.rope(q, k)
        # 回到原形状
        x = q.transpose(1, 2).contiguous().view(B, T, d_model)
        # ------------------------------------------

        for layer in self.layers:
            x = layer(x, src_mask=mask)

        x = self.norm(x)
        logits = self.fc_out(x)
        logits = torch.clamp(logits, min=-1e2, max=1e2)
        return logits
