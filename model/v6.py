import torch
import torch.nn as nn
import math


class RoPE(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def forward(self, q, k):
        L = q.size(-2)
        cos = self.cos[:, :, :L, :]
        sin = self.sin[:, :, :L, :]
        return (
            q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin
        )

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.ln = nn.LayerNorm(d_model)  # Pre-LN
        self.dropout = nn.Dropout(dropout)

        self.rope = RoPE(self.d_head, max_len)

    def forward(self, x, attn_mask):
        """
        x: [B, T, D]
        attn_mask: [T, T] (causal)
        """
        residual = x
        x = self.ln(x)

        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)  # [B, H, T, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.rope(q, k)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        scores = scores + attn_mask  # attn_mask 是 -inf 的上三角
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        return out + residual


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(self.ln(x))

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, max_len, dropout)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x, attn_mask):
        x = self.attn(x, attn_mask)
        x = self.ffn(x)
        return x


class decoder_only_lm(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=3, max_len=80, dropout=0.2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, max_len, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: [B, T]
        """
        B, T = x.shape
        x = self.token_emb(x)

        # causal mask: [T, T]
        attn_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device),
            diagonal=1
        )

        for layer in self.layers:
            x = layer(x, attn_mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
