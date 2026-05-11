####
#   该版本为原始的transformer v2基础上加入 rope位置编码，pre_LN，激活函数SWIGLU
###

import torch
import torch.nn as nn
import math
import numpy as np



def get_attn_pad_mask(seq_q, seq_k):
    """
    此函数用于mask掉序列中pad的字符
    :param seq_q: [batch_size, seq_len]
    :param seq_k: [batch_size, seq_len]
    :return:
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k) # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    """
    防止自回归模型看到未来信息
    seq: [batch_size, seq_len]
    """
    batch_size, seq_len = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((seq_len, seq_len), device=seq.device),
        diagonal=1
    ).bool()  # True 表示要 mask
    return subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)


class scaled_dot_product_attention(nn.Module):
    def __init__(self,d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self,q,k,v,attn_mask):
        '''
         k_len = v_len & d_q = d_k
        :param q:  [batch_size,n_heads,q_len,d_q]
        :param k: [batch_size,n_heads,k_len,d_k]
        :param v: [batch_size,n_heads,v_len,d_v]
        :param attn_mask: [batch_size,n_heads,q_len,k_len]
        :return:
        '''
        scores = torch.matmul(q,k.transpose(-1,-2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, float('-inf')) # mask填充一个极大的负数
        attn = nn.Softmax(dim = -1)(scores)
        context = torch.matmul(attn,v)
        return context,attn



class multi_head_attention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v,max_len, device):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_v, bias=False)

        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.ln = nn.LayerNorm(d_model)
        self.rope = rope(d_k,max_len)

    def forward(self, x, attn_mask):
        """
        x: [B, L, D]
        """
        residual = x
        x = self.ln(x)

        B, L, _ = x.size()
        q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_v).transpose(1, 2)

        q, k = self.rope(q, k)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(attn_mask.unsqueeze(1), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.fc(context)

        return out + residual, attn



class swiglu(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = self.w3(torch.nn.functional.silu(self.w1(x)) * self.w2(x))
        return x + residual



def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q, k, sin, cos):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class rope(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :])
        self.register_buffer("sin", emb.sin()[None, None, :, :])

    def forward(self, q, k):
        seq_len = q.size(-2)
        cos = self.cos[..., :seq_len, :]
        sin = self.sin[..., :seq_len, :]
        return apply_rope(q, k, sin, cos)


class decoderlayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff,max_len, device):
        super().__init__()
        self.self_attn = multi_head_attention(
            d_model, n_heads, d_k, d_v, max_len,device
        )
        self.ffn = swiglu(d_model, d_ff)

    def forward(self, x, attn_mask):
        x, attn = self.self_attn(x, attn_mask)
        x = self.ffn(x)
        return x, attn



class decoder_only_lm(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        d_model,
        n_heads,
        d_k,
        d_v,
        d_ff,
        n_layers,
        device
    ):
        super().__init__()
        self.device = device

        self.token_emb = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            decoderlayer(d_model, n_heads, d_k, d_v, d_ff, max_len,device)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        """
        input_ids: [batch_size, seq_len]
        """
        x = self.token_emb(input_ids)

        pad_mask = get_attn_pad_mask(input_ids, input_ids)
        causal_mask = get_attn_subsequence_mask(input_ids)
        attn_mask = (pad_mask | causal_mask).to(input_ids.device)

        for layer in self.layers:
            x, _ = layer(x, attn_mask)

        x = self.norm(x)
        logits = self.lm_head(x)  # [B, L, vocab_size]
        return logits












