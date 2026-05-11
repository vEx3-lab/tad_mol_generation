####
#   该版本为原始的transformer配置
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
    def __init__(self,d_model,n_heads,d_k,d_v,device):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.device = device

        self.w_q = nn.Linear(self.d_model,self.n_heads * self.d_k , bias= False)
        self.w_k = nn.Linear(self.d_model,self.n_heads * self.d_k, bias= False)
        self.w_v = nn.Linear(self.d_model,self.n_heads * self.d_v,bias= False)

        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model , bias= False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self,input_q,input_k,input_v,attn_mask):
        '''
        k_len =  v_len
        :param input_q: [batch_size,nq_len,d_model]
        :param input_k: [batch_size,k_len,d_model]
        :param input_v: [batch_size,v_len,d_model]
        :param attn_mask: [batch_size,q_len,k_len]
        :return:
        '''
        x , batch_size = input_q ,input_q.size(0)
        q = self.w_q(input_q).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2) # [batch_size,q_len,n_heads,d_k]
        k = self.w_k(input_k).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2) # [batch_size,q_len,n_heads,d_k]
        v = self.w_v(input_v).view(batch_size,-1,self.n_heads,self.d_v).transpose(1,2) # [batch_size,q_len,n_heads,d_v]

        attn_mask =attn_mask.unsqueeze(1) # [batch_size, 1,q_len,v_len]

        context,attn = scaled_dot_product_attention(self.d_k)(q,k,v,attn_mask)
        context =context.transpose(1,2).reshape(batch_size,-1,self.n_heads * self.d_v)

        output =self.fc(context)
        return self.layer_norm(output + x) ,attn


class pos_wise_feedforward_net(nn.Module):
    def __init__(self,d_model,d_ff,device):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(d_model,d_ff,bias= False),
            nn.ReLU(),
            nn.Linear(d_ff,d_model,bias= False)
        )
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self,inputs):
        '''

        :param inputs:  [batch_size,seq_len,d_model]
        :return:
        '''
        x = inputs
        output = self.fc(x)
        return self.layer_norm(x + output)


class positional_encoding(nn.Module):
    def __init__(self,d_model, device, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.pe = torch.zeros(max_len,d_model).to(device)
        position = torch.arange(0,max_len,dtype= torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:,0::2] = torch.sin(position * div_term)
        self.pe[:,1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(1) # [seq_len,1,d_model]

    def forward(self,x ):
        '''

        :param x: [seq_len,batch_size,d_model]
        :return:
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class decoderlayer(nn.Module):
    '''
    decoder only transformer

    '''
    def __init__(self,d_model,n_heads,d_k,d_v,d_ff,device):
        super().__init__()
        self.dec_self_attn = multi_head_attention(d_model,n_heads,d_k,d_v,device)
        self.pos_ffn =pos_wise_feedforward_net(d_model,d_ff,device)

    def forward(self,dec_inputs,dec_self_attn_mask):
        '''

        :param dec_inputs: [batch_size,seq_len,d_model]
        :param dec_self_attn_mask: [batch_size, seq_len, seq_len]
        :return: dec_outputs:[batch_size, seq_len, d_model], dec_self_attn :[batch_size, n_heads, seq_len, seq_len]
        '''
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs , dec_self_attn



class decoder(nn.Module):
    def __init__(self,vocab_size,seq_len,d_model,n_heads,d_k,d_v,d_ff,nums_layers,drop,device):
        super().__init__()
        self.seq_len = seq_len
        self.device = device
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = positional_encoding(d_model, device, drop)
        self.layers = nn.ModuleList([decoderlayer(d_model, n_heads, d_k, d_v, d_ff, device) for _ in range(nums_layers)])


    def forward(self,dec_inputs):
        '''

        :param dec_inputs:  [batch_size,seq_len]
        :return:
        '''
        dec_outputs = self.token_emb(dec_inputs) # [batch_size,seq_len,d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0,1).to(self.device)
        # mask
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(self.device)
        dec_self_attn_sub_mask = get_attn_subsequence_mask(dec_inputs).to(self.device)
        dec_self_attn_mask = (dec_self_attn_pad_mask | dec_self_attn_sub_mask).to(self.device)

        dec_self_attns = []

        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(
                dec_outputs,
                dec_self_attn_mask
            )
            dec_self_attns.append(dec_self_attn)

        return dec_outputs, dec_self_attns
















