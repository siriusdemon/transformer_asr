
import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        #print(51,scores[0][0])
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            #print(51,scores.shape,mask.shape,q.shape,k.shape)
            scores = scores.masked_fill(mask, min_value)
#            print(52,scores.shape,mask)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)



# class MultiHeadAttention(nn.Module):
#     ''' Multi-Head Attention module '''

#     def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
#         super().__init__()

#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v

#         self.w_qs = nn.Linear(d_model, n_head * d_k)
#         self.w_ks = nn.Linear(d_model, n_head * d_k)
#         self.w_vs = nn.Linear(d_model, n_head * d_v)
#         nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
#         nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
#         nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

#         self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
#                                                    attn_dropout=dropout)
#         self.layer_norm = nn.LayerNorm(d_model)

#         self.fc = nn.Linear(n_head * d_v, d_model)
#         nn.init.xavier_normal_(self.fc.weight)

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, q, k, v, mask=None):
#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

#         sz_b, len_q, _ = q.size()
#         sz_b, len_k, _ = k.size()
#         sz_b, len_v, _ = v.size()

#         residual = q

#         q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
#         k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
#         v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

#         q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
#         k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
#         v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

#         if mask is not None:
#             mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

#         output, attn = self.attention(q, k, v, mask=mask)

#         output = output.view(n_head, sz_b, len_q, d_v)
#         output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

#         output = self.dropout(self.fc(output))
#         output = self.layer_norm(output + residual)

#         return output, attn

