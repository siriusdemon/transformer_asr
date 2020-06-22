import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from asrapp.utils import get_seq_mask, get_dec_seq_mask
from asrapp.module import LayerNorm, PositionalEncoding, PositionwiseFeedForward
from asrapp.attention import MultiHeadedAttention
#from otrans.layer import TransformerDecoderLayer, TransformerEncoderLayer


class TransformerDecoder(nn.Module):
    def __init__(self, output_size, d_model=256, attention_heads=4, linear_units=2048, num_blocks=6, pos_dropout_rate=0.0, 
                 slf_attn_dropout_rate=0.0, src_attn_dropout_rate=0.0, ffn_dropout_rate=0.0, residual_dropout_rate=0.1,
                 activation='relu', normalize_before=True, concat_after=False, share_embedding=False):
        super(TransformerDecoder, self).__init__()

        self.normalize_before = normalize_before

        self.embedding = torch.nn.Embedding(output_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, pos_dropout_rate)

        self.blocks = nn.ModuleList([
            TransformerDecoderLayer(attention_heads, d_model, linear_units, slf_attn_dropout_rate, src_attn_dropout_rate,
                                    ffn_dropout_rate, residual_dropout_rate, normalize_before=normalize_before, concat_after=concat_after,
                                    activation=activation) for _ in range(num_blocks)
        ])

        if self.normalize_before:
            self.after_norm = LayerNorm(d_model)

        self.output_layer = nn.Linear(d_model, output_size)

        if share_embedding:
            assert self.embedding.weight.size() == self.output_layer.weight.size()
            self.output_layer.weight = self.embedding.weight

    def forward(self, targets, target_length, memory, memory_mask):

        dec_output = self.embedding(targets)
        dec_output = self.pos_encoding(dec_output)
        dec_mask = get_dec_seq_mask(targets, target_length)
        for _, block in enumerate(self.blocks):
            dec_output, dec_mask = block(dec_output, dec_mask, memory, memory_mask)

        if self.normalize_before:
            dec_output = self.after_norm(dec_output)
        logits = self.output_layer(dec_output)
        return logits, dec_mask

    def recognize(self, preds, memory, memory_mask, last=True):

        dec_output = self.embedding(preds)
        dec_mask = get_seq_mask(preds)

        for _, block in enumerate(self.blocks):
            dec_output, dec_mask = block(dec_output, dec_mask, memory, memory_mask)

        if self.normalize_before:
            dec_output = self.after_norm(dec_output)

        logits = self.output_layer(dec_output)

        log_probs = F.log_softmax(logits[:, -1] if last else logits, dim=-1)

        return log_probs




class TransformerDecoderLayer(nn.Module):

    def __init__(self, attention_heads, d_model, linear_units, slf_attn_dropout_rate, src_attn_dropout_rate, 
                 ffn_dropout_rate, residual_dropout_rate, normalize_before=True, concat_after=False, activation='relu'):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(attention_heads, d_model, slf_attn_dropout_rate)
        self.src_attn = MultiHeadedAttention(attention_heads, d_model, src_attn_dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, linear_units, ffn_dropout_rate, activation)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout_rate)
        self.dropout2 = nn.Dropout(residual_dropout_rate)
        self.dropout3 = nn.Dropout(residual_dropout_rate)

        self.normalize_before = normalize_before
        self.concat_after = concat_after

        if self.concat_after:
            self.concat_linear1 = nn.Linear(d_model * 2, d_model)
            self.concat_linear2 = nn.Linear(d_model * 2, d_model)

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Compute decoded features

        :param torch.Tensor tgt: decoded previous target features (batch, max_time_out, size)
        :param torch.Tensor tgt_mask: mask for x (batch, max_time_out)
        :param torch.Tensor memory: encoded source features (batch, max_time_in, size)
        :param torch.Tensor memory_mask: mask for memory (batch, max_time_in)
        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)
        if self.concat_after:
            tgt_concat = torch.cat((tgt, self.self_attn(tgt, tgt, tgt, tgt_mask)), dim=-1)
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout1(self.self_attn(tgt, tgt, tgt, tgt_mask))

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat((x, self.src_attn(x, memory, memory, memory_mask)), dim=-1)
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout2(self.src_attn(x, memory, memory, memory_mask))

        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout3(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)
        return x, tgt_mask


