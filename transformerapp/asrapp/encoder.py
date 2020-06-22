import torch
import torch.nn as nn
import torch.nn.functional as F
from asrapp.module import *
from asrapp.utils import get_enc_padding_mask#, get_streaming_mask
from asrapp.attention import MultiHeadedAttention

class TransformerEncoder(nn.Module):

    def __init__(self, input_size, d_model=256, attention_heads=4, linear_units=2048, num_blocks=6, pos_dropout_rate=0.0,
                 slf_attn_dropout_rate=0.0, ffn_dropout_rate=0.0, residual_dropout_rate=0.1, input_layer="conv2d",
                 normalize_before=True, concat_after=False, activation='relu', type='transformer'):
        super(TransformerEncoder, self).__init__()

        self.normalize_before = normalize_before

        if input_layer == "linear":
            self.embed = LinearWithPosEmbedding(input_size, d_model, pos_dropout_rate)
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, d_model, pos_dropout_rate)
        elif input_layer == 'conv2dv2':
            self.embed = Conv2dSubsamplingV2(input_size, d_model, pos_dropout_rate)

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(attention_heads, d_model, linear_units, slf_attn_dropout_rate, ffn_dropout_rate,
                                    residual_dropout_rate=residual_dropout_rate, normalize_before=normalize_before,
                                    concat_after=concat_after, activation=activation) for _ in range(num_blocks)
        ])

        if self.normalize_before:
            self.after_norm = LayerNorm(d_model)

    def forward(self, inputs, input_length, streaming=False):

        enc_mask = get_enc_padding_mask(inputs, input_length)
        
        enc_output, enc_mask = self.embed(inputs, enc_mask)
        enc_output.masked_fill_(~enc_mask.transpose(1, 2), 0.0)
        # if streaming:
        #     length = torch.sum(enc_mask.squeeze(1), dim=-1)
        #     enc_mask = get_streaming_mask(enc_output, length, left_context=20, right_context=0)

        for _, block in enumerate(self.blocks):
            enc_output, enc_mask = block(enc_output, enc_mask)
            enc_output.masked_fill_(~enc_mask.transpose(1, 2), 0.0)

        if self.normalize_before:
            enc_output = self.after_norm(enc_output)

        return enc_output, enc_mask




class TransformerEncoderLayer(nn.Module):
    def __init__(self, attention_heads, d_model, linear_units, slf_attn_dropout_rate, 
                 ffn_dropout_rate, residual_dropout_rate, normalize_before=False,
                 concat_after=False, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(attention_heads, d_model, slf_attn_dropout_rate)
        self.feed_forward = PositionwiseFeedForward(d_model, linear_units, ffn_dropout_rate, activation)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout_rate)
        self.dropout2 = nn.Dropout(residual_dropout_rate)

        self.normalize_before = normalize_before
        self.concat_after = concat_after

        if self.concat_after:
            self.concat_linear = nn.Linear(d_model * 2, d_model)

    def forward(self, x, mask):
        """Compute encoded features

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout1(self.self_attn(x, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout2(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask


