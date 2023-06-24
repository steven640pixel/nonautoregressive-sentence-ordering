import math
import torch
import torch.nn as nn
from .neural import MultiHeadedAttention, PositionwiseFeedForward
from .neural import MultiHeadedAttentionDe, PositionwiseFeedForwardDe, clone, SublayerConnection
from .neural import positional_encodings_like, ResidualBlock, PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerInterEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(TransformerInterEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        # pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        # x = x + pos_emb
        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, 1 - mask)
        x = self.layer_norm(x)

        return x

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        #self.norm = LayerNorm(layer.size)
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward"

    def __init__(self, d_model, n_heads, d_hidden, dropout=0.1, positional=None):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttentionDe(n_heads, d_model, dropout)
        self.positional = positional
        self.src_attn = MultiHeadedAttentionDe(n_heads, d_model, dropout)
        self.pos = PositionalEncoding(d_model, dropout, max_len=5000)
        if positional:
            self.pos_selfattn = ResidualBlock(
                MultiHeadedAttentionDe(n_heads, d_model, dropout),
                d_model, dropout, pos=2)
        self.feed_forward = PositionwiseFeedForwardDe(d_model, d_hidden, dropout)
        self.sublayer = clone(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        if self.positional:
            pos_encoding = positional_encodings_like(x)

        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))

        return self.sublayer[2](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward"

    def __init__(self, d_model, n_heads, d_hidden, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttentionDe(n_heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForwardDe(d_model, d_hidden, dropout)

        self.sublayer = clone(SublayerConnection(d_model, dropout), 2)
        self.size = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
