# -*- coding: utf-8 -*-


import sys
import math
import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

torch.manual_seed(1)


def clones(module, N):

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)    
    
class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def subsequent_mask(size):

    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):

        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):

        if mask is not None:

            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        

        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        

        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        

        pe = torch.zeros(max_len, d_model)
        
        
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x


class MMSeq2SeqModel(nn.Module):

    def __init__(self, mm_encoder, history_encoder, input_encoder, response_decoder):

        super(MMSeq2SeqModel, self).__init__()
        self.history_encoder = history_encoder
        self.mm_encoder = mm_encoder
        self.input_encoder = input_encoder
        self.response_decoder = response_decoder

        self.vfr_lt = nn.Linear(256, 128)
        self.vfl_lt = nn.Linear(256, 128)
        self.a_lt = nn.Linear(256, 128)
        
        encoder_layer = EncoderLayer(size = 128, self_attn = MultiHeadedAttention(h = 8, d_model = 128), feed_forward = PositionwiseFeedForward(d_model = 128, d_ff = 128, dropout = 0.1), dropout = 0.1)
        self.encoder = Encoder(encoder_layer, 1)

    def loss(self, mx, hx, x, y, t):


        ei = self.input_encoder(None, x)
        ev_frame, ev_flow, ev_audio = self.mm_encoder(ei, mx)
        eh = self.history_encoder(None, hx)

        eh = eh[-1]
        evfr = self.vfr_lt(ev_frame)
        evfl = self.vfl_lt(ev_flow)
        eva = self.a_lt(ev_audio)

        transformer_input = torch.cat((eh.unsqueeze(1), evfr.unsqueeze(1), evfl.unsqueeze(1), eva.unsqueeze(1)), dim =1)

        transformer_output = self.encoder(transformer_input, None)
        es = torch.cat((transformer_output, ei.unsqueeze(1)), dim =1)



        es = es.view(es.size(0),es.size(1)*es.size(2))       
        if hasattr(self.response_decoder, 'context_to_state') \
            and self.response_decoder.context_to_state==True:
            ds, dy = self.response_decoder(es, None, y)
        else:
            ds, dy = self.response_decoder(None, es, y)

        if t is not None:
            tt = torch.cat(t, dim=0)
            loss = F.cross_entropy(dy, torch.tensor(tt, dtype=torch.long).cuda())
            max_index = dy.max(dim=1)[1]
            hit = (max_index == torch.tensor(tt, dtype=torch.long).cuda()).sum()
            return None, ds, loss
        else:
            return None, ds

    def generate(self, mx, hx, x, sos=2, eos=2, unk=0, minlen=1, maxlen=100, beam=5, penalty=1.0, nbest=1):

        ei = self.input_encoder(None, x)
        ev_frame, ev_flow, ev_audio = self.mm_encoder(ei, mx)
        eh = self.history_encoder(None, hx)

        eh = eh[-1]
        evfr = self.vfr_lt(ev_frame)
        evfl = self.vfl_lt(ev_flow)
        eva = self.a_lt(ev_audio)

        transformer_input = torch.cat((eh.unsqueeze(1), evfr.unsqueeze(1), evfl.unsqueeze(1), eva.unsqueeze(1)), dim =1)

        transformer_output = self.encoder(transformer_input, None)
        es = torch.cat((transformer_output, ei.unsqueeze(1)), dim =1)
        es = es.view(es.size(0),es.size(1)*es.size(2))
        

        ds = self.response_decoder.initialize(None, es, torch.from_numpy(np.asarray([sos])).cuda())
        hyplist = [([], 0., ds)]
        best_state = None
        comp_hyplist = []
        for l in six.moves.range(maxlen):
            new_hyplist = []
            argmin = 0
            for out, lp, st in hyplist:
                logp = self.response_decoder.predict(st)
                lp_vec = logp.cpu().data.numpy() + lp
                lp_vec = np.squeeze(lp_vec)
                if l >= minlen:
                    new_lp = lp_vec[eos] + penalty * (len(out) + 1)
                    new_st = self.response_decoder.update(st, torch.from_numpy(np.asarray([eos])).cuda())
                    comp_hyplist.append((out, new_lp))
                    if best_state is None or best_state[0] < new_lp:
                        best_state = (new_lp, new_st)

                for o in np.argsort(lp_vec)[::-1]:
                    if o == unk or o == eos:
                        continue
                    new_lp = lp_vec[o]
                    if len(new_hyplist) == beam:
                        if new_hyplist[argmin][1] < new_lp:
                            new_st = self.response_decoder.update(st, torch.from_numpy(np.asarray([o])).cuda())
                            new_hyplist[argmin] = (out + [o], new_lp, new_st)
                            argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                        else:
                            break
                    else:
                        new_st = self.response_decoder.update(st, torch.from_numpy(np.asarray([o])).cuda())
                        new_hyplist.append((out + [o], new_lp, new_st))
                        if len(new_hyplist) == beam:
                            argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]

            hyplist = new_hyplist

        if len(comp_hyplist) > 0:
            maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:nbest]
            return maxhyps, best_state[1]
        else:
            return [([], 0)], None

