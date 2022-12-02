## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention_3(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_3, self).__init__()
        self.num_heads = num_heads
        self.tgt_num = 1
        self.ref_num = 3
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_conv = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                      nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
                                      ])

        self.k_conv = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                      nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
                                      ])

        self.v_conv = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                      nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
                                      ])

        self.project_out =  nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,h,w = x[0].shape

        q_input = x[0] # _1_feat, shape is [b, c, h, w]
        kv_input = torch.stack(x, dim=1) # stack [_1_feat, _2_feat, _3_feat], shape is [b, ref_num, c, h, w]
        q = self.q_conv(q_input).unsqueeze(1) # shape is [b, 1, c, h, w]
        k = self.k_conv(kv_input.view(-1, *kv_input.shape[2:])).view(kv_input.shape) # shape is [b, ref_num, c, h, w]
        v = self.v_conv(kv_input.view(-1, *kv_input.shape[2:])).view(kv_input.shape) # shape is [b, ref_num, c, h, w]

        q = q.view(b, self.tgt_num, c, h, w).contiguous()
        k = k.view(b, self.ref_num, c, h, w).contiguous()
        v = v.view(b, self.ref_num, c, h, w).contiguous()
        
        q = rearrange(q, 'b num c h w -> b (h w) num c', num=self.tgt_num, h=h, w=w)
        k = rearrange(k, 'b num c h w -> b (h w) num c', num=self.ref_num, h=h, w=w)
        v = rearrange(v, 'b num c h w -> b (h w) num c', num=self.ref_num, h=h, w=w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = rearrange(out, 'b (h w) num c -> b (num c) h w', h=h, w=w, num=self.tgt_num)

        out = self.project_out(out)
        return out


class Attention_4(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_4, self).__init__()
        self.num_heads = num_heads
        self.tgt_num = 1
        self.ref_num = 4
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_conv = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                      nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
                                      ])

        self.k_conv = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                      nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
                                      ])

        self.v_conv = nn.Sequential(*[nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
                                      nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
                                      ])

        self.project_out =  nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,h,w = x[0].shape

        q_input = x[0] # 2_feat, shape is [b, c, h, w]
        kv_input = torch.stack(x, dim=1) # stack [_1_feat, _2_feat, _3_feat], shape is [b, ref_num, c, h, w]
        q = self.q_conv(q_input).unsqueeze(1) # shape is [b, 1, c, h, w]
        k = self.k_conv(kv_input.view(-1, *kv_input.shape[2:])).view(kv_input.shape) # shape is [b, ref_num, c, h, w]
        v = self.v_conv(kv_input.view(-1, *kv_input.shape[2:])).view(kv_input.shape) # shape is [b, ref_num, c, h, w]

        q = q.view(b, self.tgt_num, c, h, w).contiguous()
        k = k.view(b, self.ref_num, c, h, w).contiguous()
        v = v.view(b, self.ref_num, c, h, w).contiguous()
        
        q = rearrange(q, 'b num c h w -> b (h w) num c', num=self.tgt_num, h=h, w=w)
        k = rearrange(k, 'b num c h w -> b (h w) num c', num=self.ref_num, h=h, w=w)
        v = rearrange(v, 'b num c h w -> b (h w) num c', num=self.ref_num, h=h, w=w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = rearrange(out, 'b (h w) num c -> b (num c) h w', h=h, w=w, num=self.tgt_num)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock_3(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_3, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_3(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        [_0_feat, _1_feat, _2_feat] = x
        _0_feat = _0_feat + self.attn([self.norm1(_0_feat), self.norm1(_1_feat), self.norm1(_2_feat)])
        _0_feat = _0_feat + self.ffn(self.norm2(_0_feat))

        return _0_feat


class TransformerBlock_4(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_4, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_4(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        [_0_feat, _1_feat, _2_feat, _3_feat] = x
        _0_feat = _0_feat + self.attn([self.norm1(_0_feat), self.norm1(_1_feat), self.norm1(_2_feat), self.norm1(_3_feat)])
        _0_feat = _0_feat + self.ffn(self.norm2(_0_feat))

        return _0_feat



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

