from math import ceil

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# torch.fx.wrap('len')
# torch.fx.wrap('rearrange')

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, l=3):
    val = val if isinstance(val, tuple) else (val,)
    
    # Manually count the elements in the tuple
    count = 0
    for _ in val:
        count += 1
    
    # Use count instead of len(val)
    return (*val, *((val[-1],) * max(l - count, 0)))

def always(val):
    return lambda *args, **kwargs: val

# classes

# Part of the transformer
class FeedForward(nn.Module):
    def __init__(self, dim, mult, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.Hardswish(),  ## could be a problem
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, fmap_size, heads = 8, dim_key = 32, dim_value = 64, dropout = 0., dim_out = None, downsample = False):
        super().__init__()
        inner_dim_key = dim_key *  heads
        inner_dim_value = dim_value *  heads
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_key ** -0.5

        self.to_q = nn.Sequential(nn.Conv2d(dim, inner_dim_key, 1, stride = (2 if downsample else 1), bias = False), nn.BatchNorm2d(inner_dim_key))
        self.to_k = nn.Sequential(nn.Conv2d(dim, inner_dim_key, 1, bias = False), nn.BatchNorm2d(inner_dim_key))
        self.to_v = nn.Sequential(nn.Conv2d(dim, inner_dim_value, 1, bias = False), nn.BatchNorm2d(inner_dim_value))

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        out_batch_norm = nn.BatchNorm2d(dim_out)
        nn.init.zeros_(out_batch_norm.weight)

        self.to_out = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inner_dim_value, dim_out, 1),
            out_batch_norm,
            nn.Dropout(dropout)
        )

        # positional bias

        self.pos_bias = nn.Embedding(fmap_size * fmap_size, heads)

        q_range = torch.arange(0, fmap_size, step = (2 if downsample else 1))
        k_range = torch.arange(fmap_size)

        q_pos = torch.stack(torch.meshgrid(q_range, q_range, indexing = 'ij'), dim = -1)
        k_pos = torch.stack(torch.meshgrid(k_range, k_range, indexing = 'ij'), dim = -1)

        # q_pos, k_pos = map(lambda t: rearrange(t, 'i j c -> (i j) c'), (q_pos, k_pos))
        q_pos = q_pos.view(-1, q_pos.shape[-1])
        k_pos = k_pos.view(-1, k_pos.shape[-1])

        rel_pos = (q_pos[:, None, ...] - k_pos[None, :, ...]).abs()

        x_rel, y_rel = rel_pos.unbind(dim = -1)
        # print("x_rel shape",x_rel.shape)
        # print("fmap_size",fmap_size)

        pos_indices = (x_rel * fmap_size) + y_rel

        self.register_buffer('pos_indices', pos_indices)

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        # bias = rearrange(bias, 'i j h -> () h i j')
        bias = bias.unsqueeze(0).permute(0,3,1,2)

        # print("fmap shape",fmap.shape)
        # print("bias shape",bias.shape)
        # print("pos_indices shape",self.pos_indices.shape)
        # print("scale",self.scale)

        # exit()

        return fmap + (bias / self.scale)

    def forward(self, x):
        # This makes fx untracable
        # b, n, *_, h = *x.shape, self.heads

        # This makes fx tracable
        # print(x.shape)
        # [batch_size, channels, height, width]
        b, n, h, w = x.shape[0] , x.shape[1], x.shape[2], x.shape[3]   # Use direct assignment for shape dimensions
        h = self.heads
        d = n // h  # dimension per head
        # print(b, n, h, w, d)

        q = self.to_q(x)
        b_q, c_q, h_q, w_q = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
        d_q = c_q // h
        # print(b_q, c_q, h_q, w_q, d_q)

        # print(q.shape)
        k = self.to_k(x)
        v = self.to_v(x)
        y = q.shape[2]

        # einsum
        q = rearrange(q, 'b (h d) ... -> b h (...) d', h = h)
        # print(q.shape)

        # no einsum pytorch implementation
        # q = q.reshape(b, h, h_q*w_q ,d_q)
        # print(q.shape)

        # should become [1, 2, 400, 32]
        # exit()
        k = rearrange(k, 'b (h d) ... -> b h (...) d', h = h)
        v = rearrange(v, 'b (h d) ... -> b h (...) d', h = h)

        # q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = h), (q,k,v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = self.apply_pos_bias(dots)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult = 2, dropout = 0., dim_out = None, downsample = False):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.layers = nn.ModuleList([])
        self.attn_residual = (not downsample) and dim == dim_out

        # print("dim",dim)
        # print("fmap_size",fmap_size)
        # print("depth",depth)
        # print("heads",heads)
        # print("dim_key",dim_key)
        # print("dim_value",dim_value)
        # print("mlp_mult",mlp_mult)


        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, fmap_size = fmap_size, heads = heads, dim_key = dim_key, dim_value = dim_value, dropout = dropout, downsample = downsample, dim_out = dim_out),
                FeedForward(dim_out, mlp_mult, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            attn_res = (x if self.attn_residual else 0)
            x = attn(x) + attn_res
            x = ff(x) + x
        return x

class LeViTJim(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        n_convs,
        dim,
        depth,
        heads,
        mlp_mult,
        stages = 3,
        dim_key = 32,
        dim_value = 64,
        dropout = 0.,
        num_distill_classes = None,
        input_channels = 1,
        base_channel = 32,
    ):
        super().__init__()

        dims = cast_tuple(dim, stages)
        depths = cast_tuple(depth, stages)
        layer_heads = cast_tuple(heads, stages)

        assert all(map(lambda t: len(t) == stages, (dims, depths, layer_heads))), 'dimensions, depths, and heads must be a tuple that is less than the designated number of stages'

        # Initialize the convolutional embedding
        channel_sizes = [base_channel * 2 ** i for i in range(n_convs)]
        # print(channel_sizes)
        channel_sizes.insert(0, input_channels)  # Assuming the input has 1 channel
        channel_sizes[-1] = dims[0]  # Last channel size must be equal to the first dimension of the transformer
        conv_layers = [ 
            nn.Conv2d(in_channels=channel_sizes[i], out_channels=channel_sizes[i + 1], kernel_size=3, stride=2, padding=1)        
            for i in range(len(channel_sizes) - 1)
            ]
        
        self.conv_embedding = nn.Sequential(*conv_layers)

        fmap_size = image_size // (2 ** n_convs)
        # print('fmap_size',fmap_size)
        layers = []

        for ind, dim, depth, heads in zip(range(stages), dims, depths, layer_heads):
            is_last = ind == (stages - 1)
            layers.append(Transformer(dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult, dropout))

            if not is_last:
                next_dim = dims[ind + 1]
                layers.append(Transformer(dim, fmap_size, 1, heads * 2, dim_key, dim_value, dim_out = next_dim, downsample = True))
                fmap_size = ceil(fmap_size / 2)

        self.backbone = nn.Sequential(*layers)

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # Rearrange('... () () -> ...'),
            nn.Flatten(start_dim=1)# Rearrange('... () () -> ...')   
        )

        self.distill_head = nn.Linear(dim, num_distill_classes) if exists(num_distill_classes) else always(None)
        self.mlp_head = nn.Linear(dim, num_classes)

        # self.quant = torch.quantization.QuantStub()
        # self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # print(img.shape)
        # x = self.quant(x)
        x = self.conv_embedding(x)
        # print(x.shape)

        # exit()
        # x = self.dequant(x)
        # print(x.shape)
        x = self.backbone(x)        
        # print(x.shape)
        x = self.pool(x)

        out = self.mlp_head(x)
        distill = self.distill_head(x)

        if exists(distill):
            return out, distill

        # out = self.dequant(out)

        return out