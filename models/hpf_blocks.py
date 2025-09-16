
"""
* @name: hpf_blocks.py
* @description: Functions of ProjMLP / MonoCNN / interleave / resize_to_length / CoMambaPair, et al。
"""

from typing import Literal
import torch
from torch import nn, einsum
from einops import rearrange, repeat


__all__ = [
    "Transformer",
    "CrossTransformer",
    "ProjMLP",
    "MonoCNN",
    "interleave",
    "concat_time",
    "resize_to_length",
    "CoMambaPair",
]



@torch.no_grad()
def resize_to_length(
    x: torch.Tensor,
    target_len: int,
    mode: Literal["truncate", "pad", "pool", "linear"] = "pool",
    pad_value: float = 0.0,
) -> torch.Tensor:

    B, L0, D = x.shape
    if L0 == target_len:
        return x

    if mode == "pool":
        x_ = rearrange(x, "b l d -> b d l")
        y_ = nn.functional.adaptive_avg_pool1d(x_, target_len)
        return rearrange(y_, "b d l -> b l d")

    if mode == "linear":
        x_ = rearrange(x, "b l d -> b d l")
        y_ = nn.functional.interpolate(x_, size=target_len, mode="linear", align_corners=False)
        return rearrange(y_, "b d l -> b l d")

    if mode == "truncate":
        if L0 >= target_len:
            return x[:, :target_len, :]
        pad = x.new_full((B, target_len - L0, D), pad_value)
        return torch.cat([x, pad], dim=1)

    if mode == "pad":
        if L0 >= target_len:
            return x
        pad = x.new_full((B, target_len - L0, D), pad_value)
        return torch.cat([x, pad], dim=1)

    raise ValueError(f"wrong mode：{mode}")


def interleave(xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:

    assert xa.shape == xb.shape, f"interleave shape， {xa.shape} vs {xb.shape}"
    B, L, D = xa.shape
    y = torch.stack([xa, xb], dim=2).reshape(B, 2 * L, D)
    return y


def concat_time(xa: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:

    assert xa.shape[0] == xb.shape[0] and xa.shape[2] == xb.shape[2], "xa =xb"
    return torch.cat([xa, xb], dim=1)


class ProjMLP(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _ResBlock1D(nn.Module):

    def __init__(self, dim: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation, groups=1)
        self.act = nn.GELU()
        self.norm2 = nn.LayerNorm(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation, groups=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm1(x)
        y = rearrange(y, "b l d -> b d l")
        y = self.conv1(y)
        y = rearrange(y, "b d l -> b l d")
        y = self.act(y)

        y = self.norm2(y)
        y = rearrange(y, "b l d -> b d l")
        y = self.conv2(y)
        y = rearrange(y, "b d l -> b l d")

        return residual + y


class MonoCNN(nn.Module):

    def __init__(self, dim: int, depth: int = 2, kernel_size: int = 3, dilated: bool = True):
        super().__init__()
        blocks = []
        for i in range(depth):
            dilation = 2 ** i if dilated else 1
            blocks.append(_ResBlock1D(dim, kernel_size=kernel_size, dilation=dilation))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TorchTFEncoder(nn.Module):

    def __init__(self, dim: int, depth: int = 2, heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_ratio * dim,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


class _MambaStack(nn.Module):

    def __init__(self, dim: int, depth: int = 2, d_state: int = 64, expand: int = 2):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as e:
            raise ImportError(
                "no mamba-ssm"
            ) from e

        import inspect
        sig = inspect.signature(Mamba.__init__)
        kwargs = {"d_model": dim, "d_state": d_state}
        if "d_conv" in sig.parameters:   # 不同版本兼容
            kwargs["d_conv"] = 4
        if "expand" in sig.parameters:
            kwargs["expand"] = expand

        layers = [Mamba(**kwargs) for _ in range(depth)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoMambaPair(nn.Module):

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        use_mamba: bool = True,
        d_state: int = 64,
        expand: int = 2,
        fallback_heads: int = 8,
        fallback_mlp_ratio: int = 4,
    ):
        super().__init__()
        self.use_mamba = use_mamba
        if use_mamba:
            try:
                self.encoder = _MambaStack(dim=dim, depth=depth, d_state=d_state, expand=expand)
            except ImportError:
                self.use_mamba = False
                self.encoder = _TorchTFEncoder(dim=dim, depth=depth, heads=fallback_heads,
                                               mlp_ratio=fallback_mlp_ratio)
        else:
            self.encoder = _TorchTFEncoder(dim=dim, depth=depth, heads=fallback_heads,
                                           mlp_ratio=fallback_mlp_ratio)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_pair: torch.Tensor) -> torch.Tensor:
        y = self.encoder(x_pair)
        y = self.norm(y)
        return y



def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNormForward(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)
        return self.fn(q, k, v)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = self.attend(dots)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, save_hidden=False):
        if save_hidden:
            hidden_list = [x]
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x


class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return target_x


class Transformer(nn.Module):

    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()

        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
            self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)
        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape
        if self.token_len is not None:
            extra_token = repeat(self.extra_token, "1 n d -> b n d", b=b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, : n + self.token_len]
        else:
            x = x + self.pos_embedding[:, : n]
        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)
        return x


class CrossTransformer(nn.Module):

    def __init__(self, *, source_num_frames, tgt_num_frames, dim, depth, heads, mlp_dim,
                 pool='cls', dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()

        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames + 1, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames + 1, dim))
        self.extra_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.CrossTransformerEncoder = CrossTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, source_x, target_x):
        b, n_s, _ = source_x.shape
        b, n_t, _ = target_x.shape

        extra_token = repeat(self.extra_token, "1 1 d -> b 1 d", b=b)

        source_x = torch.cat((extra_token, source_x), dim=1)
        source_x = source_x + self.pos_embedding_s[:, : n_s + 1]

        target_x = torch.cat((extra_token, target_x), dim=1)
        target_x = target_x + self.pos_embedding_t[:, : n_t + 1]

        source_x = self.dropout(source_x)
        target_x = self.dropout(target_x)

        x_s2t = self.CrossTransformerEncoder(source_x, target_x)
        return x_s2t
