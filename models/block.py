from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import LayerScale


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                output = output.to(self.weight.dtype)
            output = output * self.weight
        else:
            output = output.to(x.dtype)
        return output


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, eps=1e-6)


def precompute_freqs_cis_1d(dim: int, end: int = 10000, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes 1D rotary position embeddings (cos, sin) as described in Su et al. (2024)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


def apply_rotary_emb_1d(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies 1D RoPE to query and key tensors of shape [B, num_heads, seq_len, head_dim]."""
    B, H, N, D = xq.shape
    cos = freqs_cos[:N, :].to(device=xq.device, dtype=xq.dtype).unsqueeze(0).unsqueeze(0)
    sin = freqs_sin[:N, :].to(device=xq.device, dtype=xq.dtype).unsqueeze(0).unsqueeze(0)

    xq_r, xq_i = xq[..., : D // 2], xq[..., D // 2 :]
    xk_r, xk_i = xk[..., : D // 2], xk[..., D // 2 :]

    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos

    xq_out = torch.cat([xq_out_r, xq_out_i], dim=-1)
    xk_out = torch.cat([xk_out_r, xk_out_i], dim=-1)
    return xq_out, xk_out


class MMDiTBlock(nn.Module):
    """
    Multimodal Diffusion Transformer (MM-DiT) Block from Stable Diffusion 3 (Esser et al., 2024).
    Processes audio (x) and context/text (c) jointly by concatenating keys and values,
    while maintaining separate query projections and MLPs.
    """
    def __init__(
        self,
        dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        init_values: Optional[float] = None,
        drop_path: float = 0.,
        norm_layer=LayerNorm,
        mlp_layer=Mlp,
        attn_norm_layer=RMSNorm,
        **ignored_kwargs
    ):
        super().__init__()
        attn_norm_layer = attn_norm_layer or norm_layer
        self.dim = dim
        self.context_dim = context_dim if context_dim is not None else dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        # --- Audio (x) stream ---
        self.norm1_x = norm_layer(dim)
        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm_x = attn_norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_x = attn_norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj_x = nn.Linear(dim, dim)
        self.proj_drop_x = nn.Dropout(proj_drop)
        self.ls1_x = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1_x = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_x = norm_layer(dim)
        self.mlp_x = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=proj_drop)
        self.ls2_x = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2_x = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # --- Context / Text (c) stream ---
        self.norm1_c = norm_layer(self.context_dim)
        self.qkv_c = nn.Linear(self.context_dim, dim * 3, bias=qkv_bias)
        self.q_norm_c = attn_norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_c = attn_norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj_c = nn.Linear(dim, self.context_dim)
        self.proj_drop_c = nn.Dropout(proj_drop)
        self.ls1_c = LayerScale(self.context_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1_c = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_c = norm_layer(self.context_dim)
        self.mlp_c = mlp_layer(in_features=self.context_dim, hidden_features=int(self.context_dim * mlp_ratio), drop=proj_drop)
        self.ls2_c = LayerScale(self.context_dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2_c = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        freqs_cos_x: Optional[torch.Tensor] = None,
        freqs_sin_x: Optional[torch.Tensor] = None,
        freqs_cos_c: Optional[torch.Tensor] = None,
        freqs_sin_c: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N_x, C_x = x.shape
        dtype = x.dtype

        # 1. Audio QKV
        x_norm = self.norm1_x(x)
        qkv_x = self.qkv_x(x_norm).reshape(B, N_x, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q_x, k_x, v_x = qkv_x.unbind(0)
        q_x, k_x = self.q_norm_x(q_x).to(dtype), self.k_norm_x(k_x).to(dtype)

        # Apply RoPE to Audio stream
        if freqs_cos_x is not None and freqs_sin_x is not None:
            q_x, k_x = apply_rotary_emb_1d(q_x, k_x, freqs_cos_x, freqs_sin_x)

        if context is not None:
            N_c = context.shape[1]
            c_norm = self.norm1_c(context)
            qkv_c = self.qkv_c(c_norm).reshape(B, N_c, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q_c, k_c, v_c = qkv_c.unbind(0)
            q_c, k_c = self.q_norm_c(q_c).to(dtype), self.k_norm_c(k_c).to(dtype)

            # Apply RoPE to Text stream if provided
            if freqs_cos_c is not None and freqs_sin_c is not None:
                q_c, k_c = apply_rotary_emb_1d(q_c, k_c, freqs_cos_c, freqs_sin_c)

            # Joint Keys and Values
            k_joint = torch.cat([k_x, k_c], dim=2)  # [B, num_heads, N_x + N_c, head_dim]
            v_joint = torch.cat([v_x, v_c], dim=2)  # [B, num_heads, N_x + N_c, head_dim]
            q_joint = torch.cat([q_x, q_c], dim=2)  # [B, num_heads, N_x + N_c, head_dim]

            # Joint Scaled Dot-Product Attention
            attn = F.scaled_dot_product_attention(
                query=q_joint,
                key=k_joint,
                value=v_joint,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            attn = attn.transpose(1, 2).reshape(B, N_x + N_c, self.dim)

            attn_x = attn[:, :N_x]
            attn_c = attn[:, N_x:]

            # Projections & Residuals
            x = x + self.drop_path1_x(self.ls1_x(self.proj_drop_x(self.proj_x(attn_x))))
            context = context + self.drop_path1_c(self.ls1_c(self.proj_drop_c(self.proj_c(attn_c))))

            # MLPs
            x = x + self.drop_path2_x(self.ls2_x(self.mlp_x(self.norm2_x(x))))
            context = context + self.drop_path2_c(self.ls2_c(self.mlp_c(self.norm2_c(context))))
            return x, context
        else:
            attn = F.scaled_dot_product_attention(
                query=q_x,
                key=k_x,
                value=v_x,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            attn = attn.transpose(1, 2).reshape(B, N_x, self.dim)
            x = x + self.drop_path1_x(self.ls1_x(self.proj_drop_x(self.proj_x(attn))))
            x = x + self.drop_path2_x(self.ls2_x(self.mlp_x(self.norm2_x(x))))
            return x, None


# Alias Block to MMDiTBlock
Block = MMDiTBlock
