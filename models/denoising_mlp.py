import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    A DiT block with Self-Attention and AdaLN modulation.
    """

    def __init__(
        self,
        channels,
        num_heads=8,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4, bias=True),
            nn.GELU(),
            nn.Linear(channels * 4, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 6 * channels, bias=True)
        )

    def forward(self, x, y):
        # x: [B, L, C], y: [B, C]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(6, dim=-1)
        
        # Self Attention
        h = modulate(self.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP
        h = modulate(self.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
        mlp_out = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift.unsqueeze(1), scale.unsqueeze(1))
        x = self.linear(x)
        return x


class SpatialDiT(nn.Module):
    """
    The Spatial DiT for Diffusion Loss.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        
        # 256 is the max sequence length (128//16 * 512//16 = 8 * 32 = 256)
        self.pos_embed = nn.Parameter(torch.zeros(1, 256, model_channels))

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(DiTBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        torch.nn.init.normal_(self.pos_embed, std=.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        x = self.input_proj(x)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :x.size(1), :]
        
        t_emb = self.time_embed(t) # [B, model_channels]
        
        c_seq = self.cond_embed(c) # [B, L, model_channels]
        x = x + c_seq
        
        y = t_emb # [B, model_channels]

        if self.grad_checkpointing and not torch.jit.is_scripting(): # type: ignore
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        
        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([eps, eps], dim=0)
        return eps
