import math
from typing import Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from models.block import Block, LayerNorm, MMDiTBlock, precompute_freqs_cis_1d
from models.diffloss import FlowMatchLoss, GaussDiffLoss
from models.jepaloss import JepaLoss
from text.symbols import symbols


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len, device=order.device)
    masking = torch.scatter(
        masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len, device=order.device)).bool()
    return masking


class JEPAT(nn.Module):
    no_weight_decay_set = {
        'diffloss', 'jepaloss', 'buffer', "head", "class_emb", "fake_latent", "z_proj",
        "phoneme_embedding", "decoder_embed", "mask_token",
    }

    def __init__(self,
                 spec_height=128, spec_width=512, patch_size=16,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
                 mlp_ratio=4, norm_layer=LayerNorm,
                 qk_norm: bool = True,
                 in_channels=1,
                 phoneme_vocab_size=None,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 diffloss_d=6,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 diffloss: str = "flow",  # gauss, flow, none
                 jepaloss: str = "jepa",  # jepa, none
                 grad_checkpointing: bool = False,
                 max_seq_len: int = 10000,
                 language='arabic',
                 **ignored_kwargs,
                 ):
        super().__init__()

        self.language = language
        self.in_channels = in_channels
        self.spec_height = spec_height
        self.spec_width = spec_width

        if isinstance(patch_size, int):
            self.patch_h = self.patch_w = patch_size
        else:
            self.patch_h, self.patch_w = patch_size

        assert spec_height % self.patch_h == 0, f"spec_height {spec_height} must be divisible by patch_h {self.patch_h}"
        assert spec_width % self.patch_w == 0, f"spec_width {spec_width} must be divisible by patch_w {self.patch_w}"

        self.seq_h = spec_height // self.patch_h
        self.seq_w = spec_width // self.patch_w
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = in_channels * self.patch_h * self.patch_w
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.grad_checkpointing = grad_checkpointing

        # --- 1D Rotary Position Embedding (RoPE) ---
        encoder_head_dim = encoder_embed_dim // encoder_num_heads
        decoder_head_dim = decoder_embed_dim // decoder_num_heads
        enc_cos, enc_sin = precompute_freqs_cis_1d(encoder_head_dim, end=max_seq_len)
        dec_cos, dec_sin = precompute_freqs_cis_1d(decoder_head_dim, end=max_seq_len)
        self.register_buffer('enc_freqs_cos', enc_cos, persistent=False)
        self.register_buffer('enc_freqs_sin', enc_sin, persistent=False)
        self.register_buffer('dec_freqs_cos', dec_cos, persistent=False)
        self.register_buffer('dec_freqs_sin', dec_sin, persistent=False)

        # --- Text / Phoneme Encoder (Pure Paper Approach) ---
        if phoneme_vocab_size is None:
            phoneme_vocab_size = max(128, len(symbols) + 10)
        self.phoneme_vocab_size = phoneme_vocab_size
        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        self.fake_latent = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))

        # --- Masking Distribution ---
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --- Encoder (MM-DiT) ---
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = norm_layer(encoder_embed_dim)

        self.encoder_blocks = nn.ModuleList([
            MMDiTBlock(dim=encoder_embed_dim,
                       context_dim=encoder_embed_dim,
                       num_heads=encoder_num_heads,
                       mlp_ratio=mlp_ratio,
                       qkv_bias=True,
                       qk_norm=qk_norm,
                       proj_drop=proj_dropout,
                       attn_drop=attn_dropout,
                       norm_layer=norm_layer)
            for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --- Decoder (MM-DiT) ---
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            MMDiTBlock(dim=decoder_embed_dim,
                       context_dim=decoder_embed_dim,
                       num_heads=decoder_num_heads,
                       mlp_ratio=mlp_ratio,
                       qkv_bias=True,
                       qk_norm=qk_norm,
                       proj_drop=proj_dropout,
                       attn_drop=attn_dropout,
                       norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.initialize_weights()

        # --- Losses ---
        self.diffloss = {
            "gauss": GaussDiffLoss,
            "flow": FlowMatchLoss,
            "none": nn.Identity,
        }[diffloss](
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing,
        )
        self.diffusion_batch_mul = diffusion_batch_mul

        self.jepaloss = {
            "jepa": JepaLoss,
            "none": nn.Identity,
        }[jepaloss](
            decoder_embed_dim, encoder_embed_dim, norm_layer, beta=2.0)

    def initialize_weights(self):
        self.apply(self._init_weights)
        torch.nn.init.normal_(self.phoneme_embedding.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        if hasattr(self, "mask_token"):
            torch.nn.init.normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        # x: [bsz, in_channels, spec_height, spec_width]
        bsz = x.shape[0]
        p_h, p_w = self.patch_h, self.patch_w
        x = F.unfold(x, kernel_size=(p_h, p_w), stride=(p_h, p_w))
        x = x.transpose(1, 2)
        return x  # [bsz, seq_len, token_embed_dim]

    def unpatchify(self, x):
        # x: [bsz, seq_len, token_embed_dim]
        bsz = x.shape[0]
        p_h, p_w = self.patch_h, self.patch_w
        c = self.in_channels
        h_, w_ = self.seq_h, self.seq_w

        x = x.transpose(1, 2).reshape(bsz, c * p_h * p_w, h_ * w_)
        x = F.fold(x, output_size=(h_ * p_h, w_ * p_w),
                   kernel_size=(p_h, p_w), stride=(p_h, p_w))
        return x  # [bsz, c, spec_height, spec_width]

    def sample_orders(self, bsz):
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).to(self.fake_latent.device).long()
        return orders

    def random_masking(self, x, orders):
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = max(0, int(np.ceil(seq_len * mask_rate)))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def get_text_context(self, phoneme_tokens):
        # phoneme_tokens: [bsz, text_len] LongTensor or list
        if isinstance(phoneme_tokens, list):
            # If list of token lists
            max_len = max(len(p) for p in phoneme_tokens)
            pad_tensor = torch.zeros(len(phoneme_tokens), max_len, dtype=torch.long, device=self.fake_latent.device)
            for i, p in enumerate(phoneme_tokens):
                pad_tensor[i, :len(p)] = torch.tensor(p, dtype=torch.long, device=self.fake_latent.device)
            phoneme_tokens = pad_tensor
        else:
            phoneme_tokens = phoneme_tokens.to(self.fake_latent.device)

        bsz = phoneme_tokens.size(0)
        c = self.phoneme_embedding(phoneme_tokens)  # [bsz, text_len, encoder_embed_dim]

        if self.training:
            drop_mask = (torch.rand(bsz, device=c.device) < self.label_drop_prob).unsqueeze(1).unsqueeze(2)
            c = torch.where(drop_mask, self.fake_latent.expand_as(c), c)
        return c

    def forward_encoder(self, x, mask, context):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # dropping masked tokens
        indices = (1 - mask.float()).nonzero(as_tuple=True)
        x = x[indices].reshape(bsz, -1, embed_dim)
        x = self.z_proj_ln(x)

        # apply MM-DiT blocks
        for blk in self.encoder_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x, context = checkpoint(blk, x, context, self.enc_freqs_cos, self.enc_freqs_sin, self.enc_freqs_cos, self.enc_freqs_sin, use_reentrant=False)
            else:
                x, context = blk(x, context=context,
                                 freqs_cos_x=self.enc_freqs_cos, freqs_sin_x=self.enc_freqs_sin,
                                 freqs_cos_c=self.enc_freqs_cos, freqs_sin_c=self.enc_freqs_sin)
        x = self.encoder_norm(x)
        return x, context

    def forward_decoder(self, x, mask, context):
        bsz = len(x)
        x = self.decoder_embed(x)

        # pad mask tokens back to full sequence
        x_after_pad = self.mask_token.repeat(bsz, self.seq_len, 1).type_as(x).clone()
        indices = (1 - mask).nonzero(as_tuple=True)
        x_after_pad[indices] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        x = x_after_pad

        # apply MM-DiT blocks
        for blk in self.decoder_blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x, context = checkpoint(blk, x, context, self.dec_freqs_cos, self.dec_freqs_sin, self.dec_freqs_cos, self.dec_freqs_sin, use_reentrant=False)
            else:
                x, context = blk(x, context=context,
                                 freqs_cos_x=self.dec_freqs_cos, freqs_sin_x=self.dec_freqs_sin,
                                 freqs_cos_c=self.dec_freqs_cos, freqs_sin_c=self.dec_freqs_sin)
        x = self.decoder_norm(x)
        return x

    def diffusion_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        # Check if denoising network is 2D (SimpleMLPAdaLN) or 3D (SpatialDiT)
        is_2d = not hasattr(self.diffloss.net, 'pos_embed')
        if is_2d:
            target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
            z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
            mask = mask.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        else:
            target = target.repeat(self.diffusion_batch_mul, 1, 1)
            z = z.repeat(self.diffusion_batch_mul, 1, 1)
            mask = mask.repeat(self.diffusion_batch_mul, 1)
        loss = self.diffloss(target=target, z=z, mask=mask)
        return loss

    def jepa_loss(self, x, ema_x, mask):
        return self.jepaloss(x, ema_x, mask)

    @torch.no_grad()
    def forward_ema_encoder(self, specs, phonemes):
        context = self.get_text_context(phonemes)
        x = self.patchify(specs)
        mask = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        x, _ = self.forward_encoder(x, mask, context)
        return x

    def forward(self, specs, phonemes, ema_x=None):
        # patchify
        x = self.patchify(specs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        context = self.get_text_context(phonemes)

        # MM-DiT encoder & decoder
        x, enc_context = self.forward_encoder(x, mask, context)
        z = self.forward_decoder(x, mask, enc_context)

        # diffusion loss
        if not isinstance(self.diffloss, nn.Identity):
            diffloss = self.diffusion_loss(z=z, target=gt_latents, mask=mask)
        else:
            diffloss = torch.zeros(1, device=z.device, dtype=z.dtype)

        # jepa prediction loss
        if not isinstance(self.jepaloss, nn.Identity) and ema_x is not None:
            jepa_loss = self.jepa_loss(z, ema_x, mask)
        else:
            jepa_loss = torch.zeros(1, device=z.device, dtype=z.dtype)

        return diffloss, jepa_loss

    def sample_tokens(self, bsz, num_iter=64, cfg_scale=1.0, labels=None, temperature=1.0, time_shifting_factor=1.0, progress=False):
        # labels: [bsz, text_len] LongTensor phonemes
        device = self.fake_latent.device
        
        if labels is not None:
            if isinstance(labels, list):
                from text import tokens_to_ids, phonemes_to_tokens, arabic_to_phonemes
                token_lists = []
                for item in labels:
                    if isinstance(item, str):
                        ph = arabic_to_phonemes(item)
                        tk = phonemes_to_tokens(ph)
                        token_lists.append(tokens_to_ids(tk))
                    else:
                        token_lists.append(item)
                max_len = max(len(p) for p in token_lists)
                labels = torch.zeros(len(token_lists), max_len, dtype=torch.long, device=device)
                for i, p in enumerate(token_lists):
                    labels[i, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)
            context = self.phoneme_embedding(labels.to(device))
        else:
            context = self.fake_latent.repeat(bsz, 1, 1)

        # CFG handling
        if cfg_scale != 1.0:
            null_context = self.fake_latent.repeat(bsz, context.size(1), 1)
            context = torch.cat([context, null_context], dim=0)

        eff_bsz = context.size(0)
        tokens = torch.zeros(eff_bsz, self.seq_len, self.token_embed_dim, device=device)
        mask = torch.ones(eff_bsz, self.seq_len, device=device)

        # Forward through MM-DiT with 100% mask
        x, enc_context = self.forward_encoder(tokens, mask, context)
        z = self.forward_decoder(x, mask, enc_context)

        # Flow match / diffusion sampling
        is_2d = not hasattr(self.diffloss.net, 'pos_embed')
        if is_2d:
            z_flat = z.reshape(eff_bsz * self.seq_len, -1)
            sampled_flat = self.diffloss.sample(
                z_flat, temperature=temperature, cfg_scale=cfg_scale, time_shifting_factor=time_shifting_factor)
            sampled_tokens = sampled_flat.reshape(eff_bsz, self.seq_len, self.token_embed_dim)
        else:
            sampled_tokens = self.diffloss.sample(
                z, temperature=temperature, cfg_scale=cfg_scale, time_shifting_factor=time_shifting_factor)

        if cfg_scale != 1.0:
            sampled_tokens, _ = sampled_tokens.chunk(2, dim=0)

        # Unpatchify back to spectrogram
        specs = self.unpatchify(sampled_tokens)
        return specs


def JEPAT_base(**kwargs):
    model = JEPAT(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, diffloss_d=6, diffloss_w=1024,
        **kwargs)
    return model


def JEPAT_large(**kwargs):
    model = JEPAT(
        encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=24, decoder_num_heads=16,
        mlp_ratio=4, diffloss_d=8, diffloss_w=1280,
        **kwargs)
    return model


def JEPAT_huge(**kwargs):
    model = JEPAT(
        encoder_embed_dim=1280, encoder_depth=32, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=32, decoder_num_heads=16,
        mlp_ratio=4, diffloss_d=12, diffloss_w=1536,
        **kwargs)
    return model
