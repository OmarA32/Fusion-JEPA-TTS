import math

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from models.block import Block, LayerNorm
from models.diffloss import FlowMatchLoss, GaussDiffLoss
from models.jepaloss import JepaLoss
import clip

from text import arabic_to_tokens
from text.symbols import symbols

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :].to(x.dtype)
        return self.dropout(x)


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len, device=order.device)
    masking = torch.scatter(
        masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len, device=order.device)).bool()
    return masking


class JEPAT(nn.Module):
    no_weight_decay_set = {
        'diffloss', 'jepaloss', 'buffer', "head", "class_emb", "fake_latent", "z_proj",
        "encoder_pos_embed_learned", "decoder_embed", "mask_token", "decoder_pos_embed_learned",
        "clip"
    }

    def __init__(self, spec_height=128, spec_width=512, patch_size=16,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4, norm_layer=LayerNorm,
                 qk_norm: bool = False,
                 in_channels=1,
                 language='arabic',
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,

                 diffloss: str = "gauss",  # gauss, flow, none
                 jepaloss: str = "jepa",  # jepa, none
                 grad_checkpointing: bool = False,
                 **ignored_kwargs,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.language = language

        self.spec_height = spec_height
        self.spec_width = spec_width
        self.patch_size = patch_size
        self.seq_h = spec_height // patch_size
        self.seq_w = spec_width // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = in_channels * patch_size**2
        self.in_channels = in_channels
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.buffer_size = buffer_size
        self.grad_checkpointing = grad_checkpointing

        self.cross_attention = nn.MultiheadAttention(
        embed_dim=decoder_embed_dim,  
        num_heads=8,  
        batch_first=True,
        dropout=attn_dropout
        )

        # --------------------------------------------------------------------------
        # Language Module Setup
        self.num_classes = class_num

        # Unified TTS Text Processor (used for both Arabic and English)
        self.phoneme_embedding = nn.Embedding(len(symbols), 512)
        self.phoneme_pos_encoder = PositionalEncoding(d_model=512, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True, dropout=0.1)
        self.arabic_text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # CLIP output is 512, project to encoder_embed_dim
        self.class_emb = nn.Sequential(
            nn.Linear(512, encoder_embed_dim),
            nn.LayerNorm(encoder_embed_dim),
            nn.GELU(),
            nn.Linear(encoder_embed_dim, encoder_embed_dim)
        )
        
        self.label_drop_prob = label_drop_prob
        self.fuse_proj = nn.Linear(self.decoder_embed_dim + self.encoder_embed_dim,
                           self.decoder_embed_dim, bias=True)
        # Fake class embedding for cfg_scale's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # JEPAT variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # Define knowledge buffer here and append it in  encoder.
        self.buffer = nn.Parameter(torch.zeros(
            1, self.buffer_size, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # JEPAT encoder specifics (used for classification and diffusion.)
        self.z_proj = nn.Linear(self.token_embed_dim,
                                encoder_embed_dim, bias=True)
        self.z_proj_ln = norm_layer(encoder_embed_dim)

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim,
                  encoder_num_heads,
                  mlp_ratio,
                  qkv_bias=True,
                  qk_norm=qk_norm,
                  proj_drop=proj_dropout,
                  attn_drop=attn_dropout)
            for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # Positional Embedding for decoder
        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # JEPAT decoder specifics (no requires for classification task.)
        self.decoder_embed = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim,
                  decoder_num_heads,
                  mlp_ratio,
                  qkv_bias=True,
                  qk_norm=qk_norm,
                  proj_drop=proj_dropout,
                  attn_drop=attn_dropout)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # --------------------------------------------------------------------------
        # Positional Embedding for encoder
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len, decoder_embed_dim))

        # NOTE: initialize weights before diffusion loss!
        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss (only requires for diffusion learning.)

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

    def count_params(self):
        def count_param(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        total_params = count_param(self)
        encoder_params = count_param(self.encoder_blocks)
        decoder_params = count_param(self.decoder_blocks)
        denoising_params = count_param(self.diffloss)
        return {
            "# total": total_params,
            "# encoder": encoder_params,
            "# decoder": decoder_params,
            "# denoising": denoising_params
        }

    def initialize_weights(self):
        # apply default init first.
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # parameters
        # Initialize class_emb projection layers
        for m in self.class_emb.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                    
        torch.nn.init.normal_(self.fake_latent, std=.02)
        if hasattr(self, "mask_token"):
            torch.nn.init.normal_(self.mask_token, std=.02)
        if hasattr(self, "encoder_pos_embed_learned"):
            torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        if hasattr(self, "decoder_pos_embed_learned"):
            torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)

        # init for head weights
        if hasattr(self, "head"):
            torch.nn.init.trunc_normal_(self.head.weight, std=0.01)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        p = self.patch_size

        x = F.unfold(x, kernel_size=(p, p), stride=p)
        x = x.transpose(1, 2)

        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.in_channels
        h_, w_ = self.seq_h, self.seq_w

        x = x.transpose(1, 2).reshape(bsz, c * p * p, h_ * w_)
        x = F.fold(x, output_size=(h_ * p, w_ * p),
                   kernel_size=(p, p), stride=p)

        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).to(self.fake_latent.device).long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = max(0, int(np.ceil(seq_len * mask_rate)))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_encoder(self, x, mask, class_embedding):
        """
        Returns: 
            feature: batch size, seq_len, dims
        """
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape
        x = x + self.encoder_pos_embed_learned

        # append buffer
        pooled_class_embedding = class_embedding.mean(dim=1) if class_embedding.dim() == 3 else class_embedding
        buffer = self.buffer + pooled_class_embedding.unsqueeze(1)
        x = torch.cat([buffer.type_as(x), x], dim=1)

        mask_with_buffer = torch.cat(
            [torch.zeros(bsz, self.buffer_size, device=x.device), mask], dim=1)

        # dropping
        indices = (1-mask_with_buffer.float()).nonzero(as_tuple=True)
        x = x[indices].reshape(bsz, -1, embed_dim)

        tokens = x.shape[1]
        freqs_cis = None

        # encoder position embedding
        x = self.z_proj_ln(x)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():  # type: ignore
            for blk in self.encoder_blocks:
                # pad None to support checkpoint
                x = checkpoint(blk, x, None, None, freqs_cis)
        else:
            for blk in self.encoder_blocks:
                x = blk(x, freqs_cis=freqs_cis)
        x = self.encoder_norm(x)

        return x

    def forward_decoder(self, x, mask, class_embedding=None):
        bsz = len(x)
        x = self.decoder_embed(x)

        # If class_embedding is provided, add it to the decoder input
        if class_embedding is not None:
            pooled_class_embedding = class_embedding.mean(dim=1) if class_embedding.dim() == 3 else class_embedding
            pooled_class_embedding = pooled_class_embedding.unsqueeze(1).expand(bsz, x.size(1), -1)
            x = x + pooled_class_embedding

        mask_with_buffer = torch.cat(
            [torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        x_after_pad = self.mask_token.repeat(
            bsz, self.buffer_size + self.seq_len, 1).type_as(x).clone()
        indices = (1 - mask_with_buffer).nonzero(as_tuple=True)
        x_after_pad[indices] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
    
        x_after_pad[:, self.buffer_size:] += self.decoder_pos_embed_learned
        freqs_cis = None

        # decoder position embedding
        x = x_after_pad

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():  # type: ignore
            for blk in self.decoder_blocks:
                # pad None to support checkpoint
                x = checkpoint(blk, x, None, None, freqs_cis)
        else:
            for blk in self.decoder_blocks:
                x = blk(x, freqs_cis=freqs_cis)
        x = self.decoder_norm(x)

        # drop buffer size.
        x = x[:, self.buffer_size:] #[32,320,768]
        return x

    def diffusion_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.repeat(self.diffusion_batch_mul, 1, 1)
        z = z.repeat(self.diffusion_batch_mul, 1, 1)
        mask = mask.repeat(self.diffusion_batch_mul, 1)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def jepa_loss(self, x, ema_x, mask):
        """
        ema_x: b, n, c
        x: b, n, c
        mask: b, n
        """
        return self.jepaloss(x, ema_x, mask)

    def encode_labels_with_clip(self, labels):
        """
        Encode labels using CLIP model
        """
        with torch.no_grad():
            if isinstance(labels, torch.Tensor):
                text_labels = [f"a photo of class {int(label)}" for label in labels]
                text_tokens = clip.tokenize(text_labels).to(self.fake_latent.device)
                clip_features = self.clip_model.encode_text(text_tokens).float()
            elif isinstance(labels, list) and isinstance(labels[0], str):
                text_tokens = clip.tokenize(labels).to(self.fake_latent.device)
                clip_features = self.clip_model.encode_text(text_tokens).float()
            else:
                text_labels = [f"a photo of class {int(label)}" for label in labels]
                text_tokens = clip.tokenize(text_labels).to(self.fake_latent.device)
                clip_features = self.clip_model.encode_text(text_tokens).float()
        
        return clip_features

    def get_phoneme_embedding(self, text_tokens):
        """
        Arabic FastPitch Text Processor hook.
        text_tokens can be a list of strings or an integer tensor [bsz, seq_len]
        """
        device = self.fake_latent.device
        
        if isinstance(text_tokens, list):
            bsz = len(text_tokens)
            token_lists = []
            max_len = 0
            for t in text_tokens:
                try:
                    from text import tokens_to_ids
                    phonemes = arabic_to_tokens(t)
                    tokens = tokens_to_ids(phonemes)
                except Exception as e:
                    print(f"Error phonemizing '{t}': {e}. Using dummy tokens.")
                    tokens = [0, 1, 2] # Dummy fallback
                token_lists.append(tokens)
                max_len = max(max_len, len(tokens))
                
            padded_tokens = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
            for i, tokens in enumerate(token_lists):
                padded_tokens[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        else:
            # Assumed to be a padded tensor of token IDs from Dataset
            padded_tokens = text_tokens.to(device)
            
        x = self.phoneme_embedding(padded_tokens) # [bsz, text_len, 512]
        x = self.phoneme_pos_encoder(x)
        x = self.arabic_text_encoder(x) # [bsz, text_len, 512]
        
        return x

    def get_class_embedding(self, labels):
        bsz = len(labels) if isinstance(labels, list) else labels.size(0)

        # In JEPA-TTS, both Arabic and English text use the phonetic embedding layer, not CLIP
        features = self.get_phoneme_embedding(labels)
        
        # Project features to encoder_embed_dim
        class_embedding = self.class_emb(features)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            
            if class_embedding.dim() == 3:
                drop_latent_mask = drop_latent_mask.unsqueeze(-1).unsqueeze(-1).type_as(class_embedding).to(self.fake_latent.device)
            else:
                drop_latent_mask = drop_latent_mask.unsqueeze(-1).type_as(class_embedding).to(self.fake_latent.device)
                
            class_embedding = drop_latent_mask * self.fake_latent + \
                (1 - drop_latent_mask) * class_embedding
        return class_embedding

    @torch.no_grad()
    def forward_ema_encoder(self, imgs, labels):
        class_embedding = self.get_class_embedding(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        mask = torch.zeros(x.shape[0], x.shape[1], device=x.device)

        # mae encoder
        x = self.forward_encoder(x, mask, class_embedding)
        return x[:, self.buffer_size:]  # drop the buffer size for ema.

    def forward(self, imgs, labels, ema_x=None):
        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        orders = self.sample_orders(bsz=x.size(0))
        mask = self.random_masking(x, orders)

        class_embedding = self.get_class_embedding(labels)

        # mae encoder
        x = self.forward_encoder(x, mask, class_embedding)

        # mae decoder
        z = self.forward_decoder(x, mask, class_embedding)
        # Cross attn
        if class_embedding.dim() == 2:
            # English CLIP (Single Vector)
            class_embedding_ca = class_embedding.unsqueeze(1) 
            class_embedding_concat = class_embedding.unsqueeze(1).expand(-1, z.size(1), -1) 
        else:
            # Arabic Phonemes (Full Sequence)
            class_embedding_ca = class_embedding 
            class_embedding_concat = class_embedding.mean(dim=1, keepdim=True).expand(-1, z.size(1), -1)

        z_attended, _ = self.cross_attention(query=z, key=class_embedding_ca, value=class_embedding_ca)
        z = z + z_attended 
        z = torch.cat([z, class_embedding_concat], dim=-1)
        z = self.fuse_proj(z) 

        # diffloss
        if not isinstance(self.diffloss, nn.Identity):
            diffloss = self.diffusion_loss(z=z, target=gt_latents, mask=mask)
        else:
            diffloss = torch.zeros(1).type_as(z)

        # l1loss
        if not isinstance(self.jepaloss, nn.Identity):
            assert ema_x is not None, "ema_x must be passed if jepa loss is required."
            jepa_loss = self.jepa_loss(z, ema_x, mask)
        else:
            jepa_loss = torch.zeros(1).type_as(z)

        return diffloss, jepa_loss

    def sample_tokens(self, bsz, num_iter=64, cfg_scale=1.0, cfg_schedule="linear", labels=None, temperature=1.0, time_shifting_factor=1.0, progress=False):

        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len, device=self.fake_latent.device)
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device=self.fake_latent.device)
        orders = self.sample_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and cfg_scale
            if labels is not None:
                features = self.get_phoneme_embedding(labels)
                class_embedding = self.class_emb(features)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if not cfg_scale == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                if class_embedding.dim() == 3:
                    seq_len_text = class_embedding.size(1)
                    fake_class = self.fake_latent.unsqueeze(1).repeat(bsz, seq_len_text, 1)
                else:
                    fake_class = self.fake_latent.repeat(bsz, 1)
                class_embedding = torch.cat(
                    [class_embedding, fake_class], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.forward_encoder(tokens, mask, class_embedding)

            # mae decoder
            z = self.forward_decoder(x, mask, class_embedding)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor(
                [np.floor(self.seq_len * mask_ratio)]).to(self.fake_latent.device)

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).to(self.fake_latent.device),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))  # type: ignore

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(
                    mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg_scale == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # cfg_scale schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg_scale - 1) * (self.seq_len -
                                                  mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg_scale
            else:
                raise NotImplementedError
                
            # Sample token latents for the FULL sequence using SpatialDiT
            sampled_tokens = self.diffloss.sample(
                z, temperature=temperature, cfg_scale=cfg_iter, time_shifting_factor=time_shifting_factor)
                
            if not cfg_scale == 1.0:
                sampled_tokens, _ = sampled_tokens.chunk(
                    2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(
                as_tuple=True)] = sampled_tokens[mask_to_pred.nonzero(as_tuple=True)]
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens


def JEPAT_base(**kwargs):
    # total: 212,017,440
    # encoder: 85,056,000
    # decoder: 85,056,000
    # denoising: 35,749,920
    model = JEPAT(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, diffloss_d=6, diffloss_w=1024,
        **kwargs)
    return model


def JEPAT_large(**kwargs):
    # total: 687,421,984
    # encoder: 302,312,448
    # decoder: 302,312,448
    # denoising: 72,230,432
    model = JEPAT(
        encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=24, decoder_num_heads=16,
        mlp_ratio=4, diffloss_d=8, diffloss_w=1280,
        **kwargs)
    return model


def JEPAT_huge(**kwargs):
    # total: 1,426,730,784
    # encoder: 629,683,200
    # decoder: 629,683,200
    # denoising: 151,206,944
    model = JEPAT(
        encoder_embed_dim=1280, encoder_depth=32, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=32, decoder_num_heads=16,
        mlp_ratio=4, diffloss_d=12, diffloss_w=1536,
        **kwargs)
    return model
