import os
import argparse
import torch
import matplotlib.pyplot as plt
from data.dataset import JEPADataset, jepa_collate_fn
from torch.utils.data import DataLoader
from models.jepat_lightning import JEPATLightning

def run_debug_inference(lang="arabic", db="nawar_halabi", ckpt_path="training_logs/arabic/last.ckpt"):
    device = 'cuda' if torch.cuda.is_available() else 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    print(f'Using device: {device}')

    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f'Loading weights from {ckpt_path}...')
    model_lightning = JEPATLightning.load_from_checkpoint(ckpt_path, map_location=device)
    model_lightning.eval()
    model = model_lightning.model

    print(f'Loading 1 batch of {db} data...')
    dataset = JEPADataset(split='test', lang=lang, db=db)
    loader = DataLoader(dataset, batch_size=1, collate_fn=jepa_collate_fn)
    mel_specs, text_ids, _ = next(iter(loader))
    mel_specs = mel_specs.to(device)
    text_ids = text_ids.to(device)

    with torch.no_grad():
        x = model.patchify(mel_specs)
        gt_latents = x.clone().detach()
        class_embedding = model.get_class_embedding(text_ids)

        # INFERENCE MODE: 100% Masked, 0% Audio Input!
        bsz = x.size(0)
        inference_mask = torch.ones(bsz, model.seq_len).to(device)
        blank_audio = torch.zeros(bsz, model.seq_len, model.token_embed_dim).to(device)

        # MAE Encoder & Decoder (BLIND to audio)
        x_enc = model.forward_encoder(blank_audio, inference_mask, class_embedding)
        z = model.forward_decoder(x_enc, inference_mask, class_embedding)
        
        # Cross Attention
        class_embedding_ca = class_embedding
        class_embedding_concat = class_embedding.mean(dim=1, keepdim=True).expand(-1, z.size(1), -1)
        z_attended, _ = model.cross_attention(query=z, key=class_embedding_ca, value=class_embedding_ca)
        z = z + z_attended 
        z = torch.cat([z, class_embedding_concat], dim=-1)
        z = model.fuse_proj(z)

        diffloss_module = model.diffloss
        target = gt_latents
        
        # Reshape for Diffusion MLP
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(model.diffusion_batch_mul, 1)
        z_reshaped = z.reshape(bsz*seq_len, -1).repeat(model.diffusion_batch_mul, 1)
        
        # Flow Matching sampling
        from models.diffloss.flow_match.flow_match import sample
        # We sample random t just like training, so we can see how it performs at a random timestep
        t, x0, x1 = sample(target)
        
        # Or we can force t=0.0 to see the exact start of inference!
        # Let's force t=0.0 for a true "Start of Inference" look!
        t = torch.zeros_like(t)
        
        dims = [1] * (len(x1.size()) - 1)
        t_ = t.view(t.size(0), *dims)
        
        # If t=0, xt is just pure noise (x0)
        xt = t_ * x1 + (1 - t_) * x0
        ut_real = x1 - x0
        
        model_kwargs = dict(c=z_reshaped)
        ut_predicted = diffloss_module.net(xt, t, **model_kwargs)
        
        # The 1-Step Euler Prediction
        x_pred = xt + (1 - t_) * ut_predicted
        x_perfect = xt + (1 - t_) * ut_real

    # Unpatchify for visualization
    def reshape_to_img(tensor):
        chunk = tensor[:seq_len]
        chunk = chunk.unsqueeze(0)
        return model.unpatchify(chunk).squeeze().cpu().numpy()

    img_real_mel = reshape_to_img(x1)
    img_xt = reshape_to_img(xt)
    img_ut_predicted = reshape_to_img(ut_predicted)
    img_x_pred = reshape_to_img(x_pred)
    img_x_perfect = reshape_to_img(x_perfect)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(img_xt, origin='lower', aspect='auto', cmap='viridis')
    axes[0, 0].set_title('xt (Inference Start: Pure Noise, t=0)')

    axes[0, 1].imshow(img_ut_predicted, origin='lower', aspect='auto', cmap='plasma')
    axes[0, 1].set_title('Inference Predicted Velocity (No Audio Context)')

    axes[0, 2].imshow(img_x_pred, origin='lower', aspect='auto', cmap='viridis')
    axes[0, 2].set_title("xt + (1-t) * Velocity\n(Model's 1-Step Inference!)")

    axes[1, 0].axis('off')

    axes[1, 1].imshow(img_x_perfect, origin='lower', aspect='auto', cmap='viridis')
    axes[1, 1].set_title('Perfect 1-Step Math Check')

    axes[1, 2].imshow(img_real_mel, origin='lower', aspect='auto', cmap='viridis')
    axes[1, 2].set_title('x1 (Hidden Ground Truth)')

    plt.tight_layout()
    os.makedirs('test_results', exist_ok=True)
    out_path = f"test_results/inference_step_debug_{lang}.png"
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug the Flow Matching inference step.")
    parser.add_argument("--lang", type=str, default="arabic", choices=["arabic", "english"], help="Language of the dataset.")
    parser.add_argument("--db", type=str, default="nawar_halabi", help="Dataset name (e.g. nawar_halabi, ljspeech).")
    parser.add_argument("--ckpt", type=str, default="training_logs/arabic/last.ckpt", help="Path to the checkpoint file.")
    args = parser.parse_args()

    run_debug_inference(lang=args.lang, db=args.db, ckpt_path=args.ckpt)
