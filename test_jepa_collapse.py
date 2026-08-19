import torch
import os
from models.jepa_lightning import JEPALightning

def check_jepa_collapse(ckpt_path):
    print(f"Loading checkpoint from: {ckpt_path}")
    
    # Load model
    model = JEPALightning(language="english")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        # Filter diffloss just to be safe if it's the legacy checkpoint
        stripped_state = {k: v for k, v in state_dict.items() if "diffloss" not in k}
        model.load_state_dict(stripped_state, strict=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    model.eval()

    # Create dummy inputs: two completely different random mel spectrograms and texts
    # Shape: (batch_size, channels, time, mel_bins) -> (2, 1, 512, 128)
    batch_size = 2
    mel_specs = torch.randn(batch_size, 1, 512, 128) 
    
    # Text ids: (batch_size, seq_len)
    text_ids = torch.randint(0, 100, (batch_size, 100))

    with torch.no_grad():
        # Get representations from the EMA encoder (which guides the JEPA loss and is used as targets)
        ema_repr = model.ema_model.forward_ema_encoder(mel_specs, text_ids)
        
        # Get representations from the main encoder
        main_repr = model.model.forward_ema_encoder(mel_specs, text_ids)

    print("\n--- Representation Collapse Analysis ---")
    
    def analyze_repr(name, representations):
        # representations shape: (batch_size, seq_len, embed_dim)
        print(f"\nAnalyzing {name}:")
        
        # Calculate standard deviation across the batch dimension
        # If std is 0, it means the model outputs the exact same thing for different inputs
        std_across_batch = representations.std(dim=0).mean().item()
        print(f"Mean Std Dev across batch (should be > 0): {std_across_batch:.6f}")
        
        if std_across_batch < 1e-4:
            print("[WARN] High risk of representation collapse! The model is ignoring the input.")
        else:
            print("[OK] HEALTHY: The representations vary for different inputs.")
            
        # Calculate cosine similarity between the first and second item in the batch
        # Flatten the spatial dimensions for similarity
        flat_repr_1 = representations[0].flatten().unsqueeze(0)
        flat_repr_2 = representations[1].flatten().unsqueeze(0)
        
        cos_sim = torch.nn.functional.cosine_similarity(flat_repr_1, flat_repr_2).item()
        print(f"Cosine Similarity between two different random inputs: {cos_sim:.4f}")
        
        if cos_sim > 0.99:
            print("[WARN] The embeddings for two completely random audio clips are identical! Collapse confirmed.")
        else:
            print("[OK] HEALTHY: The embeddings are sufficiently distinct.")

    analyze_repr("Main Encoder", main_repr)
    analyze_repr("EMA Encoder", ema_repr)

if __name__ == "__main__":
    ckpt = "training_logs/english/last.ckpt"
    if not os.path.exists(ckpt):
        ckpt = "training_logs/english/ARCHIVED_last.ckpt"
    check_jepa_collapse(ckpt)
