import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data.dataset import JEPADataset, jepa_collate_fn

def main():
    print("Initializing JEPADataset...")
    # Just take 10 examples for a quick test
    dataset = JEPADataset(split="train[:10]", max_frames=512)
    print(f"Dataset length: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=jepa_collate_fn)
    
    print("Fetching one batch...")
    for batch in dataloader:
        mel_specs, text_ids_pad, input_lens = batch
        print(f"Mel Spectrograms shape: {mel_specs.shape}")
        print(f"Text IDs shape: {text_ids_pad.shape}")
        print(f"Input lengths: {input_lens}")
        
        # Verify mel shape is exactly what JEPAT expects: [Batch, 1, 128, 512]
        assert mel_specs.shape[1:] == torch.Size([1, 128, 512]), "Mel Spectrogram shape mismatch!"
        print("Success! The dataset outputs match JEPA-T's expected input dimensions.")
        break

if __name__ == "__main__":
    main()
