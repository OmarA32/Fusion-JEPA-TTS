import torch
from vocoder_manager import VocoderManager

def test_vocoders():
    print("Testing Dual Vocoder Setup...")
    
    # 1. Test HiFi-GAN (expects 80 Mel bins)
    print("\n--- Testing HiFi-GAN ---")
    try:
        manager_hifi = VocoderManager(vocoder_type='hifigan', device='cpu')
        dummy_mel_hifi = torch.randn(1, 80, 200) # [Batch, Mels, Time]
        audio_hifi = manager_hifi.generate_audio(dummy_mel_hifi)
        print(f"HiFi-GAN Output Shape: {audio_hifi.shape}")
        print("HiFi-GAN test passed!")
    except Exception as e:
        print(f"HiFi-GAN test failed: {e}")

    # 2. Test Vocos (expects 100 Mel bins)
    print("\n--- Testing Vocos ---")
    try:
        manager_vocos = VocoderManager(vocoder_type='vocos', device='cpu')
        dummy_mel_vocos = torch.randn(1, 100, 200)
        audio_vocos = manager_vocos.generate_audio(dummy_mel_vocos)
        print(f"Vocos Output Shape: {audio_vocos.shape}")
        print("Vocos test passed!")
    except Exception as e:
        print(f"Vocos test failed: {e}")
        
    # 3. Test BigVGAN (expects 100 Mel bins)
    print("\n--- Testing BigVGAN ---")
    try:
        manager_bigvgan = VocoderManager(vocoder_type='bigvgan', device='cpu')
        dummy_mel_bigvgan = torch.randn(1, 100, 200)
        audio_bigvgan = manager_bigvgan.generate_audio(dummy_mel_bigvgan)
        print(f"BigVGAN Output Shape: {audio_bigvgan.shape}")
        print("BigVGAN test passed!")
    except Exception as e:
        print(f"BigVGAN test failed: {e}")

if __name__ == "__main__":
    test_vocoders()
