import os
import sys
import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
from scipy.spatial.distance import cdist

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Setup UTMOS predictor
print("Loading UTMOS model...")
try:
    utmos_model = torch.hub.load("tarepan/SpeechMOS:main", "utmos22_strong", trust_repo=True)
    utmos_model.eval()
    if torch.cuda.is_available():
        utmos_model = utmos_model.cuda()
    print("UTMOS loaded successfully.")
except Exception as e:
    print(f"Warning: could not load UTMOS: {e}")
    utmos_model = None

def compute_utmos(audio_path):
    if utmos_model is None:
        return np.nan
    try:
        wav_np, sr = sf.read(audio_path)
        if len(wav_np.shape) > 1:
            wav_np = wav_np.mean(axis=1)
        if sr != 16000:
            wav_np = librosa.resample(wav_np, orig_sr=sr, target_sr=16000)
        
        wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0)
        device = next(utmos_model.parameters()).device
        wav_tensor = wav_tensor.to(device)
        with torch.no_grad():
            score = utmos_model(wav_tensor, torch.tensor([16000], device=device)).item()
        return float(score)
    except Exception as e:
        print(f"Error computing UTMOS for {audio_path}: {e}")
        return np.nan

def extract_mel(wav, sr=44100, n_fft=1024, hop_length=256, n_mels=128):
    mel = librosa.feature.melspectrogram(
        y=wav, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels, 
        fmin=0, 
        fmax=8000
    )
    log_mel = np.log(np.maximum(mel, 1e-5))
    return log_mel # [n_mels, T]

def align_and_compute_spectral_metrics(ref_path, pred_path):
    # Load audio
    wav_ref, sr_ref = sf.read(ref_path)
    wav_pred, sr_pred = sf.read(pred_path)
    
    # Resample if needed
    if sr_ref != 44100:
        wav_ref = librosa.resample(wav_ref, orig_sr=sr_ref, target_sr=44100)
    if sr_pred != 44100:
        wav_pred = librosa.resample(wav_pred, orig_sr=sr_pred, target_sr=44100)
        
    mel_ref = extract_mel(wav_ref, sr=44100) # [128, T_ref]
    mel_pred = extract_mel(wav_pred, sr=44100) # [128, T_pred]
    
    # DTW alignment along time axis
    # Distance matrix between frames: [T_ref, T_pred]
    dist_mat = cdist(mel_ref.T, mel_pred.T, metric='cosine')
    
    # Fast DTW path extraction via librosa / scipy
    D, wp = librosa.sequence.dtw(C=dist_mat)
    # wp is array of aligned index pairs: [[idx_ref_0, idx_pred_0], ...]
    
    aligned_ref = mel_ref[:, wp[:, 0]]
    aligned_pred = mel_pred[:, wp[:, 1]]
    
    # 1. L1 (MAE)
    l1_mae = np.mean(np.abs(aligned_ref - aligned_pred))
    
    # 2. L2 (MSE)
    l2_mse = np.mean((aligned_ref - aligned_pred) ** 2)
    
    # 3. Spectral Convergence (SConv)
    # SConv = norm(ref - pred, 'fro') / norm(ref, 'fro')
    sconv = np.linalg.norm(aligned_ref - aligned_pred, 'fro') / (np.linalg.norm(aligned_ref, 'fro') + 1e-8)
    
    return float(l1_mae), float(l2_mse), float(sconv)

def main():
    base_dir = os.path.join(PROJECT_ROOT, "test_results")
    
    tasks = [
        ("Arabic (Nawar Halabi)", os.path.join(base_dir, "arabic"), [108, 545, 999]),
        ("English (LJSpeech)", os.path.join(base_dir, "english"), [100, 500, 1000])
    ]
    
    print("\n" + "="*80)
    print("FUSION-JEPA TTS: OBJECTIVE & PERCEPTUAL EVALUATION RESULTS")
    print("="*80)
    
    summary_results = []
    
    for lang_name, folder, indices in tasks:
        print(f"\n--- Evaluating {lang_name} in {folder} ---")
        l1_list, l2_list, sconv_list, utmos_gt_list, utmos_gen_list = [], [], [], [], []
        
        for idx in indices:
            ref_path = os.path.join(folder, f"ground_truth_index_{idx}_bigvgan.wav")
            pred_path = os.path.join(folder, f"inference_index_{idx}.wav")
            
            if not os.path.exists(ref_path) or not os.path.exists(pred_path):
                print(f"Skipping index {idx}: files not found.")
                continue
                
            l1, l2, sconv = align_and_compute_spectral_metrics(ref_path, pred_path)
            utmos_gt = compute_utmos(ref_path)
            utmos_gen = compute_utmos(pred_path)
            
            l1_list.append(l1)
            l2_list.append(l2)
            sconv_list.append(sconv)
            utmos_gt_list.append(utmos_gt)
            utmos_gen_list.append(utmos_gen)
            
            print(f"Index {idx:4d} | L1 (MAE): {l1:.4f} | L2 (MSE): {l2:.4f} | SConv: {sconv:.4f} | UTMOS (GT): {utmos_gt:.2f} | UTMOS (Gen): {utmos_gen:.2f}")
            
        mean_l1 = np.mean(l1_list)
        mean_l2 = np.mean(l2_list)
        mean_sconv = np.mean(sconv_list)
        mean_utmos_gt = np.mean(utmos_gt_list)
        mean_utmos_gen = np.mean(utmos_gen_list)
        
        print(f"--> {lang_name} MEAN | L1: {mean_l1:.4f} | L2: {mean_l2:.4f} | SConv: {mean_sconv:.4f} | UTMOS (Gen): {mean_utmos_gen:.2f} (GT: {mean_utmos_gt:.2f})")
        
        summary_results.append({
            "language": lang_name,
            "L1": mean_l1,
            "L2": mean_l2,
            "SConv": mean_sconv,
            "UTMOS_Gen": mean_utmos_gen,
            "UTMOS_GT": mean_utmos_gt
        })
        
    print("\n" + "="*80)
    print("FINAL SUMMARY TABLE")
    print("="*80)
    print(f"{'Model / Language':<28} | {'Mel L1 (MAE)':<14} | {'Mel L2 (MSE)':<14} | {'SConv':<10} | {'UTMOS (1-5)':<12}")
    print("-" * 85)
    for res in summary_results:
        print(f"Fusion-JEPA ({res['language']:<15}) | {res['L1']:<14.4f} | {res['L2']:<14.4f} | {res['SConv']:<10.4f} | {res['UTMOS_Gen']:<12.2f}")

if __name__ == "__main__":
    main()
