import torch
try:
    print(f"XPU Available: {torch.xpu.is_available()}")
except AttributeError:
    print("XPU Available: False (The 'torch.xpu' module is not installed or supported in this PyTorch version)")
