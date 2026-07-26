# Training/Inference Collision Rule
When working on the JEPA-TTS project, NEVER run the inference generation script (`inference.py`) while the training script (`train_xpu.py`) is actively running. 

Running inference on the GPU while the massive training loop is active can crash the process via Out-Of-Memory (OOM) or severely bottleneck the training. ALWAYS ensure the training task is fully terminated and the weights are permanently saved to the disk before attempting to run inference tests.
