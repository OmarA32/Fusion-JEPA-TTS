# Prototype Version 7: FastPitch + GAN Generator Integration

We will build a new prototype that replaces the iteration-heavy Diffusion Transformer (`SpatialDiT`) with the one-shot, GAN-augmented FastPitch generator from the `nipponjo/tts-arabic-pytorch` repository. 

This will result in a much faster inference speed while utilizing the adversarial loss to combat oversmoothing and maintain high fidelity.

## Architecture Decision
As confirmed, we will attach the new FastPitch generator **on top of our existing JEPA backbone**. 
1. The JEPA backbone will process the input text and output contextual text embeddings.
2. An Alignment Network will align these text embeddings to the target audio.
3. A Duration Predictor will learn to predict these alignments.
4. The FastPitch Decoder will expand the JEPA text embeddings based on the durations and predict the final mel-spectrogram in a single forward pass.
5. The `SpectrogramDiscriminator` will critique the output to ensure high-frequency textures are preserved.

## Proposed Changes

We will work on a new branch `prototype/v7.1.0`.

### 1. Model Extraction
#### [NEW] `models/fastpitch.py`
- Extract the FastPitch decoder architecture and the Alignment Network from `nipponjo_tts/models/fastpitch/networks.py`.
- Ensure it accepts the dimensionality of our JEPA embeddings rather than standard phoneme embeddings.
- Extract the `SpectrogramDiscriminator` from `nipponjo_tts/scripts/train_fp_adv.py`.

### 2. Loss Functions
#### [NEW] `models/adversarial_loss.py`
- Implement the LS-GAN objective:
  - Generator Loss: $L_G = (D(S_{pred}) - 1)^2$
  - Discriminator Loss: $L_D = \frac{1}{2}(D(S_{ref}) - 1)^2 + \frac{1}{2}(D(S_{pred}))^2$
  - Feature Matching Loss (MAE on discriminator intermediate layers).
- Implement the standard FastPitch L2 Mel, Pitch, Energy, and Duration losses.
- Implement the Alignment Binarization loss required to train the duration predictor.

### 3. Pipeline Integration
#### [NEW] `models/jepagan_lightning.py`
- Create a new LightningModule that encapsulates the JEPA Backbone + FastPitch Decoder + Discriminator.
- Implement `training_step` with manual optimization to alternate between `optimizer_g` (Generator: JEPA + FastPitch) and `optimizer_d` (Discriminator).
- Calculate and log the complex suite of FastPitch losses (Mel, Pitch, Duration, Alignment, Adversarial).

### 4. Data Loading
#### [MODIFY] `train.py`
- Update the training script to use the new `JEPAGANLightning` model.
- Modify `JEPADataset` to extract frame-level Pitch ($f_0$) and Energy contours since FastPitch requires them for prosodic conditioning.

## Verification Plan

### Automated Tests
- Run `python train.py --lang english --db ljspeech` on CPU for 2 batches to verify:
  1. The JEPA embeddings successfully pass through the Alignment Network.
  2. The FastPitch Decoder generates a correctly shaped spectrogram.
  3. Both Generator and Discriminator gradients are calculated properly without mode collapse.

### Manual Verification
- Deploy to Ibex A100 to ensure the dual-optimizer (GAN) setup trains stably.
