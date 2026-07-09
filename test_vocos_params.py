import torch
from vocos import Vocos

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

print("Vocos Feature Extractor:")
print(vocos.feature_extractor)
