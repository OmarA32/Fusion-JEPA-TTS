$ErrorActionPreference = 'Stop'

Write-Host "1. Installing huggingface_hub and logging in..."
pip install -U "huggingface_hub[cli]"
# Log into huggingface to get the dataset
huggingface-cli login --token "YOUR_HF_TOKEN"

Write-Host "2. Cloning repositories..."
if (-Not (Test-Path "Audio-JEPA")) {
    git clone https://github.com/LudovicTuncay/Audio-JEPA.git
}
if (-Not (Test-Path "tts-arabic-pytorch")) {
    git clone https://github.com/nipponjo/tts-arabic-pytorch.git
}

Write-Host "3. Installing espeak-ng via winget..."
# The --accept-package-agreements --accept-source-agreements prevents interactive prompts
winget install --id eSpeak-NG.eSpeak-NG -e --accept-package-agreements --accept-source-agreements

Write-Host "4. Setting up Python virtual environment and dependencies..."
if (-Not (Test-Path "venv")) {
    python -m venv venv
}
.\venv\Scripts\Activate.ps1

pip install --upgrade pip wheel cmake
pip install datasets librosa torch torchaudio torchvision
pip install camel-tools

Write-Host "5. Downloading CAMeL Tools data..."
camel_data -i light

Write-Host "Setup Complete!"
