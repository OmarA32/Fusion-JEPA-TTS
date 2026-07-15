import json

with open('supercomputer_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the index of the "### 3.5 Download Arabic Speech Corpus" markdown cell
index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and len(cell['source']) > 0 and '3.5 Download Arabic' in cell['source'][0]:
        index = i
        break

if index != -1:
    # Delete the old 3.5, code, 4A, code, 4B, code (6 cells)
    del nb['cells'][index:index+8]
    
    # Insert new cells
    new_cells = [
        {
          "cell_type": "markdown",
          "metadata": {},
          "source": [
            "### 3.5 Download Arabic Datasets\n",
            "Downloads Nawar Halabi's high-quality single-speaker dataset to be used for fine-tuning. Common Voice downloads automatically via HuggingFace during training."
          ]
        },
        {
          "cell_type": "code",
          "execution_count": None,
          "metadata": {},
          "outputs": [],
          "source": [
            "!mkdir -p data\n",
            "!wget -nc https://en.arabicspeechcorpus.com/arabic-speech-corpus.zip -O data/arabic-speech-corpus.zip\n",
            "!unzip -q -o data/arabic-speech-corpus.zip -d data/arabic-speech-corpus\n",
            "!rm data/arabic-speech-corpus.zip"
          ]
        },
        {
          "cell_type": "markdown",
          "metadata": {},
          "source": [
            "### 3.6 Download English Datasets\n",
            "Downloads LJSpeech (single speaker) and LibriTTS (multi-speaker) via torchaudio."
          ]
        },
        {
          "cell_type": "code",
          "execution_count": None,
          "metadata": {},
          "outputs": [],
          "source": [
            "import torchaudio\n",
            "torchaudio.datasets.LJSPEECH(\"./data\", download=True)\n",
            "torchaudio.datasets.LIBRITTS(\"./data\", url=\"train-clean-100\", download=True)"
          ]
        },
        {
          "cell_type": "markdown",
          "metadata": {},
          "source": [
            "### 4A. Run Pre-Training\n",
            "Train on Common Voice Arabic."
          ]
        },
        {
          "cell_type": "code",
          "execution_count": None,
          "metadata": {},
          "outputs": [],
          "source": [
            "!python train.py --lang arabic --db common_voice"
          ]
        },
        {
          "cell_type": "markdown",
          "metadata": {},
          "source": [
            "### 4B. Run Fine-Tuning\n",
            "Fine tune on Nawar Halabi (Arabic) or LJSpeech (English)."
          ]
        },
        {
          "cell_type": "code",
          "execution_count": None,
          "metadata": {},
          "outputs": [],
          "source": [
            "!python train.py --lang arabic --db nawar_halabi --resume"
          ]
        }
    ]
    
    # Also update inference cell
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and len(cell['source']) > 0 and cell['source'][0].startswith('!python inference.py'):
            if '--text' not in cell['source'][0]:
                cell['source'] = ['!python inference.py --lang arabic --db nawar_halabi']
            else:
                cell['source'] = ['!python inference.py --lang arabic --db nawar_halabi --text "\u0623\u064a \u0646\u0635 \u0639\u0631\u0628\u064a \u062a\u0631\u064a\u062f" --output "my_custom_audio.wav"']
                
    # Also update the git clone checkout branch to V3
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and len(cell['source']) > 0:
            for i, line in enumerate(cell['source']):
                if '!git checkout' in line:
                    cell['source'][i] = '!git checkout prototype/v3.0.0\n'

    nb['cells'][index:index] = new_cells

with open('supercomputer_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
