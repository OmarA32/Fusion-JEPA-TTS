import nbformat

def fully_restore_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 1. Strip the old clunky upload cell
    filtered_cells = []
    skip_next = False
    for cell in nb.cells:
        if "8. Upload Weights to Hugging Face" in cell.source:
            skip_next = True 
            continue
        if skip_next and "api.upload_folder" in cell.source:
            skip_next = False
            continue
        filtered_cells.append(cell)
    nb.cells = filtered_cells

    # 2. Build the Download Cells
    arabic_download_md = nbformat.v4.new_markdown_cell("### ⬇️ Download Arabic Model\nDownloads the specific `.ckpt` file you specify from your Hugging Face repository.")
    arabic_download_code = nbformat.v4.new_code_cell("""\
!pip install huggingface_hub

from huggingface_hub import hf_hub_download
import os

ARABIC_REPO = "KASP-JEPA/Project-Arabic"
ARABIC_FILENAME = "best-epoch=000.ckpt" # <-- Type exactly which file you want to download here!

print(f"Downloading {ARABIC_FILENAME}...")
os.makedirs("training_logs/arabic/nawar_halabi", exist_ok=True)
hf_hub_download(
    repo_id=ARABIC_REPO,
    filename=ARABIC_FILENAME,
    local_dir="training_logs/arabic/nawar_halabi"
)
print("Arabic model downloaded successfully!")
""")

    english_download_md = nbformat.v4.new_markdown_cell("### ⬇️ Download English Model\nSame as above, but for the English model.")
    english_download_code = nbformat.v4.new_code_cell("""\
# from huggingface_hub import hf_hub_download
# import os

# ENGLISH_REPO = "KASP-JEPA/Project-English"
# ENGLISH_FILENAME = "best-epoch=000.ckpt" # <-- Type exactly which file you want to download here!

# print(f"Downloading {ENGLISH_FILENAME}...")
# os.makedirs("training_logs/english/ljspeech", exist_ok=True)
# hf_hub_download(
#     repo_id=ENGLISH_REPO,
#     filename=ENGLISH_FILENAME,
#     local_dir="training_logs/english/ljspeech"
# )
# print("English model downloaded successfully!")
""")

    # 3. Build the Upload Cells
    upload_md = nbformat.v4.new_markdown_cell("## ⬆️ Model Publishing")
    upload_code = nbformat.v4.new_code_cell("""\
# 🚀 Optional: Upload Best Models to Hugging Face
# You can upload your best trained models to the Hugging Face Hub so you can download them later!
# Get your Hugging Face write token from: https://huggingface.co/settings/tokens
!pip install huggingface_hub

HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxx" # Replace with your real token!

# --- 🟢 UPLOAD ARABIC MODEL ---
ARABIC_REPO = "KASP-JEPA/Project-Arabic" 
ARABIC_CKPT = "training_logs/arabic/nawar_halabi/best-epoch=000.ckpt" # Change 000 to your best Arabic epoch!

!python upload_to_hf.py --ckpt {ARABIC_CKPT} --repo {ARABIC_REPO} --token {HF_TOKEN}

# --- 🔵 UPLOAD ENGLISH MODEL (Uncomment when ready) ---
# ENGLISH_REPO = "KASP-JEPA/Project-English"
# ENGLISH_CKPT = "training_logs/english/ljspeech/best-epoch=000.ckpt" # Change 000 to your best English epoch!

# !python upload_to_hf.py --ckpt {ENGLISH_CKPT} --repo {ENGLISH_REPO} --token {HF_TOKEN}
""")

    # Append everything to the end
    nb.cells.extend([
        nbformat.v4.new_markdown_cell("## ⬇️ Download Pre-Trained Weights\nUse these cells to download your best models from Hugging Face back into your supercomputer when you start a new session."),
        arabic_download_md, arabic_download_code, 
        english_download_md, english_download_code, 
        upload_md, upload_code
    ])

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
        
    print("Notebook fully restored and updated to KASP-JEPA!")

if __name__ == "__main__":
    fully_restore_notebook("supercomputer_training.ipynb")
