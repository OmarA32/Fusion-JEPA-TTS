import os
from huggingface_hub import HfApi, create_repo

def main():
    api = HfApi()
    
    print("Authenticating with Hugging Face...")
    user_info = api.whoami()
    username = user_info["name"]
    print(f"Logged in as: {username}")
    
    repo_id = f"{username}/JEPA-TTS-v4-Arabic"
    print(f"Target repository: {repo_id}")
    
    try:
        create_repo(repo_id, exist_ok=True)
        print("Repository verified.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
        
    checkpoint_path = r"C:\Users\g3m43\.gemini\antigravity\scratch\JEPA-TTS v2\training_logs\arabic\nawar_halabi\best-epoch=001.ckpt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find checkpoint at {checkpoint_path}")
        return
        
    print(f"Uploading {checkpoint_path}...")
    api.upload_file(
        path_or_fileobj=checkpoint_path,
        path_in_repo="best-epoch=001.ckpt",
        repo_id=repo_id
    )
    print("Upload complete! 🚀")
    print(f"Model available at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
