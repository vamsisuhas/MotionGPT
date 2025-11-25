#!/usr/bin/env python3
"""
Upload large model files to separate HuggingFace Hub repositories
This avoids the 1GB limit on HuggingFace Spaces
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, login

def upload_folder(api, repo_id, local_dir, repo_type="model"):
    """Upload a folder to HuggingFace Hub"""
    print(f"📤 Uploading {local_dir} to {repo_id}...")
    
    if not os.path.exists(local_dir):
        print(f"  ⚠️  Directory {local_dir} does not exist, skipping...")
        return False
    
    try:
        # Create repo if it doesn't exist
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            print(f"  ✓ Repository {repo_id} already exists")
        except Exception:
            print(f"  📦 Creating repository {repo_id}...")
            api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
            print(f"  ✓ Repository created")
        
        # Upload folder
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type=repo_type,
            ignore_patterns=[".git*", "__pycache__", "*.pyc"]
        )
        print(f"  ✅ Successfully uploaded {local_dir} to {repo_id}")
        return True
    except Exception as e:
        print(f"  ❌ Error uploading {local_dir}: {e}")
        return False

def main():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN environment variable not set")
        print("   Run: export HF_TOKEN=your_token_here")
        return 1
    
    hf_username = os.getenv("HF_USERNAME", "vsadhu1")
    
    print("🚀 Uploading models to separate HuggingFace Hub repositories...")
    print(f"👤 Username: {hf_username}\n")
    
    # Login
    login(token=hf_token)
    api = HfApi()
    
    # Upload checkpoint
    checkpoint_path = "checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar"
    if os.path.exists(checkpoint_path):
        repo_id = f"{hf_username}/MotionGPT-checkpoint"
        # Create a temp directory with just the checkpoint
        temp_dir = Path("/tmp/motiongpt_checkpoint")
        temp_dir.mkdir(exist_ok=True)
        shutil.copy2(checkpoint_path, temp_dir / "motiongpt_s3_h3d.tar")
        upload_folder(api, repo_id, str(temp_dir))
        shutil.rmtree(temp_dir)
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
    
    # Upload T5 model
    t5_path = "deps/flan-t5-base"
    if os.path.exists(t5_path):
        repo_id = f"{hf_username}/MotionGPT-t5-base"
        upload_folder(api, repo_id, t5_path)
    else:
        print(f"⚠️  T5 model not found: {t5_path}")
    
    # Upload Whisper model
    whisper_path = "deps/whisper-large-v2"
    if os.path.exists(whisper_path):
        repo_id = f"{hf_username}/MotionGPT-whisper-large-v2"
        upload_folder(api, repo_id, whisper_path)
    else:
        print(f"⚠️  Whisper model not found: {whisper_path}")
    
    # Upload SMPL models
    smpl_path = "deps/smpl_models"
    if os.path.exists(smpl_path):
        repo_id = f"{hf_username}/MotionGPT-smpl-models"
        upload_folder(api, repo_id, smpl_path)
    else:
        print(f"⚠️  SMPL models not found: {smpl_path}")
    
    print("\n✅ Model upload complete!")
    print("\n📝 Next steps:")
    print("   1. Update app.py to download models at runtime")
    print("   2. Deploy to HuggingFace Spaces (without large files)")
    return 0

if __name__ == "__main__":
    exit(main())




