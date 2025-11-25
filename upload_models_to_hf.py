#!/usr/bin/env python3
"""
Script to upload large model files to HuggingFace Spaces using the Hub API.
This is an alternative to Git LFS for very large files.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login
from tqdm import tqdm

def upload_file(api, repo_id, local_path, remote_path):
    """Upload a single file to HuggingFace Hub"""
    if not os.path.exists(local_path):
        print(f"⚠️  File not found: {local_path}")
        return False
    
    file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
    print(f"📤 Uploading {local_path} ({file_size:.2f} MB) -> {remote_path}")
    
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="space"
        )
        print(f"✅ Uploaded: {remote_path}")
        return True
    except Exception as e:
        print(f"❌ Error uploading {local_path}: {e}")
        return False

def upload_directory(api, repo_id, local_dir, remote_dir, extensions=None):
    """Upload all files in a directory"""
    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"⚠️  Directory not found: {local_dir}")
        return
    
    files = []
    for root, dirs, filenames in os.walk(local_path):
        for filename in filenames:
            file_path = Path(root) / filename
            if extensions is None or file_path.suffix in extensions:
                rel_path = file_path.relative_to(local_path)
                remote_path = f"{remote_dir}/{rel_path}".replace("\\", "/")
                files.append((str(file_path), remote_path))
    
    print(f"📦 Found {len(files)} files to upload from {local_dir}")
    for local_file, remote_file in tqdm(files, desc="Uploading"):
        upload_file(api, repo_id, local_file, remote_file)

def main():
    print("🚀 MotionGPT Model Uploader to HuggingFace Spaces")
    print("=" * 50)
    
    # Get space ID
    space_id = input("Enter your HuggingFace Space ID (e.g., username/MotionGPT): ").strip()
    if not space_id:
        print("❌ Space ID is required!")
        return
    
    # Login
    print("\n🔐 Logging in to HuggingFace...")
    login()
    
    # Initialize API
    api = HfApi()
    
    # Base directory
    base_dir = Path(__file__).parent
    
    print("\n📦 Uploading files...")
    print("-" * 50)
    
    # 1. Upload checkpoint
    checkpoint_path = base_dir / "checkpoints" / "MotionGPT-base" / "motiongpt_s3_h3d.tar"
    if checkpoint_path.exists():
        upload_file(api, space_id, str(checkpoint_path), 
                   "checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar")
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
    
    # 2. Upload T5 model
    t5_dir = base_dir / "deps" / "flan-t5-base"
    if t5_dir.exists():
        print("\n📤 Uploading T5 model...")
        upload_directory(api, space_id, str(t5_dir), "deps/flan-t5-base")
    else:
        print(f"⚠️  T5 model directory not found: {t5_dir}")
    
    # 3. Upload Whisper model
    whisper_dir = base_dir / "deps" / "whisper-large-v2"
    if whisper_dir.exists():
        print("\n📤 Uploading Whisper model...")
        upload_directory(api, space_id, str(whisper_dir), "deps/whisper-large-v2")
    else:
        print(f"⚠️  Whisper model directory not found: {whisper_dir}")
    
    # 4. Upload SMPL models
    smpl_dir = base_dir / "deps" / "smpl_models"
    if smpl_dir.exists():
        print("\n📤 Uploading SMPL models...")
        upload_directory(api, space_id, str(smpl_dir), "deps/smpl_models",
                        extensions=[".pkl", ".h5", ".npz", ".faces"])
    else:
        print(f"⚠️  SMPL models directory not found: {smpl_dir}")
    
    # 5. Upload assets (excluding generated files)
    assets_dir = base_dir / "assets"
    if assets_dir.exists():
        print("\n📤 Uploading assets...")
        # Only upload specific asset subdirectories
        for subdir in ["css", "images", "videos", "meta"]:
            subdir_path = assets_dir / subdir
            if subdir_path.exists():
                upload_directory(api, space_id, str(subdir_path), f"assets/{subdir}")
    else:
        print(f"⚠️  Assets directory not found: {assets_dir}")
    
    print("\n" + "=" * 50)
    print("✅ Upload complete!")
    print(f"🌐 Check your space: https://huggingface.co/spaces/{space_id}")

if __name__ == "__main__":
    main()




