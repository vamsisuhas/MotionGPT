#!/usr/bin/env python3
"""
Upload entire folders to HuggingFace Spaces to reduce commit count.
Also handles the storage limit issue by referencing models from Hub.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login
from tqdm import tqdm
import time

def upload_folder(api, repo_id, local_dir, remote_dir):
    """Upload an entire folder at once (creates single commit)"""
    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"⚠️  Directory not found: {local_dir}")
        return False
    
    print(f"📤 Uploading folder: {local_dir} -> {remote_dir}")
    try:
        api.upload_folder(
            folder_path=str(local_path),
            path_in_repo=remote_dir,
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=[".git", "__pycache__", "*.pyc", "*.pyo"]
        )
        print(f"✅ Uploaded folder: {remote_dir}")
        return True
    except Exception as e:
        print(f"❌ Error uploading {local_dir}: {e}")
        return False

def main():
    print("🚀 MotionGPT Folder Upload to HuggingFace Spaces")
    print("=" * 60)
    
    space_id = "vsadhu1/MotionGPT"
    token = "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"
    
    print(f"📁 Space: {space_id}")
    print("\n⚠️  IMPORTANT: Free HuggingFace Spaces have 1GB storage limit")
    print("   Your models exceed this. Solutions:")
    print("   1. Use HuggingFace Hub to store models separately")
    print("   2. Upgrade to paid plan")
    print("   3. Use smaller models or compress")
    print()
    
    # Login
    print("🔐 Logging in...")
    login(token=token)
    
    # Initialize API
    api = HfApi()
    
    base_dir = Path("hf_space")
    if not base_dir.exists():
        print(f"❌ hf_space directory not found!")
        return
    
    print("\n📦 Uploading folders (this reduces commit count)...")
    print("-" * 60)
    
    # Upload folders one at a time with delays to avoid rate limits
    folders_to_upload = [
        ("app.py", "app.py", "file"),
        ("requirements.txt", "requirements.txt", "file"),
        ("README.md", "README.md", "file"),
        ("configs", "configs", "folder"),
        ("mGPT", "mGPT", "folder"),
        ("assets/css", "assets/css", "folder"),
        ("assets/images", "assets/images", "folder"),
        ("assets/meta", "assets/meta", "folder"),
    ]
    
    # Skip large model files - they exceed storage limit
    print("\n⚠️  Skipping large model files (exceed 1GB limit):")
    print("   - checkpoints/ (1.3GB)")
    print("   - deps/flan-t5-base/ (~990MB)")
    print("   - deps/whisper-large-v2/ (large)")
    print("   - deps/smpl_models/ (large)")
    print("\n💡 Solution: Store models in separate HuggingFace Hub repos")
    print("   and download them at runtime in app.py")
    
    for local_item, remote_item, item_type in folders_to_upload:
        local_path = base_dir / local_item
        if not local_path.exists():
            print(f"⚠️  Skipping {local_item} (not found)")
            continue
        
        if item_type == "file":
            try:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=remote_item,
                    repo_id=space_id,
                    repo_type="space"
                )
                print(f"✅ Uploaded: {remote_item}")
            except Exception as e:
                print(f"⚠️  Failed {remote_item}: {e}")
        else:
            upload_folder(api, space_id, str(local_path), remote_item)
        
        # Small delay to avoid rate limits
        time.sleep(2)
    
    print("\n" + "=" * 60)
    print("✅ Upload complete!")
    print(f"🌐 Check your space: https://huggingface.co/spaces/{space_id}")
    print("\n📝 Next steps:")
    print("   1. Store models in separate HuggingFace Hub repositories")
    print("   2. Modify app.py to download models at runtime")
    print("   3. Or upgrade to paid plan for more storage")

if __name__ == "__main__":
    main()

