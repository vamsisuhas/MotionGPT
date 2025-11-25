#!/usr/bin/env python3
"""
Deploy MotionGPT to HuggingFace Spaces using Python API
This avoids CLI dependency conflicts
"""

import os
import subprocess
import shutil
from pathlib import Path

def run_cmd(cmd, check=True):
    """Run shell command"""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"  Error: {result.stderr}")
    return result

def main():
    print("🚀 Deploying MotionGPT to HuggingFace Spaces...")
    
    # Check HF token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN environment variable not set")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        print("   Then run: export HF_TOKEN=your_token_here")
        return 1
    
    # Space name
    hf_username = os.getenv("HF_USERNAME", "vsadhu1")
    space_name = f"{hf_username}/MotionGPT"
    print(f"📦 Space: {space_name}")
    
    # Use Python API to create space if needed
    try:
        from huggingface_hub import HfApi, login
        api = HfApi()
        
        print("🔐 Logging in to HuggingFace...")
        login(token=hf_token)
        
        # Create space if it doesn't exist
        try:
            api.repo_info(repo_id=space_name, repo_type="space")
            print(f"✅ Space already exists: {space_name}")
        except Exception:
            print(f"📂 Creating new space: {space_name}")
            api.create_repo(
                repo_id=space_name,
                repo_type="space",
                space_sdk="gradio",
                private=False
            )
            print("✅ Space created!")
    except ImportError:
        print("⚠️  huggingface_hub not available, will use git directly")
    except Exception as e:
        print(f"⚠️  Could not create space via API: {e}")
        print("   Will try to clone directly...")
    
    # Clone or update the space
    if os.path.exists("hf_space"):
        print("📂 Using existing hf_space directory...")
        os.chdir("hf_space")
        run_cmd(["git", "pull", "origin", "main"], check=False)
    else:
        print("📂 Cloning space repository...")
        repo_url = f"https://huggingface.co/spaces/{space_name}"
        # Use token in URL for authentication
        auth_url = f"https://{hf_token}@huggingface.co/spaces/{space_name}"
        run_cmd(["git", "clone", auth_url, "hf_space"])
        os.chdir("hf_space")
    
    # Copy necessary files
    print("📋 Copying files...")
    base_dir = Path("..")
    
    # Core files
    for file in ["app.py", "requirements.txt", "README.md"]:
        if (base_dir / file).exists():
            shutil.copy2(base_dir / file, ".")
            print(f"  ✓ {file}")
    
    # Copy directories
    for dir_name in ["mGPT", "configs", "assets"]:
        if (base_dir / dir_name).exists():
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
            shutil.copytree(base_dir / dir_name, dir_name)
            print(f"  ✓ {dir_name}/")
    
    # Skip large model files - they will be downloaded at runtime
    print("⚠️  Skipping large model files (checkpoints, deps)")
    print("   Models will be downloaded from separate HF repos at runtime")
    print("   Make sure to upload models first using: python3 upload_models_separate.py")
    
    # Setup Git LFS
    print("🔧 Setting up Git LFS...")
    run_cmd(["git", "lfs", "install"], check=False)
    
    gitattributes = """*.tar filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.mp4 filter=lfs diff=lfs merge=lfs -text
*.gif filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
"""
    with open(".gitattributes", "w") as f:
        f.write(gitattributes)
    print("  ✓ .gitattributes created")
    
    # Commit and push
    print("💾 Committing changes...")
    run_cmd(["git", "add", "."], check=False)
    run_cmd(["git", "commit", "-m", "Deploy MotionGPT"], check=False)
    
    print("📤 Pushing to HuggingFace...")
    # Configure git to use token in URL
    auth_url = f"https://{hf_token}@huggingface.co/spaces/{space_name}"
    run_cmd(["git", "remote", "set-url", "origin", auth_url], check=False)
    
    # Configure git credential helper to use token automatically
    print("  Configuring Git credentials...")
    run_cmd(["git", "config", "credential.helper", "store"], check=False)
    
    # Store credentials (token as both username and password)
    cred_file = Path.home() / ".git-credentials"
    cred_line = f"https://{hf_token}@huggingface.co\n"
    # Remove old entry if exists
    if cred_file.exists():
        with open(cred_file, "r") as f:
            lines = [l for l in f if "huggingface.co" not in l]
        with open(cred_file, "w") as f:
            f.writelines(lines)
    # Add new entry
    with open(cred_file, "a") as f:
        f.write(cred_line)
    os.chmod(cred_file, 0o600)  # Secure permissions
    
    # Now push (should not prompt for password)
    print("  Pushing to HuggingFace...")
    result = run_cmd(["git", "push", "origin", "main"], check=False)
    if result.returncode != 0:
        print(f"\n⚠️  Git push failed. Error: {result.stderr}")
        print("\n💡 Manual push option:")
        print(f"   1. cd hf_space")
        print(f"   2. git push origin main")
        print(f"   3. When prompted for password, leave it EMPTY (just press Enter)")
        print(f"      The token in the URL is sufficient for authentication.")
        return 1
    
    os.chdir("..")
    
    print("\n✅ Deployment complete!")
    print(f"🌐 Your app will be available at: https://huggingface.co/spaces/{space_name}")
    print("⏳ It may take a few minutes for the space to build and start.")
    return 0

if __name__ == "__main__":
    exit(main())

