#!/usr/bin/env python3
"""
Push existing files in hf_space to HuggingFace Spaces
"""

import os
import shutil
import subprocess
from pathlib import Path

def run_cmd(cmd, cwd=None):
    """Run shell command"""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"  Error: {result.stderr}")
    return result

def main():
    if not os.path.exists("hf_space"):
        print("❌ Error: hf_space directory not found")
        print("   Run deploy_to_hf.py first to copy files")
        return 1
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN environment variable not set")
        print("   Run: export HF_TOKEN=your_token_here")
        return 1
    
    hf_username = os.getenv("HF_USERNAME", "vsadhu1")
    space_name = f"{hf_username}/MotionGPT"
    
    print("📤 Pushing files to HuggingFace Spaces...")
    print(f"📦 Space: {space_name}")
    
    os.chdir("hf_space")
    
    # Configure git remote with token (use username:token format)
    # HuggingFace requires: https://USERNAME:TOKEN@huggingface.co/...
    auth_url = f"https://{hf_username}:{hf_token}@huggingface.co/spaces/{space_name}"
    print("🔧 Configuring Git remote...")
    run_cmd(["git", "remote", "set-url", "origin", auth_url])
    
    # Configure credential helper
    print("🔧 Configuring Git credentials...")
    run_cmd(["git", "config", "credential.helper", "store"])
    
    # Store credentials (username:token format)
    cred_file = Path.home() / ".git-credentials"
    cred_line = f"https://{hf_username}:{hf_token}@huggingface.co\n"
    if cred_file.exists():
        with open(cred_file, "r") as f:
            lines = [l for l in f if "huggingface.co" not in l]
        with open(cred_file, "w") as f:
            f.writelines(lines)
    with open(cred_file, "a") as f:
        f.write(cred_line)
    os.chmod(cred_file, 0o600)
    
    # Remove large files and binary images (they'll be downloaded at runtime or not needed)
    print("🧹 Removing large model files and binary images...")
    large_paths = [
        "checkpoints",
        "deps"
    ]
    
    # Remove binary PNG images (HuggingFace requires Xet storage for binaries)
    print("  Removing binary PNG images...")
    image_files = [
        "assets/images/figure10.png",
        "assets/images/figure12.png",
        "assets/images/figure13.png",
        "assets/images/pipeline.png",
        "assets/images/table15.png",
        "assets/images/table7.png",
        "assets/images/table8.png"
    ]
    for img_file in image_files:
        if os.path.exists(img_file):
            result = run_cmd(["git", "rm", "--cached", img_file])
            if result.returncode == 0:
                os.remove(img_file)
                print(f"  ✓ Removed {img_file}")
            elif os.path.exists(img_file):
                # If not in git, just remove the file
                os.remove(img_file)
                print(f"  ✓ Removed {img_file} (not in git)")
    
    # First, remove from Git LFS if tracked
    print("  Removing from Git LFS...")
    for path in large_paths:
        if os.path.exists(path):
            # Remove from Git LFS
            run_cmd(["git", "lfs", "untrack", path])
            # Remove from git cache
            result = run_cmd(["git", "rm", "-r", "--cached", path])
            if result.returncode != 0:
                print(f"  ⚠️  Could not remove {path} from git cache (may not be tracked)")
            # Remove physical files
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            print(f"  ✓ Removed {path}/")
    
    # Clean Git LFS cache
    print("  Cleaning Git LFS cache...")
    run_cmd(["git", "lfs", "prune"])
    
    # Force garbage collection to remove large files from history
    print("  Cleaning Git history...")
    run_cmd(["git", "reflog", "expire", "--expire=now", "--all"])
    run_cmd(["git", "gc", "--prune=now", "--aggressive"])
    
    # Add, commit
    print("💾 Staging changes...")
    run_cmd(["git", "add", "."])
    
    print("💾 Committing changes...")
    result = run_cmd(["git", "commit", "-m", "Deploy MotionGPT - models downloaded at runtime"])
    if result.returncode != 0 and "nothing to commit" not in result.stderr.lower():
        print(f"  Warning: Commit may have failed: {result.stderr}")
    
    # Create a fresh branch without history to avoid size limit
    print("🔄 Creating fresh branch without large file history...")
    result = run_cmd(["git", "checkout", "--orphan", "fresh"])
    if result.returncode == 0:
        # Remove PNG files before adding (they're already removed, but double-check)
        png_files = [
            "assets/images/figure10.png",
            "assets/images/figure12.png",
            "assets/images/figure13.png",
            "assets/images/pipeline.png",
            "assets/images/table15.png",
            "assets/images/table7.png",
            "assets/images/table8.png"
        ]
        for img_file in png_files:
            if os.path.exists(img_file):
                os.remove(img_file)
        run_cmd(["git", "add", "."])
        run_cmd(["git", "commit", "-m", "Initial commit - MotionGPT without large models"])
        run_cmd(["git", "branch", "-D", "main"])
        run_cmd(["git", "branch", "-m", "main"])
    else:
        print("  ⚠️  Could not create orphan branch, continuing with current branch...")
    
    print("📤 Pushing to HuggingFace...")
    # Approve credentials using git credential (username:token format)
    print("  Approving Git credentials...")
    credential_input = f"url=https://huggingface.co\nusername={hf_username}\npassword={hf_token}\n\n"
    approve_process = subprocess.Popen(
        ["git", "credential", "approve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    approve_process.communicate(input=credential_input)
    
    # Force push to replace history (needed to remove large files from history)
    print("  ⚠️  Using force push to replace repository history...")
    result = run_cmd(["git", "push", "--force", "origin", "main"])
    
    os.chdir("..")
    
    if result.returncode == 0:
        print("\n✅ Push complete!")
        print(f"🌐 Your app will be available at: https://huggingface.co/spaces/{space_name}")
        print("⏳ It may take a few minutes for the space to build and start.")
        return 0
    else:
        print("\n⚠️  Push failed. Try manually:")
        print(f"   cd hf_space")
        print(f"   git push origin main")
        print(f"   (When prompted for password, leave it EMPTY - just press Enter)")
        return 1

if __name__ == "__main__":
    exit(main())
