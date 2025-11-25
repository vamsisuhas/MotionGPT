#!/usr/bin/env python3
"""
Quick push for requirements.txt update
"""

import os
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
        return 1
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Error: HF_TOKEN environment variable not set")
        return 1
    
    hf_username = os.getenv("HF_USERNAME", "vsadhu1")
    space_name = f"{hf_username}/MotionGPT"
    
    print("📤 Quick push: requirements.txt update...")
    
    os.chdir("hf_space")
    
    # Copy updated requirements.txt
    if os.path.exists("../requirements.txt"):
        import shutil
        shutil.copy2("../requirements.txt", "requirements.txt")
        print("  ✓ Updated requirements.txt")
    
    # Configure git
    auth_url = f"https://{hf_username}:{hf_token}@huggingface.co/spaces/{space_name}"
    run_cmd(["git", "remote", "set-url", "origin", auth_url])
    
    # Commit and push
    run_cmd(["git", "add", "requirements.txt"])
    run_cmd(["git", "commit", "-m", "Fix chumpy build dependencies"])
    run_cmd(["git", "push", "origin", "main"])
    
    os.chdir("..")
    print("\n✅ Push complete!")
    return 0

if __name__ == "__main__":
    exit(main())




