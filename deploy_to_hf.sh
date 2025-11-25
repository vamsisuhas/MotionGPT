#!/bin/bash
# Deploy MotionGPT to HuggingFace Spaces
# This script calls the Python deployment script to avoid CLI dependency conflicts

set -e

echo "🚀 Deploying MotionGPT to HuggingFace Spaces..."
echo "📥 Using Python API (no CLI dependencies required)"

# Check if HF token is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN environment variable not set"
    echo "   Get your token from: https://huggingface.co/settings/tokens"
    echo "   Then run: export HF_TOKEN=your_token_here"
    exit 1
fi

# Run Python deployment script
python3 deploy_to_hf.py
