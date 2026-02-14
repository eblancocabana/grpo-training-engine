#!/bin/bash
# scripts/install_dependencies.sh

echo "🔧 Setting up environment for RTX 3060 Ti..."

# Ensure conda is active
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "❌ Error: Activate conda environment first: conda activate grpo-3060ti"
    exit 1
fi

echo "📦 Installing PyTorch with CUDA 12.1 support..."
# It is crucial to force CUDA 12.1 URL for bitsandbytes compatibility
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "📦 Installing base libraries..."
# accelerate: Device map management
# bitsandbytes: Quantization
# scipy: Entropy
# datasets: GSM8K loading
pip install transformers accelerate bitsandbytes scipy numpy datasets

echo "✅ Installation completed. Run 'python src/core/model_loader.py' to test."
