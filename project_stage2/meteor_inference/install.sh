#!/bin/bash
# =============================================
# Meteor-Mamba meteorite classification server setup
# For Linux GPU servers (Ubuntu/CentOS)
# =============================================

set -e

echo "=========================================="
echo "Meteor-Mamba Environment setup"
echo "=========================================="

# 1. Install PyTorch for the server CUDA version (select the matching build)
echo "[1/5] Installing CUDA PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. Install base dependencies
echo "[2/5] Installing base dependencies..."
pip install transformers accelerate bitsandbytes einops timm
pip install pandas tqdm pillow numpy scikit-learn

# 3. Install mamba-ssm and causal-conv1d (native Linux support)
echo "[3/5] Installing mamba-ssm and causal-conv1d..."
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation

# 4. Clone the Meteor repository
echo "[4/5] Cloning Meteor repository..."
cd /tmp
if [ -d "Meteor" ]; then
    rm -rf Meteor
fi
git clone https://github.com/ByungKwanLee/Meteor.git
cd Meteor

# 5. Install Meteor
echo "[5/5] Installing Meteor package..."
pip install -e .

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Verify installation:"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
echo ""
echo "Next: Run 'bash run.sh' to start inference"
