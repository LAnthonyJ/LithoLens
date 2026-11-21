#!/bin/bash
# =============================================
# Meteor-Mamba 陨石识别 - 服务器安装脚本
# 适用于 Linux GPU 服务器 (Ubuntu/CentOS)
# =============================================

set -e

echo "=========================================="
echo "Meteor-Mamba 环境安装"
echo "=========================================="

# 1. 安装 CUDA PyTorch (根据服务器 CUDA 版本选择)
echo "[1/5] Installing CUDA PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. 安装基础依赖
echo "[2/5] Installing base dependencies..."
pip install transformers accelerate bitsandbytes einops timm
pip install pandas tqdm pillow numpy scikit-learn

# 3. 安装 mamba-ssm 和 causal-conv1d (Linux 原生支持)
echo "[3/5] Installing mamba-ssm and causal-conv1d..."
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation

# 4. 克隆 Meteor 仓库
echo "[4/5] Cloning Meteor repository..."
cd /tmp
if [ -d "Meteor" ]; then
    rm -rf Meteor
fi
git clone https://github.com/ByungKwanLee/Meteor.git
cd Meteor

# 5. 安装 Meteor
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
