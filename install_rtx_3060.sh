#!/bin/bash
# install_rtx_3060_fixed.sh

echo "🎯 Installing DeepSeek-OCR for RTX 3060 12GB..."

# First, let's detect all available GPUs
echo "=== Detecting GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Find the RTX 3060 index
RTX_INDEX=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | grep -i "3060" | cut -d',' -f1 | tr -d ' ')
if [ -z "$RTX_INDEX" ]; then
    echo "❌ RTX 3060 not found via nvidia-smi"
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name --format=csv,noheader
    exit 1
fi

echo "✅ Found RTX 3060 at index: $RTX_INDEX"
export CUDA_VISIBLE_DEVICES=$RTX_INDEX

# Use the full path to conda activate
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate deepseek-ocr

# Check directory
if [ ! -f "setup.py" ]; then
    echo "❌ Please run from repository root with setup.py"
    exit 1
fi

echo "Installing PyTorch for RTX 3060..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

echo "Installing other dependencies..."
pip install "numpy<2.0.0"
pip install transformers==4.35.0 Pillow>=10.0.0 opencv-python>=4.8.0 
pip install requests>=2.31.0 tqdm>=4.66.0 gputil>=1.4.0 psutil>=5.9.0

# Fix the setup.py first
cat > setup.py << 'EOF'
import os
from setuptools import setup, find_packages

setup(
    name="deepseek-ocr-hllset",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "numpy<2.0.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "requests>=2.31.0",
        "tqdm>=4.66.0",
        "gputil>=1.4.0",
        "psutil>=5.9.0",
    ],
)
EOF

echo "Installing package..."
pip install .

echo "✅ Installation completed for RTX 3060!"
echo "🎯 Verifying GPU setup..."

python -c "
import torch
print('=== GPU Verification ===')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {props.name}')
    print(f'  Memory: {props.total_memory / 1024**3:.1f} GB')
    
# Test GPU computation
if torch.cuda.is_available():
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.randn(1000, 1000, device='cuda')
    c = torch.matmul(a, b)
    print('✅ GPU computation test passed!')
else:
    print('❌ GPU computation test failed')
"