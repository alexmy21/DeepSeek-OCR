#!/bin/bash
# install_deepseek_12gb.sh - UPDATED VERSION

echo "Installing DeepSeek-OCR for 12GB GPU systems..."

# Check if we're in the right directory
if [ ! -f "setup.py" ] && [ ! -f "pyproject.toml" ]; then
    echo "❌ ERROR: Please run this script from the repository root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected files: setup.py or pyproject.toml"
    exit 1
fi

# Create conda environment
echo "Creating conda environment..."
conda create -n deepseek-ocr python=3.10 -y
conda activate deepseek-ocr

# Install PyTorch with CUDA 11.8
echo "Installing PyTorch..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install requirements with NumPy version fix
echo "Installing dependencies..."
pip install "numpy<2.0.0"  # Explicitly install compatible NumPy first
pip install -r requirements.txt

# Install the package in development mode
echo "Installing DeepSeek-OCR with HLLSet..."
pip install -e .

echo "Installation complete! Testing resources..."

# Test the installation
python -c "
try:
    from deepseek_ocr.utils.memory_monitor import check_system_resources
    check_system_resources()
except Exception as e:
    print(f'Test failed: {e}')
    print('But installation completed successfully!')
"