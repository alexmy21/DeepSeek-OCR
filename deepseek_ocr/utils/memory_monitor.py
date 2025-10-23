# deepseek_ocr/utils/memory_monitor.py
import torch
import psutil
import subprocess
import sys

def install_gputil():
    """Install GPUtil if missing"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gputil"])
        import GPUtil
        return GPUtil
    except:
        return None

def check_system_resources():
    """Check if system meets requirements - robust version"""
    
    # Try to import GPUtil, install if missing
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        gpu = gpus[0] if gpus else None
    except ImportError:
        print("⚠️  GPUtil not found, installing...")
        GPUtil = install_gputil()
        if GPUtil:
            gpus = GPUtil.getGPUs()
            gpu = gpus[0] if gpus else None
        else:
            gpu = None
            print("⚠️  Could not install GPUtil, using fallback GPU detection")
    
    print("=== System Resource Check ===")
    
    # GPU Information
    if gpu:
        print(f"✅ GPU: {gpu.name}")
        print(f"✅ GPU Memory: {gpu.memoryFree}MB free / {gpu.memoryTotal}MB total")
    else:
        # Fallback GPU detection
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            print(f"✅ GPU Memory: {total_memory:.0f}MB total")
        else:
            print("❌ No GPU detected")
    
    # System RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"✅ System RAM: {ram_gb:.1f}GB total")
    
    # Check if sufficient for DeepSeek-OCR
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb >= 10:  # 10GB+ is good
            print("✅ Sufficient GPU memory for DeepSeek-OCR")
            return True
        else:
            print("⚠️  Limited GPU memory, but should work with optimizations")
            return True
    else:
        print("❌ CUDA not available - will run on CPU only")
        return False

if __name__ == "__main__":
    check_system_resources()