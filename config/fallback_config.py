# config/fallback_config.py
import torch
import os

def get_safe_config():
    """Safe configuration that works with current driver setup"""
    
    # Check if we have a working CUDA device with sufficient memory
    cuda_works = False
    if torch.cuda.is_available():
        try:
            # Test if we can actually use CUDA
            test_tensor = torch.tensor([1.0]).cuda()
            # Check if GPU has enough memory (> 8GB)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory >= 8:
                cuda_works = True
            del test_tensor
        except:
            cuda_works = False
    
    if cuda_works:
        print("✅ Using GPU-accelerated mode")
        return {
            "device": "cuda",
            "batch_size": 2,
            "max_image_size": 1024,
            "precision": "fp16"
        }
    else:
        print("⚠️  Using CPU-optimized mode (GPU drivers need update)")
        return {
            "device": "cpu", 
            "batch_size": 1,
            "max_image_size": 768,
            "precision": "fp32",
            "enable_gradient_checkpointing": True
        }

def setup_environment():
    """Setup environment for current GPU constraints"""
    config = get_safe_config()
    
    if config["device"] == "cpu":
        # Optimize for CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide GPUs
        torch.set_num_threads(8)  # Use more CPU threads
        print("🎯 Configured for CPU processing")
    else:
        # Use GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        print("🎯 Configured for GPU processing")
    
    return config