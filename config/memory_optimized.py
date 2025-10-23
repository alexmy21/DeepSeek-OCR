# config/memory_optimized.py
import torch
import os

# Optimize for 12GB GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def get_memory_optimized_config():
    return {
        "batch_size": 1,  # Process one document at a time
        "max_image_size": 1024,  # Limit input resolution
        "precision": "fp16",  # Use mixed precision
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "enable_gradient_checkpointing": True,  # Trade compute for memory
    }