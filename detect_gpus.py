# detect_gpus.py
import torch
import subprocess

def detect_all_gpus():
    print("=== GPU Detection ===")
    
    # Method 1: PyTorch
    print("PyTorch Detection:")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("  No CUDA devices found via PyTorch")
    
    # Method 2: nvidia-smi
    print("\nNVIDIA-SMI Detection:")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv'], 
                              capture_output=True, text=True)
        print(result.stdout)
    except:
        print("  nvidia-smi not available")

if __name__ == "__main__":
    detect_all_gpus()