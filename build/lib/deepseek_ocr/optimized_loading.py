# optimized_loading.py
from transformers import AutoModel, AutoProcessor
import torch

def load_models_optimized(model_path: str):
    """Load models with memory optimizations"""
    
    # Load with 8-bit precision if available
    try:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Half precision
            device_map="auto",          # Automatic device placement
            low_cpu_mem_usage=True,     # Reduce CPU memory during loading
            trust_remote_code=True
        )
    except:
        # Fallback to standard loading
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    return model

def process_document_memory_safe(image_path, model, processor):
    """Process documents with memory management"""
    # Clear cache before processing
    torch.cuda.empty_cache()
    
    # Process with gradient disabled
    with torch.no_grad():
        with torch.cuda.amp.autocast():  # Mixed precision
            result = model.process(image_path)
    
    # Clear cache after processing
    torch.cuda.empty_cache()
    
    return result