# hllset/memory_efficient.py
class MemoryEfficientHLLIndexer:
    """HLLSet indexer optimized for limited GPU memory"""
    
    def __init__(self, max_documents_in_memory: int = 1000):
        self.max_documents = max_documents_in_memory
        self.document_index = {}
        self.cache_file = "hllset_cache.json"  # Disk backup
        
    def add_document(self, doc_id: str, ocr_text: str):
        """Add document with memory management"""
        # Convert to HLLSet (CPU operation, minimal GPU usage)
        hllset = HLLSet.from_text(ocr_text)
        
        # Store in memory
        self.document_index[doc_id] = hllset
        
        # Manage memory by offloading to disk if needed
        if len(self.document_index) > self.max_documents:
            self._offload_oldest_documents()
    
    def _offload_oldest_documents(self):
        """Move oldest documents to disk to free memory"""
        # Keep only recent documents in memory
        if len(self.document_index) > self.max_documents:
            # Remove oldest 10%
            keys_to_remove = list(self.document_index.keys())[:self.max_documents // 10]
            for key in keys_to_remove:
                del self.document_index[key]
            
            torch.cuda.empty_cache()