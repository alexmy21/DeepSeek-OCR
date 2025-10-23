# deepseek_ocr/hllset/indexer.py
"""
Shadow indexer that runs alongside DeepSeek-OCR processing
"""
from typing import Dict, List
from .core import HLLSet

class HLLSetShadowIndexer:
    def __init__(self):
        self.document_index: Dict[str, HLLSet] = {}
        self.cortex_relationships: List[tuple] = []
    
    def add_document(self, doc_id: str, ocr_text: str, metadata: dict = None):
        """Index a document from OCR output"""
        hllset = HLLSet.from_text(ocr_text)
        self.document_index[doc_id] = hllset
        
        # Discover relationships with existing documents
        self._discover_relationships(doc_id, hllset)
    
    def query_similar(self, query_text: str, top_k: int = 5) -> List[tuple]:
        """Find documents similar to query"""
        query_hll = HLLSet.from_text(query_text)
        similarities = []
        
        for doc_id, doc_hll in self.document_index.items():
            similarity = query_hll.similarity_to(doc_hll)
            if similarity > query_hll.tau:  # Threshold check
                similarities.append((doc_id, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]