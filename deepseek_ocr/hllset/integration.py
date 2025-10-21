# deepseek_ocr/hllset/integration.py
"""
Non-intrusive integration with existing DeepSeek-OCR
"""
from deepseek_ocr import OCRModel  # Import existing class

class OCRWithHLLIndexing(OCRModel):
    """Wrapper that adds shadow indexing to existing OCR"""
    
    def __init__(self, *args, enable_shadow_indexing: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if enable_shadow_indexing:
            self.shadow_indexer = HLLSetShadowIndexer()
    
    def process(self, image_path: str, **kwargs):
        """Process image with OCR and shadow indexing"""
        # Original OCR processing
        result = super().process(image_path, **kwargs)
        
        # Shadow indexing in parallel
        if hasattr(self, 'shadow_indexer'):
            doc_id = self._generate_doc_id(image_path)
            self.shadow_indexer.add_document(
                doc_id=doc_id,
                ocr_text=result.text,
                metadata={'image_path': image_path, **result.metadata}
            )
        
        return result