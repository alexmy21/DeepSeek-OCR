"""
Shadow extension for DeepSeek-OCR that integrates HLLSet Cortex
"""

import os
import sys
from typing import Dict, List, Optional

# Add the current directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

from .hllset_cortex.integration.deepseek_ocr import DeepSeekOCRIntegration

class DeepSeekOCRWithCortex:
    """
    Wrapper class that adds HLLSet Cortex capabilities to DeepSeek-OCR
    """
    
    def __init__(self, 
                 redis_url: str = None,
                 git_repo_path: str = None,
                 enable_cortex: bool = True):
        
        self.enable_cortex = enable_cortex
        
        if enable_cortex:
            # Initialize HLLSet Cortex integration
            self.cortex_integration = DeepSeekOCRIntegration(
                redis_url=redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379'),
                git_repo_path=git_repo_path or os.getenv('GIT_CORTEX_PATH', './hllset_cortex_repo')
            )
        
        # Initialize original DeepSeek-OCR
        self.original_ocr = self._initialize_original_ocr()
    
    def _initialize_original_ocr(self):
        """Initialize the original DeepSeek-OCR system"""
        # This would initialize the actual DeepSeek-OCR
        # For now, return a mock
        class MockOCR:
            def recognize(self, image_path):
                return {"text": f"Mock OCR result for {image_path}", "tokens": ["mock", "ocr", "result"]}
        
        return MockOCR()
    
    def recognize(self, image_path: str, use_cortex: bool = True) -> Dict:
        """
        Recognize text from image with optional HLLSet Cortex integration
        """
        # Step 1: Use original OCR
        ocr_result = self.original_ocr.recognize(image_path)
        
        if self.enable_cortex and use_cortex:
            # Step 2: Process through HLLSet Cortex
            cortex_result = self.cortex_integration.process_document(
                image_path, "image"
            )
            
            # Step 3: Find similar documents
            similar_docs = self.cortex_integration.query_similar_documents(
                ocr_result['text']
            )
            
            # Enhance OCR result with cortex information
            ocr_result['cortex'] = {
                'hllset_sha': cortex_result['hllset_sha'],
                'similar_documents': similar_docs,
                'cardinality': cortex_result['cardinality']
            }
        
        return ocr_result
    
    def query_similar(self, query_text: str, max_results: int = 10) -> List[Dict]:
        """Query similar documents using HLLSet Cortex"""
        if not self.enable_cortex:
            return []
        
        return self.cortex_integration.query_similar_documents(
            query_text
        )[:max_results]
    
    def get_document_context(self, document_path: str) -> Optional[Dict]:
        """Get context information for a processed document"""
        if not self.enable_cortex:
            return None
        
        # This would retrieve the context from Git Cortex
        return {"status": "Context retrieval not implemented"}

# Global instance for easy access
_cortex_instance = None

def get_cortex_extension(redis_url: str = None, git_repo_path: str = None):
    """Get or create the cortex extension instance"""
    global _cortex_instance
    if _cortex_instance is None:
        _cortex_instance = DeepSeekOCRWithCortex(redis_url, git_repo_path)
    return _cortex_instance

def recognize_with_cortex(image_path: str) -> Dict:
    """Convenience function for recognition with cortex"""
    cortex_ext = get_cortex_extension()
    return cortex_ext.recognize(image_path)

def query_similar_documents(query_text: str, max_results: int = 10) -> List[Dict]:
    """Convenience function for querying similar documents"""
    cortex_ext = get_cortex_extension()
    return cortex_ext.query_similar(query_text, max_results)