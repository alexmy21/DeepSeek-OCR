# examples/demo_mock_ocr.py
from deepseek_ocr.hllset.indexer import HLLSetShadowIndexer
import time

class MockOCRProcessor:
    """Mock OCR processor for demonstration"""
    
    def __init__(self):
        self.indexer = HLLSetShadowIndexer()
    
    def process_document(self, file_path: str, text_content: str):
        """Simulate OCR processing with shadow indexing"""
        print(f"📄 Processing: {file_path}")
        
        # Simulate OCR processing time
        time.sleep(0.5)
        
        # Shadow indexing happens in parallel
        doc_id = f"doc_{hash(file_path)}"
        self.indexer.add_document(doc_id, text_content, {"file_path": file_path})
        
        return {"text": text_content, "doc_id": doc_id}
    
    def semantic_search(self, query: str):
        """Semantic search across processed documents"""
        return self.indexer.query_similar(query)

def demo_mock_system():
    """Demo the complete system with mock documents"""
    processor = MockOCRProcessor()
    
    # Process sample documents
    sample_docs = [
        ("research_paper.pdf", "Deep learning transformers have revolutionized natural language processing and computer vision. The attention mechanism allows models to focus on relevant parts of the input."),
        ("technical_report.docx", "Machine learning algorithms require large datasets for training. Data preprocessing and feature engineering are crucial steps in the ML pipeline."),
        ("presentation.pptx", "Artificial intelligence is transforming healthcare, finance, and transportation. Neural networks can identify patterns in complex data."),
        ("meeting_notes.txt", "We discussed implementing AI solutions for customer service. Natural language understanding can help automate support tickets."),
    ]
    
    print("🚀 Processing documents with shadow indexing...")
    for file_path, content in sample_docs:
        processor.process_document(file_path, content)
    
    # Test semantic search
    queries = [
        "neural networks and AI",
        "natural language processing",
        "machine learning data",
        "customer service automation"
    ]
    
    print("\n🎯 Semantic Search Results:")
    for query in queries:
        results = processor.semantic_search(query)
        print(f"\n🔍 Query: '{query}'")
        for result in results:
            print(f"   📄 {result['doc_id']} (score: {result['similarity']:.3f})")

if __name__ == "__main__":
    demo_mock_system()