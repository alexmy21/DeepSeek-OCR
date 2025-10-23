# deepseek_ocr/hllset/indexer.py
import json
import pickle
from typing import Dict, List, Optional
from pathlib import Path
from .core import HLLSet

class HLLSetShadowIndexer:
    """Shadow indexer that runs alongside OCR processing"""
    
    def __init__(self, storage_path: str = "hllset_index"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.document_index: Dict[str, HLLSet] = {}
        self.morphisms: List[dict] = []  # Store relationships
        self.load_index()
    
    def add_document(self, doc_id: str, text: str, metadata: dict = None):
        """Add a document to the index"""
        # Create HLLSet from text
        hllset = HLLSet.from_text(text)
        
        # Store in memory and discover relationships
        self.document_index[doc_id] = hllset
        self._discover_relationships(doc_id, hllset)
        
        # Save to disk
        self.save_index()
        
        print(f"✅ Indexed document: {doc_id} (HLLSet: {hllset})")
    
    def _discover_relationships(self, new_doc_id: str, new_hll: HLLSet):
        """Discover relationships with existing documents"""
        for existing_id, existing_hll in self.document_index.items():
            if existing_id == new_doc_id:
                continue
            
            # Calculate similarity in both directions
            similarity_forward = new_hll.similarity_to(existing_hll)
            similarity_backward = existing_hll.similarity_to(new_hll)
            
            # Check if morphism exists (using BSS thresholds)
            if similarity_forward >= max(new_hll.tau, existing_hll.tau):
                self.morphisms.append({
                    'source': new_doc_id,
                    'target': existing_id,
                    'strength': similarity_forward,
                    'type': 'forward'
                })
            
            if similarity_backward >= max(new_hll.tau, existing_hll.tau):
                self.morphisms.append({
                    'source': existing_id,
                    'target': new_doc_id,
                    'strength': similarity_backward,
                    'type': 'backward'
                })
    
    def query_similar(self, query_text: str, top_k: int = 5, min_similarity: float = 0.1):
        """Find documents similar to query text"""
        query_hll = HLLSet.from_text(query_text)
        results = []
        
        for doc_id, doc_hll in self.document_index.items():
            similarity = query_hll.similarity_to(doc_hll)
            if similarity >= min_similarity:
                results.append({
                    'doc_id': doc_id,
                    'similarity': similarity,
                    'hllset': doc_hll
                })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def get_related_documents(self, doc_id: str, relationship_type: str = "all"):
        """Get documents related to a specific document"""
        related = []
        
        for morphism in self.morphisms:
            if morphism['source'] == doc_id:
                related.append({
                    'target': morphism['target'],
                    'strength': morphism['strength'],
                    'type': morphism['type']
                })
        
        return related
    
    def save_index(self):
        """Save index to disk"""
        index_data = {
            'document_index': {
                doc_id: {
                    'registers': hllset.registers.tolist(),
                    'tau': hllset.tau,
                    'rho': hllset.rho,
                    'name': hllset.name
                }
                for doc_id, hllset in self.document_index.items()
            },
            'morphisms': self.morphisms
        }
        
        with open(self.storage_path / "index.json", 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def load_index(self):
        """Load index from disk"""
        index_file = self.storage_path / "index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            # Reconstruct HLLSets
            for doc_id, hll_data in index_data['document_index'].items():
                registers = np.array(hll_data['registers'])
                self.document_index[doc_id] = HLLSet(
                    registers=registers,
                    tau=hll_data['tau'],
                    rho=hll_data['rho'],
                    name=hll_data['name']
                )
            
            self.morphisms = index_data['morphisms']
            print(f"✅ Loaded index with {len(self.document_index)} documents")

# Test the indexer
if __name__ == "__main__":
    indexer = HLLSetShadowIndexer()
    
    # Sample documents
    documents = {
        "doc1": "Machine learning and artificial intelligence are transforming industries worldwide.",
        "doc2": "Deep learning neural networks represent a significant advancement in AI technology.",
        "doc3": "Natural language processing enables computers to understand human language.",
        "doc4": "Computer vision and image recognition are key applications of machine learning.",
    }
    
    # Index documents
    for doc_id, text in documents.items():
        indexer.add_document(doc_id, text)
    
    # Query the index
    query = "AI and machine learning"
    results = indexer.query_similar(query)
    
    print(f"\n🔍 Query: '{query}'")
    print("Similar documents:")
    for result in results:
        print(f"  📄 {result['doc_id']} (similarity: {result['similarity']:.3f})")
    
    # Show relationships
    print(f"\n🕸️  Document relationships:")
    for doc_id in documents.keys():
        related = indexer.get_related_documents(doc_id)
        if related:
            print(f"  {doc_id} relates to: {[r['target'] for r in related]}")