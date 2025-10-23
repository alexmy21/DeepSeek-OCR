# deepseek_ocr/hllset/core.py
import numpy as np
import hashlib
from typing import List, Optional
import re

class HLLSet:
    def __init__(self, registers: np.ndarray, tau: float = 0.7, rho: float = 0.3, name: str = ""):
        self.registers = registers
        self.tau = tau  # Inclusion threshold
        self.rho = rho  # Exclusion threshold  
        self.name = name
    
    @classmethod
    def from_text(cls, text: str, p: int = 10):
        """Create HLLSet from text - completely CPU-based"""
        m = 1 << p  # 2^p registers
        registers = np.zeros(m, dtype=np.uint8)
        
        # Advanced tokenization for better semantic capture
        tokens = cls.tokenize_text(text)
        
        for token in tokens:
            hash_val = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
            index = hash_val & (m - 1)
            remaining_bits = hash_val >> p
            leading_zeros = cls._count_leading_zeros(remaining_bits) + 1
            
            if leading_zeros > registers[index]:
                registers[index] = leading_zeros
        
        return cls(registers, name=f"text_{hash(text)[:8]}")
    
    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        """Advanced tokenization for semantic understanding"""
        # Convert to lowercase and split
        text = text.lower().strip()
        
        # Extract words, phrases, and semantic chunks
        tokens = []
        
        # Basic word tokens
        words = re.findall(r'\b[a-z]{3,20}\b', text)
        tokens.extend(words)
        
        # Bigrams for phrases
        if len(words) > 1:
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
            tokens.extend(bigrams)
        
        # Semantic chunks (sentences)
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                # Add sentence hash as token
                sentence_hash = hashlib.md5(sentence.encode()).hexdigest()[:8]
                tokens.append(f"sentence_{sentence_hash}")
        
        return tokens
    
    @staticmethod
    def _count_leading_zeros(x: int) -> int:
        """Count leading zeros in a 32-bit number"""
        if x == 0:
            return 32
        return 31 - x.bit_length()
    
    def similarity_to(self, other: 'HLLSet') -> float:
        """Compute BSS similarity between two HLLSets"""
        if len(self.registers) != len(other.registers):
            return 0.0
        
        # Count matching registers (intersection approximation)
        intersection = np.sum(self.registers == other.registers)
        total = len(self.registers)
        
        return intersection / total if total > 0 else 0.0
    
    def __repr__(self):
        return f"HLLSet(name='{self.name}', tau={self.tau}, rho={self.rho}, size={len(self.registers)})"

# Test the implementation
if __name__ == "__main__":
    # Test with sample documents
    doc1 = "Machine learning and artificial intelligence are transforming industries."
    doc2 = "Deep learning neural networks are a subset of artificial intelligence."
    query = "AI and machine learning technologies"
    
    hll1 = HLLSet.from_text(doc1)
    hll2 = HLLSet.from_text(doc2) 
    hll_query = HLLSet.from_text(query)
    
    print(f"Document 1: {hll1}")
    print(f"Document 2: {hll2}")
    print(f"Query: {hll_query}")
    print(f"Similarity between doc1 and doc2: {hll1.similarity_to(hll2):.3f}")
    print(f"Similarity between query and doc1: {hll_query.similarity_to(hll1):.3f}")
    print(f"Similarity between query and doc2: {hll_query.similarity_to(hll2):.3f}")