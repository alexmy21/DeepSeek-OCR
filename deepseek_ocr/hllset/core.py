# deepseek_ocr/hllset/core.py
"""
Minimal, efficient HLLSet implementation for DeepSeek-OCR
"""
import numpy as np
import hashlib
from typing import List, Optional

class HLLSet:
    def __init__(self, registers: np.ndarray, tau: float = 0.7, rho: float = 0.3, name: str = ""):
        self.registers = registers
        self.tau = tau
        self.rho = rho
        self.name = name
    
    @classmethod
    def from_text(cls, text: str, p: int = 12, tokenizer=None):
        """Create HLLSet from OCR text output"""
        m = 1 << p  # 2^p registers
        registers = np.zeros(m, dtype=np.uint8)
        
        # Use provided tokenizer or default
        if tokenizer is None:
            tokens = text.lower().split()
        else:
            tokens = tokenizer(text)
        
        for token in tokens:
            # HyperLogLog update logic
            hash_val = int(hashlib.sha256(token.encode()).hexdigest()[:16], 16)
            index = hash_val & (m - 1)
            remaining_bits = hash_val >> p
            leading_zeros = cls._count_leading_zeros(remaining_bits) + 1
            
            if leading_zeros > registers[index]:
                registers[index] = leading_zeros
        
        return cls(registers, name=f"text_{hash(text)[:8]}")
    
    def similarity_to(self, other: 'HLLSet') -> float:
        """BSS similarity measure"""
        intersection = np.sum(self.registers == other.registers)
        total = len(self.registers)
        return intersection / total if total > 0 else 0.0