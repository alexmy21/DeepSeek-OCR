from .HLLSets import HLLSet
from typing import Tuple

class HLLSetSimilarity:
    @staticmethod
    def bss_tau(A: HLLSet, B: HLLSet) -> float:
        """Compute BSS_tau (inclusion) A → B = |A ∩ B| / |B|"""
        intersection = A.intersection(B)
        b_cardinality = B.cardinality()
        
        if b_cardinality == 0:
            return 0.0
        return intersection.cardinality() / b_cardinality
    
    @staticmethod
    def bss_rho(A: HLLSet, B: HLLSet) -> float:
        """Compute BSS_rho (exclusion) A → B = |A \ B| / |B|"""
        difference = A.difference(B)
        b_cardinality = B.cardinality()
        
        if b_cardinality == 0:
            return 0.0
        return difference.cardinality() / b_cardinality
    
    @staticmethod
    def directional_similarity(A: HLLSet, B: HLLSet, 
                             tau_threshold: float = 0.7,
                             rho_threshold: float = 0.3) -> Tuple[bool, float, float]:
        """
        Check if morphism A → B exists based on tau/rho thresholds
        Returns: (exists, bss_tau, bss_rho)
        """
        bss_tau_val = HLLSetSimilarity.bss_tau(A, B)
        bss_rho_val = HLLSetSimilarity.bss_rho(A, B)
        
        exists = (bss_tau_val >= tau_threshold and 
                 bss_rho_val < rho_threshold)
        
        return exists, bss_tau_val, bss_rho_val