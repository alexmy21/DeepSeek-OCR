"""
HLLSets.py: A Python wrapper for the HllSets.jl Julia module.

This module provides a Pythonic interface to the high-performance HyperLogLog
implementation in Julia. It uses the `juliacall` package to bridge the two
languages.

Prerequisites:
1. Julia must be installed and accessible from your system's PATH.
2. The required Julia packages (SHA, JSON3) must be installed:
   In Julia REPL: `using Pkg; Pkg.add(["SHA", "JSON3"])`
3. The `juliacall` Python package must be installed:
   In terminal: `pip install juliacall`
"""

import os
import math
from pathlib import Path
from typing import Union, List, Set, Dict, Tuple, Any
import json

try:
    import juliacall
    from juliacall import Main, Ptr
    JULIA_AVAILABLE = True
except ImportError:
    print("juliacall not available, using Python fallback")
    JULIA_AVAILABLE = False

# --- Julia Module Loading ---

def _load_julia_module():
    """
    Loads the HllSets.jl module from the same directory as this Python file.
    """
    try:
        # Get the directory of this Python file
        current_dir = Path(__file__).parent
        julia_file_path = current_dir / "HllSets.jl"

        if not julia_file_path.exists():
            raise FileNotFoundError(f"Could not find HllSets.jl at {julia_file_path}")

        # Include the Julia module
        Main.include(str(julia_file_path))
        
        # Load the module
        Main.eval("using .HllSets")
        
        print("✅ HllSets.jl loaded successfully")
        return Main.HllSets

    except Exception as e:
        raise RuntimeError(f"Failed to load the HllSets.jl module: {e}")

# Load the Julia module
if JULIA_AVAILABLE:
    try:
        jl_HllSets = _load_julia_module()
        JULIA_LOADED = True
    except Exception as e:
        print(f"Failed to load Julia module: {e}")
        jl_HllSets = None
        JULIA_LOADED = False
else:
    jl_HllSets = None
    JULIA_LOADED = False


# --- Python Wrapper Class ---

class HllSet:
    """
    A Python wrapper for the HllSets.HllSet Julia object.
    """

    def __init__(self, p: int = 10):
        """
        Initializes a new HllSet.
        """
        if not JULIA_LOADED:
            raise RuntimeError("HllSets.jl module could not be loaded.")
        
        # Create Julia HllSet instance - use eval to avoid constructor issues
        self._jl_hll = Main.eval(f"HllSets.HllSet({p})")
        self.p = p

    @classmethod
    def _from_julia(cls, jl_hll_instance):
        """
        Internal helper to create a Python HllSet from a Julia HllSet instance.
        """
        instance = cls.__new__(cls)
        instance._jl_hll = jl_hll_instance
        # Get precision from register count
        register_count = len(jl_hll_instance.counts)
        instance.p = int(math.log2(register_count))
        return instance

    # --- Core Operations ---

    def add(self, element: Any, seed: int = 0):
        """
        Adds an element to the HLL set.
        """
        # Use Main.eval to call Julia functions with !
        if isinstance(element, (list, set)):
            # For collections, use the collection version
            elements_list = list(element)
            Main.eval(f"HllSets.add!({self._jl_hll}, {elements_list}, seed={seed})")
        else:
            # For single elements
            if isinstance(element, str):
                element_str = f'"{element}"'
            else:
                element_str = str(element)
            Main.eval(f"HllSets.add!({self._jl_hll}, {element_str}, seed={seed})")

    def add_all(self, elements: Union[List[Any], Set[Any]], seed: int = 0):
        """
        Adds a collection of elements to the HLL set.
        """
        self.add(elements, seed=seed)

    @property
    def count(self) -> int:
        """Estimates the cardinality (number of unique elements) in the set."""
        return int(Main.eval(f"HllSets.count({self._jl_hll})"))

    @property
    def is_empty(self) -> bool:
        """Checks if the HLL set is empty."""
        return bool(Main.eval(f"HllSets.isempty({self._jl_hll})"))

    @property
    def id(self) -> str:
        """Computes a stable SHA1 hash of the HLL set's contents."""
        return str(Main.eval(f"HllSets.id({self._jl_hll})"))

    # --- Set Operations ---

    def union(self, other: 'HllSet') -> 'HllSet':
        """
        Computes the union of this HLL set with another.
        """
        new_jl_hll = Main.eval(f"HllSets.union({self._jl_hll}, {other._jl_hll})")
        return HllSet._from_julia(new_jl_hll)

    def union_inplace(self, other: 'HllSet'):
        """
        Modifies this HLL set in-place to be the union with another.
        """
        Main.eval(f"HllSets.union!({self._jl_hll}, {other._jl_hll})")
        return self

    def intersect(self, other: 'HllSet') -> 'HllSet':
        """
        Computes the intersection of this HLL set with another.
        """
        new_jl_hll = Main.eval(f"HllSets.intersect({self._jl_hll}, {other._jl_hll})")
        return HllSet._from_julia(new_jl_hll)

    def diff(self, other: 'HllSet') -> Dict[str, 'HllSet']:
        """
        Computes the set difference between this HLL set and another.
        """
        jl_result = Main.eval(f"HllSets.diff({self._jl_hll}, {other._jl_hll})")
        return {
            "left_exclusive": HllSet._from_julia(jl_result.left_exclusive),
            "intersection": HllSet._from_julia(jl_result.intersection),
            "right_exclusive": HllSet._from_julia(jl_result.right_exclusive),
        }

    def symmetric_difference(self, other: 'HllSet') -> 'HllSet':
        """
        Computes the symmetric difference (XOR) of this HLL set with another.
        """
        new_jl_hll = Main.eval(f"HllSets.set_xor({self._jl_hll}, {other._jl_hll})")
        return HllSet._from_julia(new_jl_hll)

    # --- Similarity Measures ---

    def jaccard_similarity(self, other: 'HllSet') -> float:
        """
        Computes the Jaccard similarity (as a percentage) with another HLL set.
        """
        return float(Main.eval(f"HllSets.match({self._jl_hll}, {other._jl_hll})"))

    def cosine_similarity(self, other: 'HllSet') -> float:
        """
        Computes the cosine similarity with another HLL set.
        """
        return float(Main.eval(f"HllSets.cosine({self._jl_hll}, {other._jl_hll})"))

    # --- Serialization ---

    def serialize(self) -> str:
        """
        Serializes the HLL set to a JSON-compatible string.
        """
        # Get the counts vector and serialize
        counts = [int(x) for x in self._jl_hll.counts]
        return json.dumps({
            'p': self.p,
            'counts': counts,
            'type': 'HllSet'
        })

    def deserialize(self, data: str) -> 'HllSet':
        """
        Deserializes an HLL set from a string.
        """
        obj = json.loads(data)
        p = obj['p']
        counts = obj['counts']
        
        # Create new HllSet
        hll = HllSet(p)
        # Convert Python list to Julia array and restore
        jl_array = Main.eval(f"UInt32[{', '.join(map(str, counts))}]")
        Main.eval(f"HllSets.restore!({hll._jl_hll}, {jl_array})")
        return hll

    def to_vector(self) -> List[int]:
        """
        Gets the internal registers as a list of integers.
        """
        return [int(x) for x in self._jl_hll.counts]

    @classmethod
    def from_vector(cls, data: List[int], p: int) -> 'HllSet':
        """
        Creates an HLL set from a list of integers.
        """
        hll = cls(p)
        jl_array = Main.eval(f"UInt32[{', '.join(map(str, data))}]")
        Main.eval(f"HllSets.restore!({hll._jl_hll}, {jl_array})")
        return hll

    # --- Python Magic Methods ---

    def __len__(self) -> int:
        """Allows using len(hll_set) to get the estimated count."""
        return self.count

    def __str__(self) -> str:
        """User-friendly string representation."""
        return f"HllSet(p={self.p}, estimated_count={self.count})"

    def __repr__(self) -> str:
        """Developer-friendly string representation."""
        return f"HllSet(p={self.p})"


# Alternative Python implementation
class PythonHllSet:
    """Pure Python fallback implementation of HLLSet"""
    
    def __init__(self, p: int = 10):
        self.p = p
        self.elements = set()
        
    def add(self, element: Any, seed: int = 0):
        self.elements.add(str(element))
        return self
        
    def add_all(self, elements: Union[List[Any], Set[Any]], seed: int = 0):
        for element in elements:
            self.elements.add(str(element))
        return self
        
    @property
    def count(self) -> int:
        return len(self.elements)
        
    @property 
    def is_empty(self) -> bool:
        return len(self.elements) == 0
        
    @property
    def id(self) -> str:
        import hashlib
        content = ''.join(sorted(self.elements))
        return hashlib.sha1(content.encode()).hexdigest()
        
    def serialize(self) -> str:
        return json.dumps({
            'p': self.p,
            'elements': list(self.elements),
            'type': 'PythonHllSet'
        })
        
    def deserialize(self, data: str) -> 'PythonHllSet':
        obj = json.loads(data)
        self.p = obj['p']
        self.elements = set(obj['elements'])
        return self
        
    def __len__(self) -> int:
        return self.count
        
    def __str__(self) -> str:
        return f"PythonHllSet(p={self.p}, count={self.count})"


# Choose which implementation to use
if JULIA_LOADED:
    HllSet = HllSet
    print("✅ Using Julia HllSet implementation")
else:
    HllSet = PythonHllSet
    print("⚠️  Using Python fallback HllSet implementation")


# Simple test
if __name__ == '__main__':
    try:
        print("Testing HLLSets...")
        hll = HllSet(p=10)
        hll.add("test1")
        hll.add("test2")
        print(f"Count: {hll.count}")
        serialized = hll.serialize()
        print(f"Serialized: {serialized[:50]}...")
        
        # Test deserialization
        hll2 = HllSet(p=10)
        hll2.deserialize(serialized)
        print(f"Deserialized count: {hll2.count}")
        
        print("✅ Test passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()