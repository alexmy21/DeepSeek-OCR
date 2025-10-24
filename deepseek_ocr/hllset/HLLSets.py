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
from pathlib import Path
from typing import Union, List, Set, Dict, Tuple, Any

try:
    import juliacall
except ImportError:
    raise ImportError(
        "The 'juliacall' package is required. Please install it with 'pip install juliacall'."
    )

# --- Julia Module Loading ---

def _load_julia_module():
    """
    Loads the HllSets.jl module from the same directory as this Python file.
    This function is called once when the module is imported.
    """
    try:
        # Get the directory of this Python file
        current_dir = Path(__file__).parent
        julia_file_path = current_dir / "HllSets.jl"

        if not julia_file_path.exists():
            raise FileNotFoundError(
                f"Could not find HllSets.jl at {julia_file_path}. "
                "Ensure it is in the same directory as HLLSets.py"
            )

        # Include the Julia module. This makes it available as juliacall.Main.HllSets
        juliacall.Main.include(str(julia_file_path))
        
        # Verify that Julia dependencies are installed
        HllSets = juliacall.Main.HllSets
        juliacall.Main.eval("using HllSets")
        
        return HllSets

    except Exception as e:
        raise RuntimeError(f"Failed to load the HllSets.jl module: {e}")

# Load the Julia module when this script is first imported
try:
    jl_HllSets = _load_julia_module()
except Exception as e:
    # If loading fails, we create a dummy module to raise errors on use
    print(f"ERROR: Could not initialize HLLSets wrapper. {e}")
    jl_HllSets = None


# --- Python Wrapper Class ---

class HllSet:
    """
    A Python wrapper for the HllSets.HllSet Julia object.

    This class provides a Pythonic API for interacting with HyperLogLog sets,
    including adding elements, estimating cardinality, and performing set operations
    like union, intersection, and difference.

    Example:
        >>> hll1 = HllSet(p=10)
        >>> hll1.add("apple")
        >>> hll1.add("banana")
        >>> print(hll1.count)
        2
        >>> hll2 = HllSet(p=10)
        >>> hll2.add("banana")
        >>> hll2.add("cherry")
        >>> union_hll = hll1.union(hll2)
        >>> print(union_hll.count)
        3
    """

    def __init__(self, p: int = 10):
        """
        Initializes a new HllSet.

        Args:
            p (int): The precision of the HLL. Determines the number of registers
                     (2^p). Must be between 4 and 18. Default is 10.
        """
        if jl_HllSets is None:
            raise RuntimeError("HllSets.jl module could not be loaded.")
        self._jl_hll = jl_HllSets.HllSet(p)
        self.p = p

    @classmethod
    def _from_julia(cls, jl_hll_instance: Any):
        """
        Internal helper to create a Python HllSet from a Julia HllSet instance.
        This avoids re-calling the Julia constructor.
        """
        instance = cls.__new__(cls)  # Create instance without calling __init__
        instance._jl_hll = jl_hll_instance
        instance.p = jl_hll_instance.P # Access the type parameter P from Julia
        return instance

    # --- Core Operations ---

    def add(self, element: Any, seed: int = 0):
        """
        Adds an element to the HLL set.

        Args:
            element (Any): The element to add. Can be any hashable type.
            seed (int): An optional seed for the hash function.
        """
        self._jl_hll.add!(element, seed=seed)

    def add_all(self, elements: Union[List[Any], Set[Any]], seed: int = 0):
        """
        Adds a collection of elements to the HLL set.

        Args:
            elements (list or set): A collection of elements to add.
            seed (int): An optional seed for the hash function.
        """
        self._jl_hll.add!(elements, seed=seed)

    @property
    def count(self) -> int:
        """Estimates the cardinality (number of unique elements) in the set."""
        return int(self._jl_hll.count())

    @property
    def is_empty(self) -> bool:
        """Checks if the HLL set is empty."""
        return bool(self._jl_hll.isempty())

    @property
    def get_id(self) -> str:
        """Computes a stable SHA1 hash of the HLL set's contents."""
        return str(self._jl_hll.id())

    # --- Set Operations ---

    def union(self, other: 'HllSet') -> 'HllSet':
        """
        Computes the union of this HLL set with another.

        Args:
            other (HllSet): The other HLL set.

        Returns:
            HllSet: A new HLL set representing the union.
        """
        new_jl_hll = self._jl_hll.union(other._jl_hll)
        return HllSet._from_julia(new_jl_hll)

    def union_inplace(self, other: 'HllSet') -> 'HllSet':
        """
        Modifies this HLL set in-place to be the union with another.

        Args:
            other (HllSet): The other HLL set.

        Returns:
            HllSet: Returns self to allow for method chaining.
        """
        self._jl_hll.union!(other._jl_hll)
        
        return self

    def intersect(self, other: 'HllSet') -> 'HllSet':
        """
        Computes the intersection of this HLL set with another.

        Args:
            other (HllSet): The other HLL set.

        Returns:
            HllSet: A new HLL set representing the intersection.
        """
        new_jl_hll = self._jl_hll.intersect(other._jl_hll)
        return HllSet._from_julia(new_jl_hll)

    def diff(self, other: 'HllSet') -> Dict[str, 'HllSet']:
        """
        Computes the set difference between this HLL set and another.

        Returns a dictionary containing three HllSets:
        - 'left_exclusive': Elements in self but not in other.
        - 'intersection': Elements in both self and other.
        - 'right_exclusive': Elements in other but not in self.

        Args:
            other (HllSet): The other HLL set.

        Returns:
            dict: A dictionary of the three resulting HllSets.
        """
        # The Julia `diff` returns a named tuple, which juliacall converts to a Python object
        jl_result = self._jl_hll.diff(other._jl_hll)
        return {
            "left_exclusive": HllSet._from_julia(jl_result.DEL),
            "intersection": HllSet._from_julia(jl_result.RET),
            "right_exclusive": HllSet._from_julia(jl_result.NEW),
        }

    def symmetric_difference(self, other: 'HllSet') -> 'HllSet':
        """
        Computes the symmetric difference (XOR) of this HLL set with another.

        Args:
            other (HllSet): The other HLL set.

        Returns:
            HllSet: A new HLL set representing the symmetric difference.
        """
        new_jl_hll = self._jl_hll.set_xor(other._jl_hll)
        return HllSet._from_julia(new_jl_hll)

    # --- Similarity Measures ---

    def jaccard_similarity(self, other: 'HllSet') -> float:
        """
        Computes the Jaccard similarity (as a percentage) with another HLL set.

        Args:
            other (HllSet): The other HLL set.

        Returns:
            float: Jaccard similarity score between 0.0 and 100.0.
        """
        return float(self._jl_hll.match(other._jl_hll))

    def cosine_similarity(self, other: 'HllSet') -> float:
        """
        Computes the cosine similarity with another HLL set.

        Args:
            other (HllSet): The other HLL set.

        Returns:
            float: Cosine similarity score between 0.0 and 1.0.
        """
        return float(self._jl_hll.cosine(other._jl_hll))

    # --- Serialization ---

    def to_vector(self) -> List[int]:
        """
        Serializes the HLL set's internal registers to a list of integers.
        This is equivalent to the deprecated `dump` function in Julia.
        """
        return list(self._jl_hll.dump())

    @classmethod
    def from_vector(cls, data: List[int], p: int) -> 'HllSet':
        """
        Restores an HLL set from a list of integers created by `to_vector`.

        Args:
            data (list[int]): The list of register counts.
            p (int): The precision of the HLL set to be restored.

        Returns:
            HllSet: The restored HLL set.
        """
        hll = cls(p)
        hll._jl_hll.restore!(data)  # Ensure no invalid characters or encoding issues
        
        return hll

    # --- Python Magic Methods ---

    def __len__(self) -> int:
        """Allows using len(hll_set) to get the estimated count."""
        return self.count

    def __eq__(self, other: object) -> bool:
        """Checks if two HLL sets are identical (have the same registers)."""
        if not isinstance(other, HllSet):
            return NotImplemented
        return bool(self._jl_hll.isequal(other._jl_hll))

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the HllSet."""
        return f"HllSet(p={self.p}, estimated_count={self.count})"


# --- Usage Example ---

if __name__ == '__main__':
    if jl_HllSets is None:
        print("Cannot run example: HllSets.jl module failed to load.")
    else:
        print("--- HLLSets Python Wrapper Example ---")

        # 1. Create two HLL sets
        hll_a = HllSet(p=12)
        hll_b = HllSet(p=12)

        # 2. Add elements to the sets
        data_a = ["apple", "banana", "cherry", "date", "elderberry"]
        data_b = ["banana", "cherry", "fig", "grape", "honeydew"]

        hll_a.add_all(data_a)
        hll_b.add_all(data_b)

        print(f"Set A created with {len(data_a)} elements. Estimated count: {hll_a.count}")
        print(f"Set B created with {len(data_b)} elements. Estimated count: {hll_b.count}")
        print("-" * 20)

        # 3. Perform set operations
        union_ab = hll_a.union(hll_b)
        print(f"Union of A and B. Estimated count: {union_ab.count}")

        intersection_ab = hll_a.intersect(hll_b)
        print(f"Intersection of A and B. Estimated count: {intersection_ab.count}")

        diff_result = hll_a.diff(hll_b)
        print(f"Items in A not in B: {diff_result['left_exclusive'].count}")
        print(f"Items in B not in A: {diff_result['right_exclusive'].count}")
        print("-" * 20)

        # 4. Calculate similarity
        jaccard = hll_a.jaccard_similarity(hll_b)
        print(f"Jaccard Similarity between A and B: {jaccard:.2f}%")
        
        cosine = hll_a.cosine_similarity(hll_b)
        print(f"Cosine Similarity between A and B: {cosine:.4f}")
        print("-" * 20)

        # 5. Serialization and Deserialization
        print(f"ID of Set A: {hll_a.get_id[:16]}...")
        
        serialized_data = hll_a.to_vector()
        print(f"Serialized Set A to a vector of length {len(serialized_data)}")
        
        restored_hll_a = HllSet.from_vector(serialized_data, p=12)
        print(f"Restored Set A is equal to original: {restored_hll_a == hll_a}")
        print(f"Restored Set A estimated count: {restored_hll_a.count}")
        print("-" * 20)

        # 6. Demonstrate __len__ and __repr__
        print(f"len(hll_a) returns: {len(hll_a)}")
        print(f"repr(hll_a) returns: {repr(hll_a)}")