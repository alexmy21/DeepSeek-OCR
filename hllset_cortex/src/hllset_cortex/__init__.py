# hllset_cortex
"""
HLLSet Cortex — semantic document intelligence for DeepSeek-OCR.

A reference implementation for HLLSet Algebra applications per the
hllset-next STANDARD.md. This is a black-box module that sits between
DeepSeek-OCR's vision encoder and language decoder.

Architecture:
    OCR text → tokenizer.json GATE → valid words
      → DocumentTokenizer (words + bigrams + sentence hashes)
        → MurmurHash3 → HLLSet (32,768-bit bitmap)
          → TokenLut (monotonic TF accumulation)
            → TF-ranked materialization → BPE token IDs → Decoder

Key properties (IICA):
    - Idempotent: same text → same HLLSet, every time
    - Immutable: HLLSets never change once created
    - Content-Addressed: HLLSet key = SHA1 of serialized bytes

Dependencies:
    hllset-py (Rust PyO3 bindings to hllset-next/hllset-core)
    Python 3.10+

Reference docs:
    - hllset-next/_DOCS/dev/STANDARD.md (governing standard)
    - hllset-next/README.md (platform overview)
    - DESIGN.md (this module's design)
"""

from hllset_cortex.domain import DocumentTokenizer
from hllset_cortex.filter import HLLSetFilter, FilterResult, FilterStats, FilterComparison
from hllset_cortex.pipeline import OCRPipeline, PipelineResult, GateInfo

__all__ = [
    # Tokenizer
    "DocumentTokenizer",
    # Filter
    "HLLSetFilter",
    "FilterResult",
    "FilterStats",
    "FilterComparison",
    # Pipeline
    "OCRPipeline",
    "PipelineResult",
    "GateInfo",
]
