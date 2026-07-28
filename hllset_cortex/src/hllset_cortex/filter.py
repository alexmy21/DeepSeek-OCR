# hllset_cortex/filter.py
"""
HLLSetFilter — semantic compressor for OCR text.

Sits between the vision encoder and the language decoder. Compresses OCR
output through the HLLSet pipeline, then materializes structurally
significant tokens back via TF-ranked disambiguation.

Architecture (per STANDARD.md):
    OCR text → DocumentTokenizer (words + bigrams + sentence hashes)
      → MurmurHash3 → HLLSet (32,768-bit bitmap)
        → ∩ gate_TF HLLSet (decoder vocabulary filter)
          → TokenLut (monotonic TF accumulation)
            → materialize (TF-ranked disambiguation)
              → BPE token IDs → Decoder

The gate_TF HLLSet is a content-addressed system-global built from the
decoder's BPE vocabulary. It filters invalid bit positions at the lattice
level via intersection — no Python-level word matching.

The LUT is a persistent singleton — created once, grows monotonically
as documents are ingested. Per STANDARD.md Appendix D: TF is earned
through experience, never seeded with equal-weight external vocabulary.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass, field
import statistics

import hllset_py

from hllset_cortex.domain import DocumentTokenizer


@dataclass
class FilterStats:
    """Statistics from one filter pass."""
    input_tokens: int = 0
    hllset_popcount: int = 0
    gate_popcount: int = 0
    output_tokens: int = 0
    compression_ratio: float = 0.0
    roundtrip_match: int = 0
    roundtrip_total: int = 0


@dataclass
class FilterResult:
    """Output from one HLLSet filter pass."""
    tokens: List[str]
    hllset: Optional[hllset_py.HLLSet] = None
    filtered_hllset: Optional[hllset_py.HLLSet] = None
    stats: FilterStats = field(default_factory=FilterStats)
    lut_size: int = 0
    error: Optional[str] = None

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def ok(self) -> bool:
        return self.error is None and len(self.tokens) > 0

    def compare_to(self, original: str) -> "FilterComparison":
        """Compare filtered output to original text."""
        import re

        original_tokens = set(re.findall(r"[a-z]{3,20}", original.lower()))
        filtered_tokens = set(self.tokens)
        common = original_tokens & filtered_tokens
        return FilterComparison(
            original_word_count=len(original_tokens),
            filtered_word_count=len(filtered_tokens),
            common_words=len(common),
            lost_words=sorted(original_tokens - filtered_tokens),
            novel_words=sorted(filtered_tokens - original_tokens),
            jaccard=len(common) / max(len(original_tokens | filtered_tokens), 1),
        )


@dataclass
class FilterComparison:
    """Comparison between original and filtered text."""
    original_word_count: int
    filtered_word_count: int
    common_words: int
    lost_words: List[str]
    novel_words: List[str]
    jaccard: float


@dataclass
class HLLSetFilter:
    """HLLSet-based semantic filter for OCR pipeline.

    Maintains a single TokenLut that accumulates vocabulary and TF
    across all processed documents. The LUT is never rebuilt — TF
    is monotonic (CRDT property).

    Optional gate_TF HLLSet filters invalid bit positions at the
    lattice level via intersection. Built once from the decoder's
    BPE vocabulary and never changes (IICA: immutable gate).

    Usage:
        filt = HLLSetFilter(max_tokens=4096)
        result = filt.process(ocr_text)
        # With gate:
        result = filt.process(ocr_text, gate_hllset=gate)

        # ... more documents ...
        print(filt.summary())  # convergence stats
    """

    max_tokens: int = 4096
    _lut: Optional[hllset_py.TokenLut] = None
    _history: List[FilterStats] = field(default_factory=list)
    _gate_hllset: Optional[hllset_py.HLLSet] = None

    @property
    def lut(self) -> hllset_py.TokenLut:
        """The persistent TokenLut singleton.

        Created on first access. Never rebuilt — TF accumulates
        monotonically across all documents.
        """
        if self._lut is None:
            self._lut = hllset_py.TokenLut()
        return self._lut

    @property
    def history(self) -> List[FilterStats]:
        return self._history

    @property
    def gate_hllset(self) -> Optional[hllset_py.HLLSet]:
        """The content-addressed gate_TF HLLSet.

        Built once from the decoder's BPE vocabulary. Intersection
        with this HLLSet filters invalid bit positions at the lattice
        level — tokens the decoder can't encode have their bits removed.
        None means no gate (all tokens pass through).
        """
        return self._gate_hllset

    @gate_hllset.setter
    def gate_hllset(self, hllset: Optional[hllset_py.HLLSet]):
        self._gate_hllset = hllset

    def process(self, text: str) -> FilterResult:
        """Run one filter pass: tokenize → HLLSet → gate ∩ → LUT → materialize.

        1. DocumentTokenizer extracts words + bigrams + sentence hashes
        2. HLLSet.from_tokens() hashes via MurmurHash3 into 32,768-bit bitmap
        3. Gate intersection: if gate_hllset is set, intersect to filter
           invalid bit positions (decoder can't encode those tokens)
        4. TokenLut.record_all() increments TF for ALL tokens (monotonic)
           — TF is accumulated from the full token stream, not just gated
        5. materialize() returns highest-TF token at each active bit position
        """
        tokenizer = DocumentTokenizer(max_tokens=self.max_tokens)

        try:
            tokens = tokenizer.tokenize(text)
        except Exception as e:
            return FilterResult(tokens=[], error=str(e))

        if not tokens:
            return FilterResult(
                tokens=[],
                stats=FilterStats(input_tokens=0),
                lut_size=self.lut.len(),
            )

        # Create HLLSet fingerprint from ALL tokens
        hllset = hllset_py.HLLSet.from_tokens(tokens)

        # Accumulate TF for ALL tokens in the persistent LUT (monotonic CRDT)
        # TF reflects the full token stream — not just gated tokens
        self.lut.record_all(tokens)

        # Gate intersection: filter bit positions not in the decoder vocabulary
        # This is an indirect filter — tokens whose bits survive the intersection
        # are (probabilistically) within the decoder's vocabulary
        if self._gate_hllset is not None:
            filtered = hllset.intersection(self._gate_hllset)
        else:
            filtered = hllset

        # Materialize from the filtered HLLSet: TF-ranked disambiguation
        materialized = hllset_py.materialize(filtered, self.lut)

        # Compute roundtrip stats against bare words (what the decoder sees)
        bare_tokens = tokenizer.tokenize_words_only(text)
        bare_set = set(bare_tokens)
        matched = [t for t in materialized if t in bare_set]

        stats = FilterStats(
            input_tokens=len(bare_tokens),
            hllset_popcount=hllset.popcount(),
            gate_popcount=filtered.popcount(),
            output_tokens=len(materialized),
            compression_ratio=len(materialized) / max(len(bare_tokens), 1),
            roundtrip_match=len(matched),
            roundtrip_total=len(materialized),
        )
        self._history.append(stats)

        return FilterResult(
            tokens=materialized,
            hllset=hllset,
            filtered_hllset=filtered,
            stats=stats,
            lut_size=self.lut.len(),
        )

    def process_batch(self, texts: List[str]) -> List[FilterResult]:
        """Process multiple documents sequentially.

        Each document feeds the shared LUT — TF accumulates across
        the batch, improving disambiguation for later documents.
        """
        return [self.process(text) for text in texts]

    def summary(self) -> Dict:
        """Convergence statistics across all processed documents."""
        if not self._history:
            return {
                "documents": 0,
                "lut_size": self.lut.len(),
                "lut_positions": self.lut.position_count(),
            }
        return {
            "documents": len(self._history),
            "lut_size": self.lut.len(),
            "lut_positions": self.lut.position_count(),
            "gate_popcount": self._gate_hllset.popcount() if self._gate_hllset else 0,
            "avg_input_tokens": statistics.mean(
                s.input_tokens for s in self._history
            ),
            "avg_hllset_popcount": statistics.mean(
                s.hllset_popcount for s in self._history
            ),
            "avg_gate_popcount": statistics.mean(
                s.gate_popcount for s in self._history
            ),
            "avg_output_tokens": statistics.mean(
                s.output_tokens for s in self._history
            ),
            "avg_roundtrip_match": statistics.mean(
                s.roundtrip_match / max(s.roundtrip_total, 1)
                for s in self._history
            ),
        }
