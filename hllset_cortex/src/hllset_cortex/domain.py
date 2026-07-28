# hllset_cortex/domain.py
"""
Document tokenizer for OCR text — domain-specific token definitions.

Per STANDARD.md §5.7: "What changes is the token definition — what
string you feed to the hash function." This module defines the token
shapes for OCR document text: words, bigrams, trigrams, and sentence
hashes. The 3-gram encoding is standard — it provides structural
fingerprinting that makes the probabilistic gate intersection robust.

The tokenizer produces token strings that are fed into hllset_py's
MurmurHash3 → HLLSet pipeline. No gate filtering here — that happens
at the HLLSet level via intersection with a gate_TF HLLSet (built from
the decoder's BPE vocabulary). No n-gram classification, no manual
LUT management — the Rust layer handles hashing and TF accumulation.

Gate intersection is probabilistic:
  An invalid word may survive the gate iff all of its hash positions
  collide with valid-vocabulary words. With 1-gram alone this is a
  single collision; with 3-gram encoding (word + bigram + trigram),
  the invalid word must collide at 3 independent positions — a ~10^-5
  event for a 32,768-bit bitmap with ~36,000 valid words. Effectively
  deterministic for practical document volumes.
"""

from typing import List
from dataclasses import dataclass


@dataclass
class DocumentTokenizer:
    """Tokenizer for OCR text: words + bigrams + trigrams + sentence hashes.

    This is the ONLY domain-specific code in hllset_cortex. Everything
    else — hashing, HLLSet creation, LUT management, TF accumulation,
    materialization, gate filtering — is handled by hllset_py (Rust).

    Token shapes:
        Words:      regex [a-z0-9]{3,20} (lowercased)
        Bigrams:    "word1::word2"  (adjacent word pairs)
        Trigrams:   "word1::word2::word3" (adjacent word triples)
        Sentences:  "sentence_<md5hex8>" (sentence-level structural hash)

    The 3-gram structural encoding ensures that an invalid word must
    survive 3 independent hash position intersections with the gate_TF
    HLLSet — making false positives negligible.

    The gate (decoder vocabulary filter) is applied at the HLLSet level
    by intersecting with a gate_TF HLLSet — not at the tokenizer level.
    """

    max_tokens: int = 4096
    min_word_len: int = 3
    max_word_len: int = 20

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words + bigrams + trigrams + sentence hashes.

        Returns up to max_tokens tokens. No gate filtering — all tokens
        pass through. The gate is applied at the HLLSet lattice level.
        """
        words = self._extract_words(text)
        tokens = list(words)
        tokens.extend(self._bigrams(words))
        tokens.extend(self._trigrams(words))
        tokens.extend(self._sentence_hashes(text))
        return tokens[: self.max_tokens]

    def tokenize_words_only(self, text: str) -> List[str]:
        """Extract only word tokens (no n-grams, no sentence hashes).

        Used for roundtrip comparison — matches the "bare" token set
        that the OCR decoder actually sees.
        """
        return self._extract_words(text)

    def _extract_words(self, text: str) -> List[str]:
        """Extract lowercase words matching [a-z0-9]{min,max}."""
        import re

        text = text.lower().strip()
        words = re.findall(
            rf'\b[a-z0-9]{{{self.min_word_len},{self.max_word_len}}}\b', text
        )
        return words[: self.max_tokens]

    def _bigrams(self, words: List[str]) -> List[str]:
        """Build adjacent word-pair bigrams: 'word1::word2'."""
        if len(words) < 2:
            return []
        limit = min(len(words) - 1, self.max_tokens // 3)
        return [
            f"{words[i]}::{words[i + 1]}"
            for i in range(limit)
        ]

    def _trigrams(self, words: List[str]) -> List[str]:
        """Build adjacent word-triple trigrams: 'word1::word2::word3'."""
        if len(words) < 3:
            return []
        limit = min(len(words) - 2, self.max_tokens // 3)
        return [
            f"{words[i]}::{words[i + 1]}::{words[i + 2]}"
            for i in range(limit)
        ]

    def _sentence_hashes(self, text: str) -> List[str]:
        """Sentence-level structural hashes: 'sentence_<md5hex8>'."""
        import hashlib
        import re

        sentences = re.split(r'[.!?\n]+', text)
        hashes = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10 and len(hashes) < 50:
                h = hashlib.md5(sent.encode()).hexdigest()[:8]
                hashes.append(f"sentence_{h}")
        return hashes
