# HLLSet Cortex — Design Document

## Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    DeepSeek-OCR                         │
│                                                         │
│  Image → Vision Encoder (SAM+CLIP) → OCR text           │
│                                         │               │
│  ┌──────────────────────────────────────┘               │
│  │                                                      │
│  │   ┌──────────┐     ┌──────────┐     ┌────────────┐   │
│  │   │  GATE    │ ──→ │  HLLSet  │ ──→ │ TokenLut   │   │
│  │   │ (static) │     │ (32K bit)│     │ (dynamic)  │   │
│  │   │          │     │          │     │            │   │
│  │   │ ~36K     │     │ murmur3  │     │ TF learned │   │
│  │   │ valid    │     │ hash fn  │     │ monotonic  │   │
│  │   │ words    │     │          │     │            │   │
│  │   └──────────┘     └──────────┘     └─────┬──────┘   │
│  │                                           │          │
│  │                materialized tokens ←──────┘          │
│  │                     │                                │
│  │                     ▼                                │
│  │              BPE token IDs                           │
│  │                     │                                │
│  │                     ▼                                │
│  │            Language Decoder → output text            │
│  └──────────────────────────────────────────────────────┘

```

## Component Roles

### Gate (tokenizer.json, static)

Filters raw OCR words through the decoder's vocabulary. Only words that
the decoder can encode as BPE token IDs pass through. This is a
correctness requirement — the decoder cannot process unknown tokens.

- Source: `tokenizer.json` from the DeepSeek-OCR model
- Size: 128K BPE tokens → ~36K English-like words extracted
- Operation: set membership check (O(1))
- The gate never changes at runtime

### HLLSet (32,768 bits, fixed)

The structural fingerprint. Every valid word is hashed via MurmurHash3
to a bit position in a fixed 1,024 × 32 register array. The same word
always lands at the same position (IICA property). Multiple words may
collide at the same position — the LUT resolves ambiguity.

- Size: 4,116 bytes (1,024 registers × 32 bits + header)
- Content-addressed: key = SHA1(bytes)
- Operations: AND, OR, XOR, popcount — all single-cycle on FPGA

### TokenLut (dynamic, learned)

The Lookup Table maps bit positions back to tokens. It starts empty and
accumulates vocabulary + term frequency as documents are ingested. When
materializing, the highest-TF token at each active bit position is
selected. TF is monotonic (CRDT) — never decreases.

- Start: empty (cold start per STANDARD.md Appendix D)
- Growth: each document adds new tokens, increments TF for seen tokens
- Convergence: after ~50-100 documents, disambiguation stabilizes

## The LUT Initialization Constraint

Per STANDARD.md Appendix D (July 28, 2026): Loading the LUT with
a large external vocabulary where all tokens have equal TF causes
random materialization (Jaccard ≈ 0.03).

**Rule:** The LUT may only contain tokens whose TF reflects actual
experience. Three valid states:

| State | Vocabulary | TF | When |
| ------- | ----------- | ----- | ------ |
| Cold start | Empty | N/A | New system |
| Lattice-covered | From current HLLSet corpus | From materialization | Resume session |
| Donor transfer | From donor LUT | Copied from donor | Knowledge transfer |

The tokenizer vocabulary is never loaded INTO the LUT — it acts only
as a GATE that filters acceptable words.

### Gate Intersection: Probabilistic Filtering

The gate_TF HLLSet intersection is **probabilistic**: an invalid word
may survive iff its hash positions collide with valid-vocabulary words.
With 3-gram encoding (word + bigram + trigram), the invalid word must
survive 3 independent position intersections — ~10⁻⁵ probability for a
32,768-bit space with ~36K valid words. Effectively deterministic.

### Latent Vocabulary (TF Storage vs Gate Access)

The LUT accumulates TF from **all tokens** — including those filtered
by the gate. This creates a **latent vocabulary**: tokens that are
currently "illegal" (not in the decoder's vocabulary) still accumulate
experience in the LUT. When the BPE vocabulary is updated (model
upgrade, new tokenizer), rebuilding the gate_TF HLLSet makes previously
illegal tokens legal — and they materialize immediately at their earned
TF rank. No cold start penalty.

This is the natural consequence of the TF-vs-rank separation principle
(STANDARD.md §3.1): **TF is stored monotonically (pre-gate), rank is
derived at query time (post-gate).** The gate controls what's rankable,
not what's storable. When the gate expands, the LUT is already warm.

## Rust Backend (hllset_py)

Unlike the previous implementation that depended on caal-llm's Python
bindings, hllset_cortex now uses a thin PyO3 crate (`hllset_py`) that
wraps hllset-next directly:

```text
hllset_py/
├── src/
│   ├── lib.rs      — module: HLLSet, TokenLut, materialize, hashing
│   ├── hllset.rs   — HLLSet Python class (IICA properties)
│   └── lut.rs      — TokenLut with monotonic TF tracking
└── Cargo.toml      — depends only on hllset-core + pyo3
```

This is the reference pattern: a minimal PyO3 wrapper that exposes
hllset-next's core capabilities to Python, with zero caal-llm code.

## Dependencies

- **hllset-py**: Thin PyO3 wrapper around hllset-core (Rust)
- **Python 3.10+**: OCR pipeline integration layer
- **No GPU required** for HLLSet operations (CPU-only, 32K bit ops)

## Relationship to DeepSeek-OCR

This module does not modify the DeepSeek-OCR codebase. It is a
self-contained Python package that imports alongside the existing
OCR pipeline. The integration point is after the vision encoder
produces text and before the decoder processes token IDs.

## References

- STANDARD.md: HLLSet Development Standard (§0.3 defines roles)
- IICA principles: Idempotent, Immutable, Content-Addressed
- MurmurHash3: deterministic hash function
- Appendix D: LUT initialization constraint
