# HLLSet Cortex — Semantic Document Intelligence

A **reference implementation** for HLLSet Algebra applications built on
[hllset-next](https://github.com/SGS_lib/fractal_manifold/hllset-next).
This module is a black-box semantic compression layer for DeepSeek-OCR.

## Concept

OCR text passes through a HyperLogLog-based filter that compresses it
to a fixed 32,768-bit structural fingerprint, then reconstructs only the
semantically significant tokens. The filter learns over time — each
document refines the vocabulary and improves accuracy.

```
OCR text → DocumentTokenizer → HLLSet (32,768-bit) → TokenLut (TF) → materialize → Decoder
               │                      │                    │
          words+bigrams          MurmurHash3        monotonic CRDT
          (domain-specific)     (hllset-core)       (TF earned)
```

## Architecture

Built directly on [hllset-next](https://github.com/SGS_lib/fractal_manifold/hllset-next)
per the [STANDARD.md](https://github.com/SGS_lib/fractal_manifold/hllset-next/_DOCS/dev/STANDARD.md).
No caal-llm dependency — this is a reference implementation demonstrating
how to build an HLLSet Algebra application on the platform.

| Layer | Crate | Role |
|-------|-------|------|
| Python domain logic | `domain.py` | OCR tokenizer (words, bigrams, trigrams, sentence hashes) |
| Python pipeline | `filter.py`, `pipeline.py` | Filter orchestration, BPE gate |
| Rust bindings | `hllset_py` (crates/) | Thin PyO3 wrapper around hllset-next |
| Rust core | `hllset-core` | HLLSet, MurmurHash3, IICA, BSS |

Three files, three concepts:

| File | What | Lines |
|------|------|-------|
| `domain.py` | OCR tokenizer (words, bigrams, trigrams, sentence hashes) | ~95 |
| `filter.py` | HLLSet filter with persistent LUT + TF | ~190 |
| `pipeline.py` | Black-box OCR pipeline (GATE + BPE) | ~180 |

## Quick Start

```bash
# One-time setup (from hllset_cortex/ project root)
python3 -m venv .venv
source .venv/bin/activate
pip install maturin

# Build and install the Rust engine
cd crates/hllset_py
maturin build --release
pip install target/wheels/hllset_py-*.whl
cd ../..

# Install the Python package
pip install -e .

# (optional) Register Jupyter kernel for notebooks
pip install jupyter ipykernel
python -m ipykernel install --user --name hllset-cortex
```

```python
from hllset_cortex import HLLSetFilter

filt = HLLSetFilter(max_tokens=4096)
result = filt.process("The neural network model processes image data.")

print(f"Materialized: {result.tokens}")
print(f"HLLSet key: {result.hllset.content_key()[:40]}...")
```

See `notebooks/01_ocr_hllset_pipeline.ipynb` for the full validation pipeline.

## Key Properties (IICA)

Per STANDARD.md Part I — every operation satisfies:

- **Idempotent**: same text → same HLLSet, every time
- **Immutable**: HLLSets never change once created
- **Content-Addressed**: HLLSet key = SHA1 of serialized bytes

## LUT Initialization Constraint

Per STANDARD.md Appendix D: The LUT must only contain tokens whose TF
reflects actual experience. The LUT starts cold (empty) and accumulates
TF through document ingestion. Never seed with equal-TF vocabulary —
it causes random materialization (Jaccard ≈ 0.03).

## Dependencies

- `hllset-py` — Rust HLLSet engine via PyO3 (direct hllset-core binding)
- Python 3.10+

## Reference

- [STANDARD.md](https://github.com/SGS_lib/fractal_manifold/hllset-next/_DOCS/dev/STANDARD.md) — governing development standard
- [hllset-next](https://github.com/SGS_lib/fractal_manifold/hllset-next) — platform
