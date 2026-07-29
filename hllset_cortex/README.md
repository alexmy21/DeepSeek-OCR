# HLLSet Cortex — Encoding Restoration for DeepSeek-OCR

A **reference implementation** for HLLSet Algebra applications built on
[hllset-next](https://github.com/SGS_lib/fractal_manifold/hllset-next).
Receives encoding IDs from ds-OCR's vision encoder, processes them through
the HLLSet Algebra pipeline, and returns restored encoding IDs for the decoder.

ds-ocr and hllset-cortex are **independent modules**. hllset-cortex never
sees real tokens — only encoding IDs and their hashes.

## Concept

```text
ds-OCR Encoder                        ds-OCR Decoder
      │                                      ▲
      │ encoding IDs          restored enc IDs │
      ▼                                      │
╔══════════════════════════════════════════════════╗
║              hllset-cortex (black box)           ║
║                                                  ║
║  encoding IDs → Tokenizer → HLLSet → gate ∩      ║
║    → TokenLut (TF) → materialize → restored IDs  ║
║                                                  ║
║  Results: Token-LUT, HLLSets, Lattice            ║
╚══════════════════════════════════════════════════╝
```

## Scenario: PDF Book → Holographic Memory

```text
page₁ → HLLSet₁ ─┐
page₂ → HLLSet₂ ─┤
  ...            ├─ ∪ → chapter₁ ─┐
page₁₀ → HLLSet₁₀┘                ├─ ∪ → book
                    chapter₂ ─────┘
                                      │
                                      ▼
                              temporal pyramid L₀→L₆
                              (holographic memory)
```

Each scanned page produces an HLLSet. Chapters are unions of page HLLSets.
The book is a union of chapters. After scanning, the book is committed to
the temporal pyramid creating holographic memory (STANDARD.md §4.2, §4.11).

## Architecture

Built directly on hllset-next per STANDARD.md. Zero caal-llm code.

| Layer | Crate | Role |
| ------- | ------- | ------ |
| Python config | `domain.py` | Tokenizer configuration for ds-ocr encoding IDs |
| Python pipeline | `filter.py`, `pipeline.py` | Filter orchestration, gate_TF HLLSet, BPE interface |
| Rust bindings | `hllset_py` (crates/) | PyO3 wrapper: HLLSet, TokenLut, Tokenizer, materialize |
| Rust core (vendored) | hllset-core, hllset-dsl | HLLSet algebra, MurmurHash3, standard tokenizer |

## Quick Start

```bash
# One-time setup
bash setup.sh
```

```python
from hllset_cortex import HLLSetFilter, default_tokenizer

filt = HLLSetFilter()
filt.tokenizer = default_tokenizer()

# Process encoding ID streams from ds-ocr
result = filt.process_text("enc10253 enc18278 enc50690 enc10325 enc1805 enc6579")

print(f"Restored IDs: {result.token_strings}")
print(f"HLLSet key:  {result.hllset.content_key()[:40]}...")
```

See `notebooks/01_ocr_hllset_pipeline.ipynb` for the full validation pipeline
including OCR encode/decode simulation.

## Key Properties (IICA)

Per STANDARD.md Part I:

- **Idempotent**: same encoding IDs → same HLLSet, every time
- **Immutable**: HLLSets never change once created
- **Content-Addressed**: HLLSet key = SHA1 of serialized bytes

## LUT Initialization Constraint

Per STANDARD.md Appendix D: The LUT starts cold (empty) and accumulates
TF through encoding stream ingestion. Never seed with equal-TF vocabulary.

## Dependencies

- `hllset-py` — self-contained Rust PyO3 binding (vendored hllset-core + hllset-dsl)
- Python 3.10+

## Reference

- [STANDARD.md](docs/STANDARD.md) — governing development standard
- [IICA_PRINCIPLES.md](docs/IICA_PRINCIPLES.md) — IICA gate definition
- [DESIGN.md](DESIGN.md) — this module's design
- [hllset-next](https://github.com/SGS_lib/fractal_manifold/hllset-next) — platform
