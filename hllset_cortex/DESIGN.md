# HLLSet Cortex — Design Document

## Architecture

ds-ocr and hllset-cortex are **independent modules**. The boundary is
encoding IDs: ds-ocr produces them, hllset-cortex processes them, ds-ocr
consumes the restored IDs. hllset-cortex never sees real tokens.

```text
┌──────────────────────────────────────────────────────────────────┐
│                        DeepSeek-OCR                              │
│                                                                  │
│  Image → Vision Encoder (SAM+CLIP)                               │
│              │                                                   │
│              ▼                                                   │
│         Real tokens (text)                                       │
│              │                                                   │
│  ┌───────────┴──────────────────────────────────────────────┐    │
│  │              OCR Encoder (token → encoding ID)           │    │
│  └───────────┬──────────────────────────────────────────────┘    │
│              │                                                   │
│              │  encoding IDs (enc10253, enc18278, ...)           │
│              ▼                                                   │
│  ╔══════════════════════════════════════════════════════════╗    │
│  ║              hllset-cortex (black box)                   ║    │
│  ║                                                          ║    │
│  ║  encoding IDs → hllset_py.Tokenizer (standard pipeline)  ║    │
│  ║    → MurmurHash3 → HLLSet (32,768-bit fingerprint)       ║    │
│  ║      → ∩ gate_TF HLLSet (decoder vocabulary filter)      ║    │
│  ║        → TokenLut (monotonic TF, pre-gate)               ║    │
│  ║          → materialize (TF-ranked disambiguation)        ║    │
│  ║                                                          ║    │
│  ║  Results:                                                ║    │
│  ║    1. Token-LUT: encoding_id → hash_position (+ TF)      ║    │
│  ║    2. HLLSets: content-addressed structural fingerprints ║    │
│  ║    3. Lattice: pages → chapters → books (union ops)      ║    │
│  ╚══════════════════════════════════════════════════════════╝    │
│              │                                                   │
│              │  restored encoding IDs                            │
│              ▼                                                   │
│  ┌───────────┴──────────────────────────────────────────────┐    │
│  │              OCR Decoder (encoding ID → token)           │    │
│  └───────────┬──────────────────────────────────────────────┘    │
│              │                                                   │
│              ▼                                                   │
│         Real tokens (text) → output document/pdf/image           │
└──────────────────────────────────────────────────────────────────┘
```

## Component Roles

### OCR Encoder/Decoder (ds-ocr, external)

The ds-OCR model maps between real tokens (words) and encoding IDs.
hllset-cortex has **no access** to this mapping — it only sees encoding
IDs as opaque byte sequences. The encoder is simulated in the notebook
for demonstration; in production, it is the model's internal vocabulary.

### hllset_py.Tokenizer (standard hllset-dsl pipeline)

The canonical HLLSet Algebra tokenizer from hllset-dsl. Processes
encoding IDs through the standard pipeline:

```text
Bytes → [Pattern Match] → Tokens → [Normalize] → [N-grams] → [Boundary Pad]
```

- N-grams use NUL (`\x00`) separator — the standard HLLSet convention
- 3-gram encoding (1..3) provides structural fingerprinting
- Composable: pattern, normalizers, n-gram range, boundary padding
- Exposed as `hllset_py.Tokenizer` — configured via `domain.py`

### Gate (gate_TF HLLSet, content-addressed)

Built once from the decoder's valid encoding ID vocabulary. Applied via
HLLSet intersection at the lattice level — a **bit-level indirect filter**.

- Content-addressed: `h:<sha1>`, immutable (IICA)
- Intersection is probabilistic: invalid ID survives only if all its
  hash positions collide with valid IDs (~10⁻⁵ with 3-gram encoding)
- Gate never changes at runtime; rebuilt on vocabulary update
- Per STANDARD.md §2.2: akin to `system:global_1`

### HLLSet (32,768 bits, fixed)

The structural fingerprint. Every encoding ID is hashed via MurmurHash3
to a bit position in a 1,024 × 32 register array. Same ID → same
position (IICA). Multiple IDs may collide — the LUT resolves ambiguity.

- Size: 4,116 bytes
- Content-addressed: key = SHA1(bytes)
- Operations: AND, OR, XOR, popcount — single-cycle on FPGA

### TokenLut (dynamic, learned)

Maps bit positions back to encoding IDs. Starts empty (cold start),
accumulates encoding IDs + TF monotonically as streams are ingested.
Materialization selects highest-TF ID at each active bit position.

- Start: empty (cold start per STANDARD.md Appendix D)
- Growth: each stream adds new IDs, increments TF for seen IDs
- TF is monotonic CRDT — never decreases
- Convergence: after ~50-100 streams, disambiguation stabilizes

### Lattice

Encoding streams form a lattice under union (OR) and intersection (AND):

```text
page₁ → HLLSet₁
page₂ → HLLSet₂
...
chapter₁ = ∪{page₁, page₂, ...}
book = ∪{chapter₁, chapter₂, ...}
```

The lattice enables structural queries: BSS similarity between pages,
R-link intersections between chapters, rank-based relevance over time.

## Usage Scenario: PDF Book Scanning

```text
1. Scan pages:
   page₁ → OCR encoder → encoding IDs → hllset-cortex → HLLSet₁
   page₂ → OCR encoder → encoding IDs → hllset-cortex → HLLSet₂
   ...

2. Build chapters:
   chapter₁ = HLLSet₁ ∪ HLLSet₂ ∪ ... ∪ HLLSet₁₀   (10 pages)
   chapter₂ = HLLSet₁₁ ∪ ... ∪ HLLSet₂₀

3. Build book:
   book = chapter₁ ∪ chapter₂ ∪ ... ∪ chapter_N

4. Commit to temporal pyramid (STANDARD.md §4.2):
   L₀ (second) → L₁ (minute) → ... → L₆ (year)
   → holographic memory (STANDARD.md §4.11)
```

At every level — page, chapter, book, temporal layer — the same five
operations apply: ∪, ∩, \, popcount, key(). No new algebra needed.

## LUT Initialization Constraint

Per STANDARD.md Appendix D: Loading the LUT with equal-TF external
vocabulary causes random materialization (Jaccard ≈ 0.03).

**Rule:** The LUT may only contain encoding IDs whose TF reflects
actual experience. Three valid states:

| State | Vocabulary | TF | When |
| ------- | ----------- | ----- | ------ |
| Cold start | Empty | N/A | New system |
| Lattice-covered | From current HLLSet corpus | From materialization | Resume session |
| Donor transfer | From donor LUT | Copied from donor | Knowledge transfer |

The decoder vocabulary is never loaded INTO the LUT — it acts only
as a GATE (gate_TF HLLSet) that filters invalid encoding IDs.

### Gate Intersection: Probabilistic Filtering

The gate_TF HLLSet intersection is **probabilistic**: an invalid encoding
ID may survive iff its hash positions collide with valid-vocabulary IDs.
With 3-gram encoding, the invalid ID must survive 3 independent position
intersections — ~10⁻⁵ for a 32,768-bit space. Effectively deterministic.

### Latent Vocabulary (TF Storage vs Gate Access)

The LUT accumulates TF from **all encoding IDs** — including those
filtered by the gate. This creates a **latent vocabulary**: IDs that are
currently "illegal" still accumulate experience. When the decoder
vocabulary expands (model upgrade), rebuilding the gate makes previously
illegal IDs legal — they materialize immediately at their earned TF.
No cold start penalty.

This is the TF-vs-rank separation (STANDARD.md §3.1): **TF is stored
monotonically (pre-gate), rank is derived at query time (post-gate).**
The gate controls what's rankable, not what's storable.

## Rust Backend (hllset_py)

Self-contained PyO3 crate wrapping hllset-next crates:

```text
crates/hllset_py/
├── src/
│   ├── lib.rs        — module: HLLSet, TokenLut, Tokenizer, materialize
│   ├── hllset.rs     — HLLSet Python class (IICA, BSS, lattice ops)
│   ├── lut.rs        — TokenLut with monotonic TF (CRDT)
│   └── tokenizer.rs  — hllset-dsl Tokenizer Python wrapper
├── vendor/
│   ├── hllset-core/  — HLLSet, MurmurHash3, BSS, content addressing
│   └── hllset-dsl/   — Tokenizer, Pattern (standard pipeline)
├── Cargo.toml
└── pyproject.toml
```

Zero external Rust dependencies beyond crates.io.

## Dependencies

- **hllset-py**: Self-contained Rust PyO3 binding (vendored hllset-core + hllset-dsl)
- **Python 3.10+**: Pipeline integration layer
- **No GPU required** — HLLSet operations are CPU-only, 32K bit ops

## Relationship to DeepSeek-OCR

ds-ocr and hllset-cortex are **totally independent**:

| Concern | ds-ocr | hllset-cortex |
| --------- | -------- | --------------- |
| Token meaning | Knows words | Sees only encoding IDs |
| Encoding map | Owns token↔ID table | No access |
| Gate vocabulary | Provides valid ID set | Builds gate_TF HLLSet |
| Input | Images | Encoding ID streams |
| Output | Encoding IDs | Restored encoding IDs |

The integration point: ds-ocr encoder → encoding IDs → hllset-cortex →
restored IDs → ds-ocr decoder. hllset-cortex is a black box at this boundary.

## References

- [STANDARD.md](docs/STANDARD.md) — governing development standard
- [IICA_PRINCIPLES.md](docs/IICA_PRINCIPLES.md) — IICA gate definition
- hllset-next notebooks: `08_holographic_memory.ipynb` — temporal pyramid
