# DeepSeek-OCR × HLLSet Cortex: A 4-Kilobyte Black Box Between Encoder and Decoder

What happens when you route an OCR model's output through a content-addressed set algebra — and why the vision encoder turns out to be optional.

Two things arrived on my desk at roughly the same time. One is DeepSeek-OCR, a model that reframes text recognition as optical 2D compression and, to my genuine surprise, runs on a single 12 GB consumer GPU. The other is LLSet Algebra, a framework whose one trick is representing anything — a page, a chapter, a book, a minute of history — as a fixed 32,768-bit fingerprint.

This post is the story of plugging one into the other. The result is hllset-cortex: a small black-box module that sits between
DeepSeek-OCR's encoder and its decoder, never sees a single real token, and turns scanned pages into what the project calls holographic memory. Everything below was executed in notebooks — I'll point to them instead of walking through every cell.

## DeepSeek-OCR in sixty seconds

DeepSeek-OCR ([paper](https://arxiv.org/abs/2510.18234), [code](https://github.com/deepseek-ai/DeepSeek-OCR)) treats a document image not
as "letters to be recognized" but as a long context that can be compressed optically onto a 2D plane. Two components: DeepEncoder, which
keeps activations low under high-resolution input and squashes the image into a small number of vision tokens, and a
DeepSeek3B-MoE-A570M decoder that reads them back out.

The paper's headline numbers:

```text
compression < 10x   -> OCR precision ~97%
compression = 20x   -> still ~60%

OmniDocBench:
    beats GOT-OCR2.0   with ~100 vision tokens (vs 256)
    beats MinerU2.0    with <800 vision tokens (vs 6,000+)

production: 200k+ pages/day on a single A100-40G
```

On our side it runs locally on an RTX 3060 in "Gundam mode" (640 px tiles, dynamic cropping): 6.3 GB of VRAM, ~5 seconds per image,
128,000-token BPE vocabulary. Very hands-on friendly.

## HLLSet Algebra in one sitting

The framework is built on one object. An HLLSet is a fixed bitmap:

HLLSet = 1,024 registers x 32 bits = 32,768 bits = 4,116 bytes

Any token is hashed (MurmurHash3) to a bit position. A document, a page, a query — each becomes a fingerprint in that same space. Three
properties matter; the project calls them IICA:

Idempotent        same input -> same fingerprint, every time
Immutable         a fingerprint never changes once created
Content-Addressed key = SHA1(bytes) -> "h:<40 hex>"

Composition preserves IICA, which is why nested structures (pages → chapters → books → memory) need no new theory.

Five operations cover every level of the hierarchy:

```text
A | B     union        OR       aggregate observations
A & B     intersection AND      shared structure (the R-link)
A \ B     difference   AND-NOT  unique structure
|A|       popcount     count    measure weight
key(A)    SHA1         hash     content-address identity
```

And one discipline — two spaces, two morphisms. Tokens (encoding IDs) and HLLSets live in different worlds and never touch directly:

```text
ingest       tokens -> HLLSet   (hash + n-gram bootstrap)
materialize  HLLSet -> tokens   (active bits -> LUT lookup)
```

Structural work (similarity, intersection, memory) happens only in HLLSet space. Anything exchanged with a model happens only in token
space. A token becomes structure only through ingest; structure becomes a token only through materialize. That single rule is what keeps
the whole thing sane.

Time is the same algebra. Every arriving observation is decomposed against the previous state:

```text
N(t) = S(t) \ H(t-1)     new
D(t) = H(t-1) \ S(t)     departed
R(t) = S(t) & H(t-1)     retained (the R-link, stored as "r:<sha1>")
```

and committed to a temporal pyramid whose layers are just unions:

H_system(t) = L0 | L1 | L2 | L3 | L4 | L5 | L6

That is the holographic memory: one 4 KB union at the top, plus a TF stack, is enough to approximate any past state.

## hllset-cortex: the black box

DeepSeek-OCR hands us encoding IDs (BPE token IDs). We route them through the algebra and hand the restored IDs back to the decoder:

DeepSeek-OCR Encoder                      DeepSeek-OCR Decoder
        |                                            ^
        | encoding IDs          restored IDs         |
        v                                            |
    +----------------------------------------------------+
    |              hllset-cortex (black box)             |
    |                                                    |
    |  encoding IDs -> Tokenizer (3-gram, NUL-separated) |
    |    -> MurmurHash3 -> HLLSet (32,768 bits)          |
    |      -> & gate_TF HLLSet (vocabulary filter)       |
    |        -> TokenLut (monotonic TF, pre-gate)        |
    |          -> materialize -> restored IDs            |
    +----------------------------------------------------+

Walking through it:

- Tokenizer — the standard hllset-dsl pipeline turns the ID stream into unigrams + bigrams + trigrams. The 3-gram encoding is what makes
structural matching meaningful.
- HLLSet — the stream becomes a fingerprint.
- Gate — the decoder's valid vocabulary is itself compressed into an HLLSet (gate_TF). One intersection filters out-of-vocabulary IDs at
the bit level; no token is ever inspected.
- TokenLut — a reverse index from bit positions back to IDs, with a term-frequency counter that only grows (monotonic CRDT). It starts
cold and learns.
- Materialize — set bits are looked up in the LUT and the highest-TF ID wins.

Two details are worth pausing on.

First, the gate has a subtlety. A single-HLLSet sketch gate is probabilistic (~10⁻⁵ false-positive by design), but under saturation it
leaks badly. The load-bearing gate is the exact-LUT forward map: "was this ID ever measured?" — 0 leak, 0 false negatives. That one fact
has a wild consequence: the encoder becomes optional. If only opaque IDs cross the boundary, and anything out-of-vocabulary is filtered
exactly at materialization, then IDs may come from anywhere — a mock stream, another tokenizer, a different model. The expensive vision
encoder is an optional front-end, not a dependency.

Second, order is recoverable. Basic materialization returns a set (no order). If you need the original sequence back, you switch
tokenizers:

```text
Tokenizer: word_pattern().lowercase().pad("<S>", "</S>").ngrams(2, 2)
Each bigram (a, b) becomes a graph edge a -> b
An Eulerian path from <S> to </S> reconstructs the sequence
ordered = hllset_py.materialize_debruijn(hllset, lut, "<S>", "</S>")
# ['<S>', 'tid0', 'tid671', 'tid18308', ..., '</S>'] -> decode -> exact text
```

## What the notebooks show

The proof of work lives in the notebooks on the branch feature/hllset-cortex-git — 11 commits, all proofs green. The main one is
01_ocr_hllset_pipeline_real.ipynb: nine validation tests with simulated IDs, then a real DeepSeek-OCR section, then grounding and search.
08_holographic_memory.ipynb covers the temporal pyramid; phase7_ewm_llm_loop.py and search_page_level.py cover the EWM↔LLM loop and
page-level retrieval. Here are the numbers; the cells are in the [notebooks
folder](https://github.com/alexmy21/DeepSeek-OCR/tree/feature/hllset-cortex-git/hllset_cortex/notebooks).

Simulated first (no GPU, mock enc{id} IDs):

gate filter      invalid IDs (enc99999, enc88888) removed by gate intersection
BSS matrix       page_1 vs page_2 (similar)   = 0.4444
                page_1 vs page_3 (different) = 0.0000
latent vocab     TF survives gate expansion; hidden IDs activate instantly
roundtrip        73% word retention (set semantics)

Then real DeepSeek-OCR on the RTX 3060. Real IDs are just a different namespace — tid{n} instead of enc{n}:

OCR text: "The neural network model processes image data for object detection tasks"
token IDs: [0, 671, 18308, 4854, 2645, 6579, 4609, 1499, 362, 2873, 11347, 10017]
encoding stream: tid0 tid671 tid18308 tid4854 tid2645 ...

Same pipeline, same code, different strings — MurmurHash3 does not care what the bytes mean:

```text
set roundtrip      82% word retention
ordered roundtrip  100% match via materialize_debruijn

The grounding section is where it gets interesting: two different hallucination diagnoses, in two different spaces.

# an encoding that never arrived through ingestion
GroundingReport(tau=0.750, rho=0.250, srho=0.083, grounded=False, flagged=1, r_link=3)

# every token known, but the response departs from the current context
GroundingReport(tau=1.000, rho=0.000, srho=0.500, grounded=False, flagged=0, r_link=1)
```

The first is token hallucination, diagnosed by the LUT in token space. The second is structural hallucination, diagnosed by BSS ρ in
HLLSet space. The LUT never hallucinates; the HLLSet the model produces does.

Search resolves to the page, not just the document:

query ids: ['tid2645', 'tid6579']
    SearchHit(page_1, tau=1.000, rho=1.000, weight=2)
localized to: page_1

And deep grounding (precedents) dives into history with the same measurement:

precedents (from history):
    Precedent(page_0, weight=2, tau=1.000)

Same measurement, three views: search ranks pages, shallow grounding checks a response against context, deep grounding retrieves
precedents. One R-link/BSS comparison over different reference sets.

## Use cases

The README lists the practical ones:

PDF book -> DeepSeek-OCR page by page
    -> one HLLSet per page -> union into chapters -> union into book
    -> commit to temporal pyramid L0..L6
    -> query by structural similarity, localized to the page

Cross-document search:
    BSS(HLLSet_a, HLLSet_b) -> related documents cluster

Model upgrade without reindexing:
    new tokenizer -> rebuild gate_TF -> old LUT stays valid -> no migration

My favorite is the last one. The LUT stores experience pre-gate; the gate only controls what may be expressed post-gate. So when a new
model ships a bigger vocabulary, previously "illegal" IDs have already been earning TF in the background. Widen the gate and they
materialize instantly. No cold start.

## Try it yourself

```text
git clone -b feature/hllset-cortex-git https://github.com/alexmy21/DeepSeek-OCR
cd DeepSeek-OCR/hllset_cortex
bash setup.sh
```

No-GPU sanity check:

from hllset_cortex import HLLSetFilter, default_tokenizer

filt = HLLSetFilter()
filt.tokenizer = default_tokenizer()
result = filt.process_text("enc10253 enc18278 enc50690 enc10325 enc1805 enc6579")

print(result.token_strings)
print(result.hllset.content_key()[:40])

Full pipeline with the real model: notebooks/e2e_dsocr_hllset.py (GPU required). Everything is documented in the
[README](https://github.com/alexmy21/DeepSeek-OCR/blob/feature/hllset-cortex-git/hllset_cortex/README.md); the algebra itself comes from
[hllset-next](https://github.com/alexmy21/hllset-next).

## The takeaway

OCR turns out to be a near-perfect citizen for HLLSet Algebra. A scanned page is a natural, physical atom — no synthetic embedding
needed. The algebra doesn't care whether the bytes say enc10253 or tid671. And once the gate is exact, the heavyweight vision encoder
becomes an optional front-end for what is, underneath, a tiny content-addressed memory that any ID stream can feed.

DeepSeek-OCR compresses a page optically. hllset-cortex compresses the memory of pages structurally. Both are the same move — keep less,
remember more.

---

Drafted with assistance from DeepSeek AI (deepseek-v4-pro). All opinions, errors, and overly enthusiastic metaphors are the author's own.
