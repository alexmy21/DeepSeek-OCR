#!/usr/bin/env python3
"""Page-level semantic search — BSS τ/ρ gated (EWM as a search engine).

A document is stored as **page atoms** (each page its own content-addressed
`o:` HLLSet) with the whole document as a `v:` union view. Measurement uses
BSS τ/ρ (gated); the R-link popcount is kept for FPGA compatibility.

Demonstrates:
  - a query resolves down to the page (coverage τ),
  - the ρ gate is the *opt-in* precision signal: it separates a clean hit
    (page exactly the query, ρ=0) from a noisy hit (page contains the answer
    but is dominated by off-topic content, ρ saturated at 1.0),
  - addressability (by_key), and IICA determinism.

No GPU required — `tid{n}` encoding IDs are simulated; real ds-ocr page atoms
plug in with zero change.
"""
import hllset_py

from hllset_cortex import Document, PageAtom, SearchConfig, search


def tid(i: int) -> str:
    return f"tid{i}"


def page_hll(ids) -> hllset_py.HLLSet:
    return hllset_py.HLLSet.from_tokens([tid(i) for i in ids])


pages = [
    ("page_1", [1, 2, 3]),
    ("page_2", [4, 5, 6]),
    ("page_3", [10, 11]),                                   # clean answer (ρ=0)
    ("page_4", [10, 11, 50, 51, 52, 53, 54, 55]),           # noisy answer (ρ=1)
    ("page_5", [10]),                                       # partial answer (τ=0.5)
]
doc = Document([PageAtom(pid, page_hll(ids)) for pid, ids in pages])
query = page_hll([10, 11])

print("=" * 70)
print("STEP 1  document = page atoms (o:) + whole-document view (v:)")
print("=" * 70)
for p in doc.pages:
    print(f"  {p.page_id:8} popcount={p.hllset.popcount():3}  key={p.key[:28]}...")
print(f"  view (v:)  popcount={doc.view.popcount()}  key={doc.view.content_key()[:28]}...")
assert len({p.key for p in doc.pages}) == len(pages), "each atom has a unique key"

print()
print("=" * 70)
print("STEP 2  BSS τ measurement: rank by coverage, resolve to the page")
print("=" * 70)
hits = search(query, doc)  # default: tau_min=0.0, rho_max=1.0, min_weight=1
for h in hits:
    print(f"  {h}")
# page_3 and page_4 both cover the query fully (τ=1); page_5 half-covers (τ=0.5);
# page_1/2 share nothing (weight=0, dropped by min_weight).
assert {h.page_id for h in hits} == {"page_3", "page_4", "page_5"}
assert hits[0].page_id == "page_3"
assert abs(hits[0].tau - 1.0) < 0.01 and hits[0].rho < 0.01
assert hits[0].weight == 2, "R-link popcount still reported (FPGA compat)"
assert hits[-1].page_id == "page_5" and abs(hits[-1].tau - 0.5) < 0.01

print()
print("=" * 70)
print("STEP 3  ρ is the opt-in precision gate (tighten rho_max)")
print("=" * 70)
tight = search(query, doc, SearchConfig(rho_max=0.5))
print("  rho_max=1.0 (off) ->", [h.page_id for h in hits])
print("  rho_max=0.5       ->", [h.page_id for h in tight])
# page_4 contains the answer but is dominated by off-topic content (ρ=1.0);
# tightening rho_max below 1.0 drops it, leaving the clean and partial hits.
assert {h.page_id for h in tight} == {"page_3", "page_5"}

print()
print("=" * 70)
print("STEP 4  whole-document view vs page granularity")
print("=" * 70)
doc_coverage = doc.view.bss_inclusion(query)
print(f"  doc view covers the query: tau = {doc_coverage:.3f} (says 'it is here')")
print(f"  page search says:          'it is on page_3'")
assert abs(doc_coverage - 1.0) < 0.01

print()
print("=" * 70)
print("STEP 5  addressability + determinism (IICA)")
print("=" * 70)
recovered = doc.by_key(hits[0].key)
assert recovered is not None and recovered.page_id == "page_3"
print(f"  by_key({hits[0].key[:16]}...) -> {recovered.page_id}")
assert [h.key for h in search(query, doc)] == [h.key for h in hits]
print("  re-running the query returns the same ranked keys")

print()
print("PAGE-LEVEL SEARCH (BSS τ/ρ gated) COMPLETE — all assertions passed")
