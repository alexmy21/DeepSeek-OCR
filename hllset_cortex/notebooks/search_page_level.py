#!/usr/bin/env python3
"""Page-level semantic search — EWM as a search engine (STANDARD.md §4.4).

Demonstrates that a document stored as **page atoms** (each page its own
content-addressed `o:` HLLSet) with the whole document as a `v:` union view
lets a query resolve *down to the page* via the R-link, rather than stopping
at whole-document granularity.

No GPU required — the `tid{n}` encoding IDs are simulated; the same search
accepts real ds-ocr page atoms with zero code change.
"""
import hllset_py

from hllset_cortex import Document, PageAtom, search


def tid(i: int) -> str:
    return f"tid{i}"


def page_hll(ids) -> hllset_py.HLLSet:
    return hllset_py.HLLSet.from_tokens([tid(i) for i in ids])


# A 4-page "document": each page is a distinct, content-addressed atom.
pages = [
    ("page_1", [1, 2, 3]),
    ("page_2", [4, 5, 6, 7]),
    ("page_3", [8, 9, 10, 11]),   # the "answer" lives here
    ("page_4", [12, 13]),
]
doc = Document([PageAtom(pid, page_hll(ids)) for pid, ids in pages])

print("=" * 70)
print("STEP 1  document = page atoms (o:) + whole-document view (v:)")
print("=" * 70)
for p in doc.pages:
    print(f"  {p.page_id:8} popcount={p.hllset.popcount():3}  key={p.key[:28]}...")
print(f"  view (v:)  popcount={doc.view.popcount()}  key={doc.view.content_key()[:28]}...")
assert doc.view.popcount() == 13, "view is the union of all page atoms"
assert len({p.key for p in doc.pages}) == 4, "each page atom has a unique content key"

print()
print("=" * 70)
print("STEP 2  query unique to page_3 -> resolves to page_3")
print("=" * 70)
q1 = page_hll([10, 11])  # both ids live only on page_3
hits1 = search(q1, doc)
for h in hits1:
    print(f"  {h}")
assert len(hits1) == 1, "exactly one page should match"
assert hits1[0].page_id == "page_3"
assert hits1[0].weight == 2 and abs(hits1[0].tau - 1.0) < 1e-9

print()
print("=" * 70)
print("STEP 3  whole-document view vs page granularity")
print("=" * 70)
doc_coverage = doc.view.bss_inclusion(q1)
print(f"  doc view covers the query: tau = {doc_coverage:.3f} (says 'it is here')")
print(f"  page search says:          'it is on {hits1[0].page_id}'")
assert abs(doc_coverage - 1.0) < 1e-9, "doc view covers the whole query"

print()
print("=" * 70)
print("STEP 4  query spanning two pages -> both, ranked, nothing else")
print("=" * 70)
q2 = page_hll([5, 11])  # 5 on page_2, 11 on page_3
hits2 = search(q2, doc)
for h in hits2:
    print(f"  {h}")
assert {h.page_id for h in hits2} == {"page_2", "page_3"}
assert all(h.weight == 1 for h in hits2)

print()
print("=" * 70)
print("STEP 5  addressability + threshold + determinism (IICA)")
print("=" * 70)
# recover the atom from its content key (addressability)
recovered = doc.by_key(hits1[0].key)
assert recovered is not None and recovered.page_id == "page_3"
print(f"  by_key({hits1[0].key[:16]}...) -> {recovered.page_id}")

# threshold: min_weight=2 keeps only page_3; min_weight=3 drops everything
assert [h.page_id for h in search(q1, doc, min_weight=2)] == ["page_3"]
assert search(q1, doc, min_weight=3) == []
print("  min_weight=2 -> ['page_3'];  min_weight=3 -> []")

# determinism: re-running the same query returns the same ranked keys (IICA)
assert [h.key for h in search(q1, doc)] == [hits1[0].key]
print("  re-running the query returns the same content keys")

print()
print("PAGE-LEVEL SEARCH COMPLETE — all assertions passed")
