# hllset_cortex/search.py
"""
Page-granular semantic search — EWM as a search engine.

A document is stored as a set of **page atoms**: each page is its own
content-addressed HLLSet (the `o:` original). The whole-document HLLSet is a
**view** (`v:`), the union of the page atoms — computed on demand, never
persisted separately.

Because the atoms are individually addressable, a query resolves **down to the
page** via the R-link (topological intersection, STANDARD.md §4.4): rank pages
by ``popcount(query ∩ page)``. BSS τ is reported as the scalar signal, not the
primary ranking key.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import hllset_py


@dataclass
class PageAtom:
    """A page, stored as its own content-addressed HLLSet (the `o:` original)."""

    page_id: str
    hllset: hllset_py.HLLSet

    @property
    def key(self) -> str:
        """Content address of the page atom (``h:<sha1>``)."""
        return self.hllset.content_key()


@dataclass
class Document:
    """A document = its page atoms + the whole-document view (union)."""

    pages: List[PageAtom] = field(default_factory=list)

    @property
    def view(self) -> hllset_py.HLLSet:
        """The `v:` view — union of the page atoms. Ephemeral, not persisted."""
        v = hllset_py.HLLSet()
        for page in self.pages:
            v = v.union(page.hllset)
        return v

    def by_key(self, key: str) -> Optional[PageAtom]:
        """Recover a page atom by its content address."""
        for page in self.pages:
            if page.key == key:
                return page
        return None


@dataclass
class SearchHit:
    """One ranked page match.

    ``weight`` is the R-link popcount — the architectural primary (§4.4).
    ``tau`` is the BSS inclusion ``|page ∩ query| / |query|`` — the fraction of
    the query covered by this page (the scalar convergence signal).
    """

    page_id: str
    key: str
    weight: int
    tau: float

    def __repr__(self) -> str:
        return f"SearchHit({self.page_id}, weight={self.weight}, tau={self.tau:.3f})"


def search(
    query: hllset_py.HLLSet, doc: Document, min_weight: int = 1
) -> List[SearchHit]:
    """Rank the document's pages by R-link weight against the query.

    R-link is the primary relevance score: ``R = query ∩ page``,
    ``weight = popcount(R)`` (§4.4).  Pages sharing fewer than ``min_weight``
    bits with the query are dropped — the default (1) returns only pages that
    actually overlap the query.  The rest are returned highest-weight first.
    """
    hits: List[SearchHit] = []
    for page in doc.pages:
        r = query.intersection(page.hllset)
        weight = r.popcount()
        if weight < min_weight:
            continue
        tau = page.hllset.bss_inclusion(query)
        hits.append(SearchHit(page.page_id, page.key, weight, tau))
    hits.sort(key=lambda h: h.weight, reverse=True)
    return hits
