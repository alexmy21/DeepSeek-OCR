# hllset_cortex/search.py
"""
Page-granular semantic search — EWM as a search engine.

A document is stored as a set of **page atoms**: each page is its own
content-addressed HLLSet (the `o:` original). The whole-document HLLSet is a
**view** (`v:`), the union of the page atoms — computed on demand, never
persisted separately.

Two relevance primitives are in play (STANDARD.md §4.4):

- **R-link** (topological intersection + popcount) is the *FPGA-native*
  primitive — single-cycle AND + popcount, no division. It is kept for FPGA
  compatibility and reported as ``weight`` (an integer bit count).
- **BSS τ/ρ** (float inclusion/exclusion) is the *measurement* used here on
  CPU, gated exactly as the BSS morphism in hllset-core:

      τ = |page ∩ query| / |query|   (coverage — the ranking key)
      ρ = |page \\ query| / |query|  (novelty — the precision gate)

  A page is a hit iff ``τ ≥ τ_min`` and ``ρ ≤ ρ_max`` (and, for FPGA compat,
  ``weight ≥ min_weight``).  Pages are ranked by τ descending.

Note on ρ: because it is normalised by |query|, it exceeds 1 whenever a page
holds more non-query content than the query itself. So the ρ gate is the
precision signal that separates a *clean* hit from a *noisy* hit; it is most
meaningful when the compared sets are comparable in size (the
response-vs-context regime, §3.6).  ``rho_max`` is therefore opt-in here.
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
class SearchConfig:
    """Search gates — the BSS morphism thresholds plus the FPGA-compat R-link
    threshold.

    - ``tau_min``: minimum BSS inclusion (coverage of the query by a page).
    - ``rho_max``: maximum BSS exclusion (novelty / precision). Loose enough
      by default not to mask partial matches.
    - ``min_weight``: minimum R-link popcount — the FPGA-native gate.
    """

    tau_min: float = 0.0
    rho_max: float = 1.0
    min_weight: int = 1


@dataclass
class SearchHit:
    """One ranked page match.

    ``tau`` is the BSS inclusion (coverage) — the measurement and ranking key.
    ``rho`` is the BSS exclusion (novelty) — the precision gate.
    ``weight`` is the R-link popcount — the FPGA-native primitive, kept for
    compatibility.
    """

    page_id: str
    key: str
    tau: float
    rho: float
    weight: int

    def __repr__(self) -> str:
        return (
            f"SearchHit({self.page_id}, tau={self.tau:.3f}, "
            f"rho={self.rho:.3f}, weight={self.weight})"
        )


def bss_rho(a: hllset_py.HLLSet, b: hllset_py.HLLSet) -> float:
    """BSSρ exclusion ``|a \\ b| / |b|`` — mirrors hllset-core::bss_exclusion.

    The ``hllset_py`` binding does not yet expose ``bss_exclusion``, so it is
    reconstructed from the exposed ``difference`` + ``cardinality``.
    """
    b_card = b.cardinality()
    if b_card == 0.0:
        return 0.0
    return min(1.0, a.difference(b).cardinality() / b_card)


def search(
    query: hllset_py.HLLSet, doc: Document, config: Optional[SearchConfig] = None
) -> List[SearchHit]:
    """Rank the document's pages against the query.

    Measurement (BSS, CPU): rank by τ = |page ∩ query| / |query|, and apply
    the morphism gates ``τ ≥ tau_min`` and ``ρ ≤ rho_max``.  The R-link
    popcount is still computed (FPGA compat) and gated by ``min_weight``.
    """
    cfg = config or SearchConfig()
    hits: List[SearchHit] = []
    for page in doc.pages:
        r = query.intersection(page.hllset)
        weight = r.popcount()
        tau = page.hllset.bss_inclusion(query)
        rho = bss_rho(page.hllset, query)
        if tau < cfg.tau_min or rho > cfg.rho_max or weight < cfg.min_weight:
            continue
        hits.append(SearchHit(page.page_id, page.key, tau, rho, weight))
    hits.sort(key=lambda h: h.tau, reverse=True)
    return hits
