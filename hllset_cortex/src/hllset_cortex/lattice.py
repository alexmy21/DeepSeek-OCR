# hllset_cortex/lattice.py
"""
The EWM lattice — the measured memory a search runs against.

Two spaces, one boundary:

- **token space** — encoding IDs (``tid{n}``): what the LLM exchanges. The
  TokenLUT (reverse index + TF) and materialization live here.
- **HLLSet space** — 32,768-bit sketches: where every structural operation
  (union, intersection, BSS, R-link, DRN, the temporal pyramid) lives.

The only morphisms between the two spaces are **ingest** (tokens → HLLSet,
hash + bootstrap) and **materialize** (HLLSet → tokens).

Every observation (a scanned page, a search query) is submitted the same way
(STANDARD.md §4.1, §4.3):

    H(t) = H(S(t), H(t-1), D(t-1), R(t-1), N(t))

- its encoding IDs are ingested: hashed into an HLLSet, and TF accumulates in
  the TokenLUT (never gated — everything measured is stored),
- the HLLSet is decomposed against the current state (DRN),
- the HLLSet is committed to the temporal pyramid.

Page atoms are registered as the searchable `o:` originals; the whole document
is the `v:` union view. A search query is itself an observation: it is
converted to an HLLSet and **submitted to the lattice** before the pages are
ranked, so the query is measured — not just compared transiently.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import hllset_py

from hllset_cortex.domain import hllset_from_ids, tid
from hllset_cortex.search import Document, PageAtom, SearchConfig, SearchHit, search
from hllset_cortex.temporal import DRN, TemporalPyramid, drn


@dataclass
class Lattice:
    """The measured memory.

    ``doc`` / ``context`` / ``pyramid`` are HLLSet-space (structural); ``lut``
    is token-space (the reverse index materialization reads). ``ingest`` is
    where the two spaces meet: ids accumulate TF in ``lut``, while the HLLSet
    enters the structural state (DRN + pyramid).
    """

    doc: Document = field(default_factory=Document)
    lut: hllset_py.TokenLut = field(default_factory=hllset_py.TokenLut)
    pyramid: TemporalPyramid = field(default_factory=TemporalPyramid)
    context: hllset_py.HLLSet = field(default_factory=hllset_py.HLLSet)

    @classmethod
    def from_pages(cls, pages) -> "Lattice":
        """Build a lattice from ``[(page_id, [encoding_id, ...]), ...]``."""
        lattice = cls()
        for page_id, ids in pages:
            lattice.add_page(page_id, ids)
        return lattice

    def add_page(self, page_id: str, ids) -> "Lattice":
        """Register a page as an `o:` atom and submit it to the lattice."""
        hllset = hllset_from_ids(ids)
        self.doc.pages.append(PageAtom(page_id, hllset))
        self.ingest(hllset, ids)
        return self

    def ingest(self, observation: hllset_py.HLLSet, ids) -> DRN:
        """Submit one observation.

        Token space: the ids accumulate TF in the LUT. HLLSet space: the
        observation is decomposed (DRN) against the context and committed to
        the temporal pyramid.
        """
        self.lut.record_all([tid(i) for i in ids])
        d = drn(observation, self.context)
        self.context = self.context.union(observation)
        self.pyramid.step(observation)
        return d

    def submit_query(self, query_ids) -> Tuple[hllset_py.HLLSet, DRN]:
        """Convert a query to an HLLSet and submit it to the lattice.

        Returns the query HLLSet and its DRN against the measured state
        (``d.n`` non-empty means the query introduced never-measured ids).
        """
        query = hllset_from_ids(query_ids)
        return query, self.ingest(query, query_ids)

    def search(
        self, query_ids, config: Optional[SearchConfig] = None
    ) -> List[SearchHit]:
        """Submit the query to the lattice, then rank the pages against it."""
        query, _ = self.submit_query(query_ids)
        return search(query, self.doc, config)
