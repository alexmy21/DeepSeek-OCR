# hllset_cortex/lattice.py
"""
The EWM lattice — the measured memory a search runs against.

Every observation (a scanned page, a search query) is submitted to the
lattice the same way (STANDARD.md §4.1, §4.3):

    H(t) = H(S(t), H(t-1), D(t-1), R(t-1), N(t))

- the observation's encoding IDs accumulate TF in the LUT (the LUT is never
  gated — everything measured is stored),
- the observation is decomposed against the current state (DRN),
- the observation is committed to the temporal pyramid.

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
    """The measured memory: page atoms + LUT + temporal pyramid + context."""

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
        """Submit one observation: LUT TF + DRN + temporal-pyramid commit."""
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
