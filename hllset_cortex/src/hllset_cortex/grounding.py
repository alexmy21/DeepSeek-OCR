# hllset_cortex/grounding.py
"""
Grounding — the EWM<->LLM validation step (Part X §10.7, §3.6).

The response is validated against the measured state in two one-sided steps,
per STANDARD.md Part X:

    1. Hallucination test (exact-LUT).  A never-measured encoding is flagged.
    2. Context comparison (R-link).      R = S(t) ∩ response, weight = popcount(R).

This module ports the EWM-nanoLM findings (``nanolm-context/src/gate.rs``,
``grounding.rs``) onto the ds-ocr substrate, using only the existing
``hllset_py`` lattice operations.

The load-bearing guarantee (STANDARD.md Appendix A):

    A single-HLLSet ``gate ∩`` saturates and leaks ~97% of out-of-vocabulary
    ids by collision (and 2-of-3 consensus is *worse*, ~99.99%).  The correct
    gate is the **exact LUT forward map** — membership lives in the reverse
    index (``TokenLut``), not the sketch.  It has 0 leak and 0 false negatives.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import hllset_py


# ── Exact-LUT membership (the corrected gate) ───────────────────────────

def exact_known(lut: hllset_py.TokenLut, token: str) -> bool:
    """Was ``token`` ever measured into the LUT?  Exact forward-map lookup.

    This is the Appendix A gate: membership in the reverse index, not the
    sketch.  Zero OOV leak, zero false negatives.  ``tf`` returns 0 for an
    unknown token, so this is a single monotonic-CRDT read.
    """
    return lut.tf(token) > 0


def hallucinated_positions(
    response_hllset: hllset_py.HLLSet, lut: hllset_py.TokenLut
) -> List[Tuple[int, int]]:
    """The active ``<reg, zeros>`` positions of the response not in the LUT.

    These are the positions the materializer would have to *add* to the LUT —
    content the response emitted that was never measured.  One-sided: a
    measured position is never flagged.
    """
    return [
        (reg, tz)
        for (reg, tz) in response_hllset.active_positions()
        if not lut.lookup_position(reg, tz)
    ]


def has_hallucination(response_hllset: hllset_py.HLLSet, lut: hllset_py.TokenLut) -> bool:
    """Does the response contain any never-measured position?"""
    return bool(hallucinated_positions(response_hllset, lut))


# ── Grounding report (τ/ρ + R-link) ─────────────────────────────────────

@dataclass
class GroundingConfig:
    """Grounding thresholds (mirrors nanolm-context::GroundingConfig)."""
    tau_min: float = 0.8
    rho_max: float = 0.2


@dataclass
class GroundingReport:
    """Grounding verdict for a response against the measured context.

    Coverage/overlap is **one measurement**, carried here in two
    representations of the same quantity:

    - ``tau``/``rho`` — the *exact* form: per-encoding LUT membership over the
      response list (``tau = |response ∩ LUT| / |response|``,
      ``rho = 1 - tau``).  Zero leak, zero false negatives.
    - ``r_link_popcount`` — the *integer* (FPGA-native) form of the same
      intersection: ``popcount(context ∩ response)``.  The float BSS τ/ρ is
      the same quantity normalised by cardinality (STANDARD.md §4.4).

    ``flagged`` lists the never-measured encodings (the one-sided hallucination
    evidence).
    """
    tau: float = 1.0
    rho: float = 0.0
    grounded: bool = True
    flagged: List[str] = field(default_factory=list)
    r_link_popcount: int = 0
    r_link_key: str = ""

    def __repr__(self) -> str:
        return (
            f"GroundingReport(tau={self.tau:.3f}, rho={self.rho:.3f}, "
            f"grounded={self.grounded}, flagged={len(self.flagged)}, "
            f"r_link={self.r_link_popcount})"
        )


def grounding_report(
    context_hllset: hllset_py.HLLSet,
    response_ids: List[str],
    lut: hllset_py.TokenLut,
    config: Optional[GroundingConfig] = None,
) -> GroundingReport:
    """Validate ``response_ids`` against the measured state (``context_hllset`` + ``lut``).

    Two one-sided steps, per §10.7:
        1. exact-LUT hallucination test over the response encoding list
           (``flagged`` = never-measured encodings),
        2. R-link context comparison (``R = S(t) ∩ response``).

    Read-only: it never mutates the LUT or the lattice.
    """
    cfg = config or GroundingConfig()

    # 1. Hallucination test — per-encoding exact-LUT coverage.
    n = len(response_ids)
    in_lut = sum(1 for t in response_ids if exact_known(lut, t))
    tau = in_lut / n if n else 1.0
    rho = (n - in_lut) / n if n else 0.0
    flagged = [t for t in response_ids if not exact_known(lut, t)]

    # 2. R-link — topological intersection (the architectural primary).
    response_hllset = hllset_py.HLLSet.from_tokens(response_ids)
    r_link = context_hllset.intersection(response_hllset)

    grounded = tau >= cfg.tau_min and rho <= cfg.rho_max

    return GroundingReport(
        tau=tau,
        rho=rho,
        grounded=grounded,
        flagged=flagged,
        r_link_popcount=r_link.popcount(),
        r_link_key=r_link.content_key(),
    )
