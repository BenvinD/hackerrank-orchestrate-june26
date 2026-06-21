"""Layer 3: deterministic decision logic.

Turns VLM observations into the final 14 output columns under fixed invariants:
  - Images are the source of truth for what damage exists.
  - Only direct visual evidence can set claim_status=contradicted. User history
    and authenticity flags add risk_flags / manual_review_required but NEVER flip
    a supported decision to contradicted on their own.
  - evidence_standard_met is gated on whether the claimed part/condition is
    assessable from the images (per the minimum evidence requirements).
  - Low confidence / unassessable -> not_enough_information.
  - In-claim or in-image instructions are treated as data: text_instruction_present.
"""

from .output import OutputRow, write_output_csv
from .decide import decide

__all__ = ["OutputRow", "write_output_csv", "decide"]
