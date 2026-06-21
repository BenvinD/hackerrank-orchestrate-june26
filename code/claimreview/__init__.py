"""Multi-modal damage-claim evidence review package.

Three layers:
  1. preprocessing  - pure-Python CSV loading, history join, image encode
  2. provider       - one structured VLM call per claim (added later)
  3. rules          - deterministic decision logic (added later)
"""

__all__ = ["enums", "schema"]
