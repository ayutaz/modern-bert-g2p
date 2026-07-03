"""Canonical evaluation metrics for Japanese G2P.

Three metrics are exposed, matching the three-pillar evaluation protocol
defined in :file:`docs/requirements.md`:

- :func:`compute_per` — phoneme error rate (JULIUS phoneme set) used against
  JSUT Basic5000 / haqumei's protocol.
- :func:`compute_cer` — kana character error rate used against JVS-3000
  (Koriyama Interspeech 2026 benchmark).
- :func:`compute_ker` — katakana kana error rate used against ROHAN 4600.

All three return a dict with the same schema so downstream reporting code can
treat them uniformly.
"""

from modernbert_g2p.metrics.cer import compute_cer
from modernbert_g2p.metrics.ker import compute_ker
from modernbert_g2p.metrics.per import compute_per

__all__ = ["compute_per", "compute_cer", "compute_ker"]
