"""Phase 2 evaluation harness — decoupled from model inference.

Callers pass a ``prediction_fn`` (``text -> hypothesis``) so tests can score
without loading torch/transformers. Real model wrappers plug the same shape.

Public surface:

- :func:`bootstrap_ci`, :func:`bootstrap_diff_ci` — pure-numpy 95% CI
  (percentile method) matching ``scipy.stats.bootstrap(method="percentile")``
  on a fixed seed.
- :func:`run_eval` — generic loop over Phase 1 :class:`Row` items.
- :func:`run_eval_p_a`, :func:`run_eval_p_c` — pilot-specific wrappers
  around ``run_eval``.
- :func:`score_jsut`, :func:`score_jvs`, :func:`score_rohan`,
  :func:`score_hardset` — dataset-specific readers wired to the canonical
  metrics (PER / CER / KER).
- :func:`evaluate_checkpoint` — pilot × dataset dispatch used by the
  ``python -m modernbert_g2p eval`` CLI hook.
- :func:`build_comparison_table` — Markdown table assembly used by
  ``python -m modernbert_g2p compare``.
"""

from modernbert_g2p.evaluation.bootstrap import bootstrap_ci, bootstrap_diff_ci
from modernbert_g2p.evaluation.eval import (
    build_comparison_table,
    evaluate_checkpoint,
    run_eval,
    run_eval_p_a,
    run_eval_p_c,
    score_hardset,
    score_jsut,
    score_jvs,
    score_rohan,
)

__all__ = [
    "bootstrap_ci",
    "bootstrap_diff_ci",
    "build_comparison_table",
    "evaluate_checkpoint",
    "run_eval",
    "run_eval_p_a",
    "run_eval_p_c",
    "score_hardset",
    "score_jsut",
    "score_jvs",
    "score_rohan",
]
