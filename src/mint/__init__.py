"""Multimodal insider-threat detection with peer-relative scoring.

A transformer that reads a person's working day across three modalities —
the typed sequence of things they did, the text they wrote and read, and the
organisational context they sit in — and asks not "is this unusual?" but
"is this unusual *for someone in this role*?".

    from mint import load_config, load_bundle, run_study

    cfg = load_config()
    bundle = load_bundle(cfg.path("artifacts") / "cert")
    results = run_study(bundle, cfg)
    results.save(cfg.path("tables"))
"""

from .artifacts import Bundle, SyntheticDataError
from .artifacts import load as load_bundle
from .artifacts import save as save_bundle
from .config import Config, load_config
from .pipeline import StudyResults, run_study

__all__ = [
    "Bundle", "SyntheticDataError", "load_bundle", "save_bundle",
    "Config", "load_config", "StudyResults", "run_study",
]
__version__ = "1.0.0"
