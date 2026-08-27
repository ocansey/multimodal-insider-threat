"""Configuration loading and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def find_root(start: Path | None = None) -> Path:
    """Walk up until we find the config file; fall back to the cwd."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "config" / "config.yaml").is_file():
            return parent
    return Path.cwd()


class Config:
    """Dictionary-backed config with attribute access to the top-level blocks.

    Deliberately thin. The value of a config object is that there is exactly
    one place to look when a number in a report needs explaining, not that it
    validates anything clever.
    """

    def __init__(self, raw: dict[str, Any], root: Path):
        self._raw = raw
        self.root = root

    def __getattr__(self, name: str) -> Any:
        try:
            return self._raw[name]
        except KeyError as exc:
            raise AttributeError(
                f"no config section '{name}'; have {sorted(self._raw)}"
            ) from exc

    def __contains__(self, name: str) -> bool:
        return name in self._raw

    def path(self, key: str) -> Path:
        return self.root / self._raw["paths"][key]

    def ensure_dirs(self) -> None:
        for key in self._raw["paths"]:
            self.path(key).mkdir(parents=True, exist_ok=True)

    @property
    def seed(self) -> int:
        return int(self._raw["model"]["seed"])

    def as_dict(self) -> dict[str, Any]:
        return self._raw

    def describe(self) -> str:
        """A one-line provenance stamp written into every result file."""
        return (
            f"release={self._raw['data']['release']} "
            f"norm={self._raw['scoring']['normalisation']} "
            f"d_model={self._raw['model']['d_model']} "
            f"seed={self.seed}"
        )


def load_config(path: str | Path | None = None) -> Config:
    root = find_root()
    target = Path(path) if path else root / "config" / "config.yaml"
    with open(target, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh), root)
