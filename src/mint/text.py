"""The content modality: turning what people wrote into vectors.

Two encoders, and which one you can run depends on where you are.

``SentenceEncoder`` wraps a sentence-transformer. It is what the real
preparation step uses, it runs on the analyst's own machine where the model
weights are reachable, and it is the one whose embeddings are worth drawing
conclusions from.

``HashingEncoder`` needs no weights, no network and no download. It is a
hashed bag of character n-grams projected to the same dimensionality by a
fixed random matrix — in other words, a random-feature text embedding. It
exists so that the pipeline runs end to end in continuous integration and on
the synthetic fixture, and because it is an honest floor: any claim that the
content modality helps should be shown against the hashing encoder as well as
the pretrained one, since a gain that a random projection also achieves is a
statement about document *volume*, not about language.

The encoder actually used is written into the artifact manifest, and the model
card reports both. Two projects in three claim a text modality helps without
ever testing that a bag of words would have done the same job.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalise(text: str) -> list[str]:
    """Lowercase word tokens plus URL-ish fragments, which carry a lot here."""
    if not text:
        return []
    return _TOKEN_RE.findall(str(text).lower())[:512]


@dataclass
class HashingEncoder:
    """Offline text embedding: hashed n-grams through a fixed random matrix."""

    dim: int = 384
    n_buckets: int = 2 ** 15
    seed: int = 0
    use_bigrams: bool = True

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # A single fixed projection shared by every document, so embeddings
        # are comparable across runs and machines.
        self._proj = rng.normal(
            0, 1 / np.sqrt(self.dim), size=(self.n_buckets, self.dim)
        ).astype(np.float32)

    @property
    def name(self) -> str:
        return f"hashing-{self.n_buckets}-{self.dim}"

    def _buckets(self, text: str) -> np.ndarray:
        toks = normalise(text)
        if not toks:
            return np.empty(0, dtype=np.int64)
        grams = list(toks)
        if self.use_bigrams:
            grams += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        return np.array(
            [int(hashlib.blake2b(g.encode(), digest_size=8).hexdigest(), 16)
             % self.n_buckets for g in grams],
            dtype=np.int64,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            b = self._buckets(text)
            if b.size == 0:
                continue
            counts = np.bincount(b, minlength=self.n_buckets).astype(np.float32)
            nz = np.nonzero(counts)[0]
            # Sublinear term weighting; raw counts let one repeated word
            # dominate a short document.
            v = (1.0 + np.log(counts[nz])) @ self._proj[nz]
            n = np.linalg.norm(v)
            out[i] = v / n if n > 0 else v
        return out


class SentenceEncoder:
    """Thin wrapper over sentence-transformers, imported only when used."""

    def __init__(self, model_name: str, dim: int = 384, batch_size: int = 128):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "sentence-transformers is not installed. Either\n"
                "  pip install sentence-transformers\n"
                "or run the preparation step with --text-encoder hashing, "
                "which needs no download."
            ) from exc
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    @property
    def name(self) -> str:
        return self.model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)


def build_encoder(kind: str, model_name: str, dim: int, seed: int = 0):
    if kind == "hashing":
        return HashingEncoder(dim=dim, seed=seed)
    if kind in ("sentence-transformers", "sentence"):
        return SentenceEncoder(model_name, dim=dim)
    raise ValueError(f"unknown text encoder '{kind}'")


def pool_documents(
    embeddings: np.ndarray, max_docs: int
) -> np.ndarray:
    """Pad or trim a day's document embeddings to a fixed count.

    Padding rather than averaging, because averaging a day's documents before
    the model sees them throws away exactly the signal that matters: one
    unusual message among forty routine ones survives as a token and vanishes
    as a mean. The attention pooling inside the model does the aggregating,
    and it can learn to ignore the routine forty.
    """
    d = embeddings.shape[1] if embeddings.size else 0
    out = np.zeros((max_docs, d), dtype=np.float32)
    if embeddings.size:
        n = min(len(embeddings), max_docs)
        out[:n] = embeddings[:n]
    return out
