"""The fusion transformer.

Three encoders and one honest question: *unusual compared to whom?*

**Behaviour.** A day is a short ordered sequence of typed events. Each position
carries a token from a fourteen-word vocabulary, the hour it happened, and four
binary facts (after hours, weekend, own machine, shared machine). Token, hour
and flags are embedded and summed, a learned ``<cls>`` position is prepended,
and the whole thing goes through a small transformer encoder. Small on purpose:
the sequences are a few hundred positions and the vocabulary is tiny, and a
large model here memorises individuals rather than learning what a working day
looks like.

**Content.** The documents a person touched that day arrive as pre-computed
sentence embeddings — the heavy encoder ran once, elsewhere, and is frozen.
They are projected into the model width and attended over. Attention rather
than averaging, because the whole point is that one unusual message among
forty routine ones should survive, and a mean erases exactly that.

**Context.** Role, business unit, department, team, plus the Big Five scores
and two organisational facts. This does not enter as extra tokens. It is used
to *modulate* the behaviour representation through FiLM — a per-channel scale
and shift produced from the context vector. That choice is the architectural
expression of the project's argument: context should change how behaviour is
read, not sit beside it as another feature to be averaged in. A director
working at 11pm and a production-line worker working at 11pm are not the same
event, and a model that concatenates role onto the end of a feature vector has
to spend capacity discovering that, while FiLM lets the role reach in and
rescale the representation directly.

**Fusion.** Behaviour queries attend over content. What a person did is read in
the light of what they wrote and read.

**Training** is self-supervised, because insider labels are far too rare to
train on — a few dozen malicious user-days in half a million. Two objectives:

* masked event modelling, the usual trick, which teaches the model what a
  plausible day looks like for this kind of person; and
* cross-modal alignment, an InfoNCE loss pulling a day's behaviour embedding
  towards its own content embedding and away from other days' in the batch.

The second objective is where the anomaly signal that interests me most comes
from. A model trained to predict what someone *wrote* from what they *did*
will fail on days where the two stop agreeing — where the file access pattern
is that of a busy analyst but the browsing is job listings, or where the email
traffic looks routine while the device activity does not. Cross-modal
disagreement is a detector you get for free from an alignment objective, and
it is not available to any single-modality model at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import MASK_ID, N_EVENT_TOKENS, PAD_ID


@dataclass
class ModelDims:
    d_model: int = 128
    n_heads: int = 4
    n_layers_behaviour: int = 4
    n_layers_fusion: int = 2
    dropout: float = 0.1
    hour_dim: int = 16
    context_dim: int = 64
    content_dim: int = 384
    n_context_categories: tuple[int, ...] = (8, 4, 3, 6, 12)
    n_context_numeric: int = 7
    proj_dim: int = 64


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positions. Sequences are short; nothing fancy needed."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class BehaviourEncoder(nn.Module):
    """Typed event sequence -> per-position representations."""

    def __init__(self, d: ModelDims):
        super().__init__()
        self.token = nn.Embedding(N_EVENT_TOKENS, d.d_model, padding_idx=PAD_ID)
        self.hour = nn.Embedding(24, d.hour_dim)
        self.hour_proj = nn.Linear(d.hour_dim, d.d_model)
        self.flag_proj = nn.Linear(4, d.d_model)
        self.pos = PositionalEncoding(d.d_model)
        self.norm_in = nn.LayerNorm(d.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d.d_model, nhead=d.n_heads, dim_feedforward=4 * d.d_model,
            dropout=d.dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, d.n_layers_behaviour)

    def forward(self, tokens, hours, flags):
        x = (self.token(tokens)
             + self.hour_proj(self.hour(hours.clamp(0, 23)))
             + self.flag_proj(flags.float()))
        x = self.norm_in(self.pos(x))
        pad_mask = tokens == PAD_ID
        return self.encoder(x, src_key_padding_mask=pad_mask), pad_mask


class ContentEncoder(nn.Module):
    """Frozen document embeddings -> attended day-level content tokens."""

    def __init__(self, d: ModelDims):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d.content_dim, d.d_model), nn.GELU(),
            nn.LayerNorm(d.d_model), nn.Dropout(d.dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d.d_model, nhead=d.n_heads, dim_feedforward=2 * d.d_model,
            dropout=d.dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.query = nn.Parameter(torch.randn(1, 1, d.d_model) * 0.02)
        self.pool = nn.MultiheadAttention(d.d_model, d.n_heads, batch_first=True)

    def forward(self, content):
        # A padded document row is all zeros; mask it out so empty days do not
        # get a confident content vector built from padding.
        pad_mask = content.abs().sum(-1) == 0
        x = self.proj(content.float())
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        q = self.query.expand(x.size(0), -1, -1)
        # A day with no documents at all would give every key a mask of True,
        # which makes attention undefined; unmask the first slot in that case.
        all_pad = pad_mask.all(dim=1)
        if all_pad.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_pad, 0] = False
        pooled, _ = self.pool(q, x, x, key_padding_mask=pad_mask)
        return x, pooled.squeeze(1), pad_mask


class ContextEncoder(nn.Module):
    """Organisational and psychometric context -> a vector, and FiLM parameters."""

    def __init__(self, d: ModelDims):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(n + 1, 16) for n in d.n_context_categories
        ])
        width = 16 * len(d.n_context_categories) + d.n_context_numeric
        self.mlp = nn.Sequential(
            nn.Linear(width, d.context_dim), nn.GELU(),
            nn.LayerNorm(d.context_dim),
        )
        # FiLM: context produces a scale and a shift applied to every position
        # of the behaviour representation.
        self.film = nn.Linear(d.context_dim, 2 * d.d_model)

    def forward(self, cat, num):
        parts = [emb(cat[:, i].clamp(min=0)) for i, emb in enumerate(self.embeds)]
        h = self.mlp(torch.cat([*parts, num.float()], dim=-1))
        gamma, beta = self.film(h).chunk(2, dim=-1)
        # Centred on one: at initialisation FiLM is close to the identity, so
        # the model starts out ignoring context and has to earn its use of it.
        return h, 1.0 + gamma.unsqueeze(1), beta.unsqueeze(1)


class PeerConditionedFusion(nn.Module):
    """The whole thing."""

    def __init__(self, d: ModelDims):
        super().__init__()
        self.dims = d
        self.behaviour = BehaviourEncoder(d)
        self.content = ContentEncoder(d)
        self.context = ContextEncoder(d)

        fusion_layer = nn.TransformerDecoderLayer(
            d_model=d.d_model, nhead=d.n_heads, dim_feedforward=4 * d.d_model,
            dropout=d.dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.fusion = nn.TransformerDecoder(fusion_layer, d.n_layers_fusion)

        self.mlm_head = nn.Sequential(
            nn.Linear(d.d_model, d.d_model), nn.GELU(),
            nn.LayerNorm(d.d_model), nn.Linear(d.d_model, N_EVENT_TOKENS),
        )
        self.proj_behaviour = nn.Linear(d.d_model, d.proj_dim)
        self.proj_content = nn.Linear(d.d_model, d.proj_dim)
        self.day_norm = nn.LayerNorm(d.d_model)

    def forward(self, tokens, hours, flags, content, cat, num):
        h_b, pad_b = self.behaviour(tokens, hours, flags)
        ctx, gamma, beta = self.context(cat, num)
        h_b = h_b * gamma + beta                       # context modulates behaviour

        h_c_tokens, h_c_pooled, pad_c = self.content(content)

        fused = self.fusion(
            tgt=h_b, memory=h_c_tokens,
            tgt_key_padding_mask=pad_b, memory_key_padding_mask=pad_c,
        )

        # The day representation is a masked mean over real positions. Mean
        # rather than a <cls> token because sequences vary in length by an
        # order of magnitude and a single learned slot ends up dominated by
        # the long ones.
        keep = (~pad_b).float().unsqueeze(-1)
        day = self.day_norm((fused * keep).sum(1) / keep.sum(1).clamp(min=1.0))

        return {
            "positions": fused,
            "day": day,
            "content_pooled": h_c_pooled,
            "context": ctx,
            "z_behaviour": F.normalize(self.proj_behaviour(day), dim=-1),
            "z_content": F.normalize(self.proj_content(h_c_pooled), dim=-1),
            "logits": self.mlm_head(fused),
            "pad_mask": pad_b,
        }


# --------------------------------------------------------------------------
# objectives
# --------------------------------------------------------------------------
def apply_masking(
    tokens: torch.Tensor, probability: float, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a fraction of real tokens with ``<mask>``.

    Padding is never masked — predicting padding is free accuracy that makes
    the loss look better and teaches nothing.
    """
    real = tokens != PAD_ID
    draw = torch.rand(tokens.shape, device=tokens.device, generator=generator)
    selected = (draw < probability) & real
    # Guarantee at least one target per row, otherwise short days contribute
    # nothing to the loss and the model never sees them.
    empty = ~selected.any(dim=1) & real.any(dim=1)
    if empty.any():
        first = real[empty].float().argmax(dim=1)
        selected[empty, first] = True
    masked = tokens.clone()
    masked[selected] = MASK_ID
    return masked, selected


def masked_token_loss(logits, targets, selected, reduction: str = "mean"):
    """Cross-entropy over masked positions only."""
    if selected.sum() == 0:
        return logits.sum() * 0.0
    flat_logits = logits[selected]
    flat_targets = targets[selected].long()
    return F.cross_entropy(flat_logits, flat_targets, reduction=reduction)


def per_row_masked_loss(logits, targets, selected):
    """Same loss, kept per user-day, for use as an anomaly score."""
    losses = torch.zeros(logits.size(0), device=logits.device)
    counts = selected.sum(dim=1).clamp(min=1)
    if selected.sum() == 0:
        return losses
    ce = F.cross_entropy(
        logits[selected], targets[selected].long(), reduction="none")
    rows = torch.nonzero(selected, as_tuple=True)[0]
    losses = losses.index_add(0, rows, ce)
    return losses / counts


def alignment_loss(z_b: torch.Tensor, z_c: torch.Tensor, temperature: float):
    """InfoNCE between a day's behaviour and its own content, both directions."""
    logits = z_b @ z_c.t() / temperature
    labels = torch.arange(z_b.size(0), device=z_b.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.t(), labels))


def build_model(model_cfg: dict, content_dim: int,
                context_cardinalities: tuple[int, ...],
                n_context_numeric: int) -> PeerConditionedFusion:
    dims = ModelDims(
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers_behaviour=int(model_cfg["n_layers_behaviour"]),
        n_layers_fusion=int(model_cfg["n_layers_fusion"]),
        dropout=float(model_cfg["dropout"]),
        hour_dim=int(model_cfg["hour_embedding_dim"]),
        context_dim=int(model_cfg["context_dim"]),
        content_dim=content_dim,
        n_context_categories=tuple(context_cardinalities),
        n_context_numeric=n_context_numeric,
    )
    return PeerConditionedFusion(dims)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
