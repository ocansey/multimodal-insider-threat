"""Self-supervised training, and the dataset that feeds it.

There are on the order of a few dozen malicious user-days in half a million.
Supervised training on that is not difficult, it is impossible — and every
paper that reports a supervised F1 on this data has, somewhere, allowed the
same campaign to appear on both sides of a split. So nothing here sees a
label. The model learns what an ordinary working day looks like for a person
in a particular role, and anomaly is defined afterwards, as the residue.

Two objectives, described in :mod:`mint.model`: masked event modelling, and
cross-modal alignment between what a person did and what they wrote. The
second is weighted by ``alignment_weight`` and is the one that gives the
project a detector no single-modality model has access to.

Early stopping watches the *calibration* window, which is later in time than
the training window and earlier than the test window. Watching a random subset
of training days instead — the usual default — would measure how well the
model memorises a period rather than how well it transfers to the next one.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .artifacts import Bundle
from .model import (
    alignment_loss,
    apply_masking,
    build_model,
    count_parameters,
    masked_token_loss,
    per_row_masked_loss,
)
from .schema import CONTEXT_CATEGORICAL, CONTEXT_NUMERIC

log = logging.getLogger(__name__)


class UserDayDataset(Dataset):
    """One item per user-day. Everything is already numeric; this just indexes."""

    def __init__(self, bundle: Bundle, rows: np.ndarray | None = None):
        self.rows = np.arange(len(bundle)) if rows is None else np.asarray(rows)
        self.tokens = bundle.tokens
        self.hours = bundle.hours
        self.flags = bundle.flags
        self.content = bundle.content
        cat_cols = [c + "_code" for c in CONTEXT_CATEGORICAL]
        self.cat = bundle.context[cat_cols].to_numpy(dtype=np.int64)
        self.num = bundle.context[CONTEXT_NUMERIC].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = int(self.rows[i])
        return {
            "row": r,
            "tokens": torch.from_numpy(self.tokens[r].astype(np.int64)),
            "hours": torch.from_numpy(self.hours[r].astype(np.int64)),
            "flags": torch.from_numpy(self.flags[r].astype(np.float32)),
            "content": torch.from_numpy(self.content[r].astype(np.float32)),
            "cat": torch.from_numpy(self.cat[r]),
            "num": torch.from_numpy(self.num[r]),
        }


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0] if k != "row"}
    out["row"] = torch.tensor([b["row"] for b in batch])
    return out


@dataclass
class TrainingReport:
    history: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_loss: float = float("inf")
    n_parameters: int = 0
    seconds: float = 0.0

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


def _split_rows(bundle: Bundle, name: str) -> np.ndarray:
    return np.nonzero((bundle.index["split"] == name).to_numpy())[0]


def context_cardinalities(bundle: Bundle) -> tuple[int, ...]:
    cols = [c + "_code" for c in CONTEXT_CATEGORICAL]
    return tuple(int(bundle.context[c].max()) + 1 for c in cols)


def train(
    bundle: Bundle,
    model_cfg: dict,
    train_cfg: dict,
    seed: int,
    device: str = "cpu",
) -> tuple[torch.nn.Module, TrainingReport]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_rows = _split_rows(bundle, "train")
    calib_rows = _split_rows(bundle, "calibrate")
    if len(train_rows) == 0:
        raise ValueError("the training split is empty")

    # Subsample the training window if it is very large. The objective is
    # self-supervised and the marginal value of the two-hundred-thousandth
    # ordinary Tuesday is close to zero, while the wall-clock cost is not.
    # Sampling is stratified by user so nobody is dropped entirely.
    cap = int(train_cfg.get("max_train_rows", 0) or 0)
    if cap and len(train_rows) > cap:
        rng = np.random.default_rng(seed)
        users = bundle.index["user"].to_numpy()[train_rows]
        order = rng.permutation(len(train_rows))
        train_rows_shuffled = train_rows[order]
        per_user = pd.Series(users[order]).groupby(
            pd.Series(users[order])).cumcount().to_numpy()
        quota = max(1, cap // max(1, len(np.unique(users))))
        chosen = train_rows_shuffled[per_user < quota]
        if len(chosen) > cap:
            chosen = chosen[:cap]
        log.info("subsampled the training window from %d to %d user-days "
                 "(<= %d per user)", len(train_rows), len(chosen), quota)
        train_rows = np.sort(chosen)

    model = build_model(
        model_cfg,
        content_dim=bundle.content.shape[-1],
        context_cardinalities=context_cardinalities(bundle),
        n_context_numeric=len(CONTEXT_NUMERIC),
    ).to(device)

    report = TrainingReport(n_parameters=count_parameters(model))
    log.info("model has %s trainable parameters", f"{report.n_parameters:,}")

    loaders = {
        "train": DataLoader(UserDayDataset(bundle, train_rows),
                            batch_size=int(train_cfg["batch_size"]), shuffle=True,
                            collate_fn=collate, drop_last=True),
        "calibrate": DataLoader(UserDayDataset(bundle, calib_rows),
                                batch_size=int(train_cfg["batch_size"]),
                                shuffle=False, collate_fn=collate),
    }

    opt = torch.optim.AdamW(model.parameters(),
                            lr=float(train_cfg["learning_rate"]),
                            weight_decay=float(train_cfg["weight_decay"]))
    total_steps = max(1, len(loaders["train"]) * int(train_cfg["epochs"]))
    warmup = int(train_cfg["warmup_steps"])

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    mask_p = float(model_cfg["mask_probability"])
    align_w = float(model_cfg["alignment_weight"])
    temp = float(model_cfg["temperature"])
    clip = float(train_cfg["grad_clip"])

    best_state, patience_left = None, int(train_cfg["patience"])
    started = time.time()

    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        sums = {"mlm": 0.0, "align": 0.0, "n": 0}
        for batch in loaders["train"]:
            masked, selected = apply_masking(batch["tokens"], mask_p)
            out = model(masked, batch["hours"], batch["flags"],
                        batch["content"], batch["cat"], batch["num"])
            mlm = masked_token_loss(out["logits"], batch["tokens"], selected)
            align = alignment_loss(out["z_behaviour"], out["z_content"], temp)
            loss = mlm + align_w * align

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            scheduler.step()

            sums["mlm"] += float(mlm) * len(batch["row"])
            sums["align"] += float(align) * len(batch["row"])
            sums["n"] += len(batch["row"])

        val = _evaluate_loss(model, loaders["calibrate"], mask_p, align_w, temp)
        row = {
            "epoch": epoch,
            "train_mlm": sums["mlm"] / max(1, sums["n"]),
            "train_align": sums["align"] / max(1, sums["n"]),
            **{f"calib_{k}": v for k, v in val.items()},
            "lr": scheduler.get_last_lr()[0],
        }
        report.history.append(row)
        log.info("epoch %2d  train mlm %.4f align %.4f | calib total %.4f",
                 epoch, row["train_mlm"], row["train_align"], val["total"])

        if val["total"] < report.best_loss - 1e-4:
            report.best_loss, report.best_epoch = val["total"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = int(train_cfg["patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("stopping early at epoch %d; best was %d",
                         epoch, report.best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    report.seconds = time.time() - started
    log.info("trained in %.1f minutes; best calibration loss %.4f at epoch %d",
             report.seconds / 60, report.best_loss, report.best_epoch)
    return model, report


@torch.no_grad()
def _evaluate_loss(model, loader, mask_p, align_w, temp) -> dict[str, float]:
    model.eval()
    # A fixed generator so the calibration loss is comparable between epochs.
    gen = torch.Generator().manual_seed(0)
    sums = {"mlm": 0.0, "align": 0.0, "n": 0}
    for batch in loader:
        masked, selected = apply_masking(batch["tokens"], mask_p, generator=gen)
        out = model(masked, batch["hours"], batch["flags"],
                    batch["content"], batch["cat"], batch["num"])
        mlm = masked_token_loss(out["logits"], batch["tokens"], selected)
        align = alignment_loss(out["z_behaviour"], out["z_content"], temp)
        sums["mlm"] += float(mlm) * len(batch["row"])
        sums["align"] += float(align) * len(batch["row"])
        sums["n"] += len(batch["row"])
    n = max(1, sums["n"])
    return {"mlm": sums["mlm"] / n, "align": sums["align"] / n,
            "total": sums["mlm"] / n + align_w * sums["align"] / n}


@torch.no_grad()
def embed_and_score(
    model, bundle: Bundle, mask_p: float, seed: int, n_masking_passes: int = 3,
    batch_size: int = 256, score_rows: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Run every user-day through the model and return the raw signals.

    Two passes with different costs, separated on purpose. The *embedding*
    pass runs once over everything, because the peer and self baselines are
    fitted on training-window embeddings and so those are needed too. The
    *masked-modelling* pass runs ``n_masking_passes`` times and only over the
    rows that will actually be scored — typically the calibration and test
    windows. On the full release that separation is the difference between an
    hour and three.

    The masked score is averaged over several independent masking patterns
    because a single pattern gives an estimate that depends on which positions
    happened to be hidden. Three is enough to stabilise the ranking. The
    generator is seeded, so the same bundle always produces the same scores.
    """
    model.eval()
    n = len(bundle)
    if score_rows is None:
        score_rows = np.arange(n)
    needs_score = np.zeros(n, dtype=bool)
    needs_score[np.asarray(score_rows)] = True

    day_vectors = np.zeros((n, model.dims.d_model), dtype=np.float32)
    mlm_scores = np.zeros(n, dtype=np.float64)
    disagreement = np.zeros(n, dtype=np.float64)

    loader = DataLoader(UserDayDataset(bundle), batch_size=batch_size,
                        shuffle=False, collate_fn=collate)
    started = time.time()
    for i, batch in enumerate(loader):
        rows = batch["row"].numpy()
        clean = model(batch["tokens"], batch["hours"], batch["flags"],
                      batch["content"], batch["cat"], batch["num"])
        day_vectors[rows] = clean["day"].numpy()
        disagreement[rows] = (
            1.0 - (clean["z_behaviour"] * clean["z_content"]).sum(-1)).numpy()

        want = needs_score[rows]
        if not want.any():
            continue
        acc = np.zeros(int(want.sum()), dtype=np.float64)
        sub = {k: v[torch.from_numpy(want)] for k, v in batch.items()
               if k != "row"}
        for p in range(n_masking_passes):
            gen = torch.Generator().manual_seed(seed + p)
            masked, selected = apply_masking(sub["tokens"], mask_p, generator=gen)
            out = model(masked, sub["hours"], sub["flags"],
                        sub["content"], sub["cat"], sub["num"])
            acc += per_row_masked_loss(
                out["logits"], sub["tokens"], selected).numpy()
        mlm_scores[rows[want]] = acc / n_masking_passes

        if i and i % 200 == 0:
            log.info("  scored %d/%d user-days (%.1f min elapsed)",
                     min((i + 1) * batch_size, n), n, (time.time() - started) / 60)

    return {
        "day": day_vectors,
        "masked_behaviour": mlm_scores,
        "cross_modal_disagreement": disagreement,
    }
