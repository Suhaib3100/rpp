"""Training loop with early stopping, checkpointing and crash recovery."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import ClassificationMetrics, compute_metrics
from src.utils.device import DeviceInfo, autocast_context


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    label_smoothing: float = 0.05
    warmup_epochs: int = 2
    patience: int = 8
    select_metric: str = "macro_f1"   # model selection criterion
    grad_clip: float = 5.0
    seed: int = 42


class Trainer:
    """Trains a model and keeps the best checkpoint by validation metric.

    The test set is never seen here: ``fit`` only touches train and val.
    Checkpoints store optimiser and scheduler state so an interrupted run
    resumes exactly where it stopped (``resume=True``).
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
        device: DeviceInfo,
        out_dir: Path,
        num_classes: int = 6,
    ) -> None:
        self.model = model.to(device.device)
        self.cfg = cfg
        self.dev = device
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.history: list[dict] = []
        # (selection metric, -val_loss): compared lexicographically so that
        # validation loss breaks ties once the metric saturates.
        self.best_score: tuple[float, float] = (-float("inf"), -float("inf"))
        self.best_epoch = -1
        self.start_epoch = 0

        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.amp)
        self.scheduler = None  # built in fit(), needs steps_per_epoch

    # ------------------------------------------------------------- schedule
    def _build_scheduler(self, steps_per_epoch: int) -> None:
        total = self.cfg.epochs * steps_per_epoch
        warm = self.cfg.warmup_epochs * steps_per_epoch

        def lr_lambda(step: int) -> float:
            if step < warm:
                return (step + 1) / max(1, warm)
            prog = (step - warm) / max(1, total - warm)
            return 0.5 * (1.0 + np.cos(np.pi * min(1.0, prog)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------ one epoch
    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x = x.to(self.dev.device, non_blocking=True)
            y = y.to(self.dev.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.dev):
                loss = self.criterion(self.model(x), y)
            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler:
                self.scheduler.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        return total / max(1, n)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, ClassificationMetrics, np.ndarray]:
        """Returns (mean loss, metrics, logits). Used for val and, once, test."""
        self.model.eval()
        total, n = 0.0, 0
        preds, trues, all_logits = [], [], []
        for x, y in loader:
            x = x.to(self.dev.device, non_blocking=True)
            y = y.to(self.dev.device, non_blocking=True)
            with autocast_context(self.dev):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            total += loss.item() * x.size(0)
            n += x.size(0)
            all_logits.append(logits.float().cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(y.cpu().numpy())
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(trues)
        return (total / max(1, n),
                compute_metrics(y_true, y_pred, self.num_classes),
                np.concatenate(all_logits))

    # ------------------------------------------------------------------ fit
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            resume: bool = True) -> dict:
        self._build_scheduler(len(train_loader))
        ckpt_last = self.out / "last.pt"
        if resume and ckpt_last.exists():
            self._load(ckpt_last)
            print(f"[resume] epoch {self.start_epoch}, best={self.best_score[0]:.4f}")

        stale = 0
        t_start = time.time()
        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()
            tr_loss = self._train_epoch(train_loader)
            va_loss, va_metrics, _ = self.evaluate(val_loader)
            # NEU-CLS saturates: a pretrained backbone can reach macro-F1 1.0 on
            # the 216-image validation split within a couple of epochs, after
            # which the selection metric is tied for every remaining epoch and
            # "best" would just mean "first to get lucky". Validation loss still
            # improves well past that point, so it breaks the tie and we keep the
            # better-calibrated model.
            score = (getattr(va_metrics, self.cfg.select_metric), -va_loss)
            dt = time.time() - t0

            rec = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "val_accuracy": va_metrics.accuracy,
                "val_macro_f1": va_metrics.macro_f1,
                "val_macro_precision": va_metrics.macro_precision,
                "val_macro_recall": va_metrics.macro_recall,
                "lr": self.optimizer.param_groups[0]["lr"],
                "seconds": dt,
            }
            self.history.append(rec)
            improved = score > self.best_score
            if improved:
                self.best_score, self.best_epoch, stale = score, epoch, 0
                self._save(self.out / "best.pt", epoch)
            else:
                stale += 1

            print(f"  ep{epoch:3d} train={tr_loss:.4f} val={va_loss:.4f} "
                  f"acc={va_metrics.accuracy:.4f} f1={va_metrics.macro_f1:.4f} "
                  f"{dt:.0f}s{' *' if improved else ''}", flush=True)

            self._save(ckpt_last, epoch)
            (self.out / "history.json").write_text(json.dumps(self.history, indent=2))

            if stale >= self.cfg.patience:
                print(f"  early stop at epoch {epoch} "
                      f"(no {self.cfg.select_metric} gain for {stale} epochs)")
                break

        return {
            "best_epoch": self.best_epoch,
            "best_val_score": self.best_score[0],
            "best_val_loss": -self.best_score[1],
            "select_metric": self.cfg.select_metric,
            "epochs_run": len(self.history),
            "total_seconds": time.time() - t_start,
            "history": self.history,
            "config": asdict(self.cfg),
        }

    # --------------------------------------------------------- checkpointing
    def _save(self, path: Path, epoch: int) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "config": asdict(self.cfg),
        }, path)

    def _load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.dev.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scheduler") and self.scheduler:
            self.scheduler.load_state_dict(ck["scheduler"])
        self.scaler.load_state_dict(ck["scaler"])
        self.start_epoch = ck["epoch"] + 1
        self.best_score = ck["best_score"]
        self.best_epoch = ck["best_epoch"]
        self.history = ck["history"]

    def load_best(self) -> None:
        ck = torch.load(self.out / "best.pt", map_location=self.dev.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
