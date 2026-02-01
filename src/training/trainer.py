"""
Training Module

End-to-end training pipeline with multi-task loss, constraints, and early stopping.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from tqdm import tqdm

from src.training.loss import MultiTaskLoss


class Trainer:
    """Multi-task model trainer with constraint-aware loss."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "auto",
    ):
        self.model = model
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)
        self.model.to(self.device)

        train_cfg = config.get("training", {})
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.get("learning_rate", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )
        self.loss_fn = MultiTaskLoss(
            weights=train_cfg.get("loss_weights", {}),
            constraint_weight=train_cfg.get("constraint_penalty_weight", 0.1),
            use_constraints=config.get("constraints", {}).get("enabled", True),
        )
        self.epochs = train_cfg.get("epochs", 100)
        self.patience = train_cfg.get("early_stopping_patience", 15)
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _prepare_dataloaders(
        self,
        X: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        batch_size: int = 32,
        val_ratio: float = 0.15,
    ):
        """Create train/val dataloaders."""
        n = len(X)
        val_size = int(n * val_ratio)
        train_size = n - val_size

        X_train, X_val = X[:train_size], X[train_size:]
        t_train = {k: v[:train_size] for k, v in targets.items()}
        t_val = {k: v[train_size:] for k, v in targets.items()}

        train_ds = TensorDataset(X_train, *[t_train[k] for k in sorted(targets.keys())])
        val_ds = TensorDataset(X_val, *[t_val[k] for k in sorted(targets.keys())])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        return train_loader, val_loader, sorted(targets.keys())

    def train_epoch(
        self,
        train_loader: DataLoader,
        target_keys: list,
    ) -> Dict[str, float]:
        """Single training epoch."""
        self.model.train()
        epoch_losses = {k: 0.0 for k in target_keys}
        epoch_losses["constraint"] = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc="Train", leave=False):
            X = batch[0].to(self.device)
            targets = {k: batch[i + 1].to(self.device) for i, k in enumerate(target_keys)}

            self.optimizer.zero_grad()
            pred = self.model(X)

            aux = None
            if hasattr(self, "_aux_data") and self._aux_data is not None:
                aux = {k: v.to(self.device) for k, v in self._aux_data.items()}

            loss, loss_dict = self.loss_fn(pred, targets, aux)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    def validate(
        self,
        val_loader: DataLoader,
        target_keys: list,
    ) -> float:
        """Validation pass."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                X = batch[0].to(self.device)
                targets = {k: batch[i + 1].to(self.device) for i, k in enumerate(target_keys)}
                pred = self.model(X)
                loss, _ = self.loss_fn(pred, targets, None)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def fit(
        self,
        X: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        aux: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, list]:
        """Full training loop."""
        self._aux_data = aux
        batch_size = self.config.get("training", {}).get("batch_size", 32)
        val_ratio = self.config.get("data", {}).get("val_split", 0.15)

        train_loader, val_loader, target_keys = self._prepare_dataloaders(
            X, targets, batch_size, val_ratio
        )

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            train_losses = self.train_epoch(train_loader, target_keys)
            val_loss = self.validate(val_loader, target_keys)

            total_train = sum(v for k, v in train_losses.items() if k != "constraint") + train_losses.get("constraint", 0)
            history["train_loss"].append(total_train)
            history["val_loss"].append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                self.patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1} | Train: {total_train:.4f} | Val: {val_loss:.4f}")

            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if hasattr(self, "best_state"):
            self.model.load_state_dict(self.best_state)
        return history

    def save(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": self.config,
        }, path)
