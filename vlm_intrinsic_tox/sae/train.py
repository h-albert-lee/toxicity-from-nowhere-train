"""Training utilities for the Top-K sparse autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from ..utils.env import set_global_seed
from ..utils.logging import get_logger
from .modules import TopKSAE


LOGGER = get_logger(__name__)


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.clip(self.std, 1e-6, None)


@dataclass
class TrainResult:
    model: TopKSAE
    scaler: Scaler
    history: List[float]


class ActivationDataset(Dataset):
    def __init__(self, data: np.ndarray, scaler: Scaler) -> None:
        self.tensor = torch.from_numpy(scaler.transform(data)).float()

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensor[idx]


def _fit_scaler(shards: List[np.ndarray]) -> Scaler:
    all_data = np.concatenate(shards, axis=0)
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)
    return Scaler(mean=mean, std=std)


def train_sae(layer: int, shards: Iterable[np.ndarray], cfg) -> TrainResult:
    shard_list = list(shards)
    if not shard_list:
        raise ValueError("No shards provided for SAE training")
    scaler = _fit_scaler(shard_list)
    data = np.concatenate(shard_list, axis=0)
    dataset = ActivationDataset(data, scaler)
    val_size = int(len(dataset) * cfg.val_fraction)
    val_size = min(max(val_size, 1), max(len(dataset) - 1, 0))
    train_size = len(dataset) - val_size
    if val_size == 0:
        train_set = dataset
        val_set = ActivationDataset(np.empty((0, data.shape[1]), dtype=np.float32), scaler)
    else:
        train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size)
    d = data.shape[1]
    m = int(d * cfg.latent_ratio)
    model = TopKSAE(d=d, m=m, k=cfg.topk).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    set_global_seed(cfg.seed)
    history: List[float] = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        count = 0
        for batch in train_loader:
            batch = batch.to(cfg.device)
            optimizer.zero_grad()
            _, recon = model(batch)
            loss = F.mse_loss(recon, batch)  # mean over all elements
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.numel()  # sum of all elements
            count += batch.numel()
        train_loss = total_loss / count  # mean per element
        val_loss = _evaluate(model, val_loader, cfg.device)
        history.append(val_loss)
        LOGGER.info("Layer %s epoch %s train_loss=%.6f val_loss=%.6f", layer, epoch, train_loss, val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, scaler=scaler, history=history)


def _evaluate(model: TopKSAE, loader: DataLoader, device: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, recon = model(batch)
            loss = F.mse_loss(recon, batch, reduction="sum")  # sum of all elements
            total += loss.item()
            count += batch.numel()  # total number of elements
    return total / max(count, 1)  # mean per element


__all__ = ["train_sae", "Scaler", "TrainResult"]
