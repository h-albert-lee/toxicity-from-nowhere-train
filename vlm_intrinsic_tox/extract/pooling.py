"""Pooling utilities for residual streams."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch


def pool_text_only_mean(hiddens: torch.Tensor, content_span: Sequence[int], **_: int) -> torch.Tensor:
    if not content_span:
        raise ValueError("content_span must not be empty")
    indices = torch.tensor(content_span, dtype=torch.long, device=hiddens.device)
    selected = hiddens.index_select(1, indices)
    return selected.mean(dim=1)


def pool_eos(hiddens: torch.Tensor, eos_idx: Sequence[int], **_: int) -> torch.Tensor:
    if not eos_idx:
        raise ValueError("eos_idx must not be empty")
    indices = torch.tensor(eos_idx, device=hiddens.device)
    gathered = hiddens.index_select(1, indices)
    return gathered[:, -1, :]


def pool_last_k_mean(hiddens: torch.Tensor, content_span: Sequence[int], k: int = 1, **_: int) -> torch.Tensor:
    if k <= 0:
        raise ValueError("k must be positive")
    indices = content_span[-k:]
    return pool_text_only_mean(hiddens, indices)


def pool_span_mean(hiddens: torch.Tensor, span: Sequence[int], **_: int) -> torch.Tensor:
    if not span:
        raise ValueError("span must not be empty")
    return pool_text_only_mean(hiddens, span)


POOLING_FNS = {
    "text_only_mean": pool_text_only_mean,
    "eos": pool_eos,
    "last_k_mean": pool_last_k_mean,
    "span_mean": pool_span_mean,
}


__all__ = [
    "pool_text_only_mean",
    "pool_eos",
    "pool_last_k_mean",
    "pool_span_mean",
    "POOLING_FNS",
]
