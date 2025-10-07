"""Report writer for evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

from ..utils.logging import get_logger
from .analyze import LatentSummary


LOGGER = get_logger(__name__)


def write_reports(
        summaries: Iterable[LatentSummary],
        path: Path, 
        top_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summaries = list(summaries)
    json_path = path.with_suffix(".json")
    md_path = path.with_suffix(".md")
    json_payload = [
        {
            "layer": summary.layer,
            "top_latents": summary.top_latents(top_k),
            "dfreq": summary.dfreq.tolist(),
            "dmag": summary.dmag.tolist(),
            "auroc": summary.auroc.tolist(),
            "kl": summary.kl.tolist(),
            "consensus": summary.consensus.tolist(),
        }
        for summary in summaries
    ]
    json_path.write_text(json.dumps(json_payload, indent=2))
    LOGGER.info("Wrote JSON report to %s", json_path)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Intrinsic Toxicity Report\n\n")
        for summary in summaries:
            handle.write(f"## Layer {summary.layer}\n\n")
            top_latents = summary.top_latents(top_k)
            handle.write("| Rank | Latent | Consensus | AUROC | dfreq | dmag | KL |\n")
            handle.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            for rank, idx in enumerate(top_latents, start=1):
                handle.write(
                    f"| {rank} | {idx} | {summary.consensus[idx]:.3f} | {summary.auroc[idx]:.3f} "
                    f"| {summary.dfreq[idx]:.3f} | {summary.dmag[idx]:.3f} | {summary.kl[idx]:.3f} |\n"
                )
            handle.write("\n")
    LOGGER.info("Wrote Markdown report to %s", md_path)


__all__ = ["write_reports"]
