"""SIUO dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class Sample:
    def __init__(self, id: str, image, prompt: str, label: Optional[int], metadata: dict):
        self.id = id
        self.image = image
        self.prompt = prompt
        self.label = label
        self.metadata = metadata


class SIUOAdapter:
    """Adapter for SIUO dataset."""
    
    def __init__(self, data_dir: str | Path, data_type: str = "gen") -> None:
        """Initialize SIUO adapter.
        
        Args:
            data_dir: Path to SIUO dataset directory
            data_type: Type of data to load ("gen" or other types)
        """
        self.data_dir = Path(data_dir)
        self.data_type = data_type
        self.meta_file = self.data_dir / f"siuo_{data_type}.json"
        
        if not self.meta_file.exists():
            raise FileNotFoundError(f"SIUO metadata file not found: {self.meta_file}")
        
        LOGGER.info(f"Loading SIUO dataset from {self.data_dir}")
    
    def iter_samples(self, prompt_template: str, labeler=None) -> Iterator[Sample]:
        """Iterate over SIUO samples.
        
        Args:
            prompt_template: Template for formatting the prompt (e.g., "<image>\n{instruction}")
            labeler: Optional labeler (not used for SIUO)
            
        Yields:
            Sample objects with prompt field
        """
        with open(self.meta_file, "r") as f:
            items = json.load(f)
        
        for i, item in enumerate(items):
            image_path = self.data_dir / "images" / item["image"]
            
            if not image_path.exists():
                LOGGER.warning(f"Image not found: {image_path}")
                continue
            
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                LOGGER.warning(f"Failed to load image {image_path}: {e}")
                continue
            
            # SIUO doesn't have explicit harmful/benign labels
            # We'll set label to None for unsupervised training
            label = None
            
            # Try to infer harmfulness from category if available
            category = item.get("category", "unknown").lower()
            if "harmful" in category or "toxic" in category or "unsafe" in category:
                label = 1
            elif "safe" in category or "benign" in category:
                label = 0
            
            # Format prompt using the template
            instruction = item["question"]
            prompt = prompt_template.format(instruction=instruction)
            
            yield Sample(
                id=f"siuo_{self.data_type}_{i}",
                image=image,
                prompt=prompt,
                label=label,
                metadata={
                    "dataset": "siuo",
                    "data_type": self.data_type,
                    "category": item.get("category", "unknown"),
                    "image_path": str(image_path),
                    "instruction": instruction
                }
            )


def load_siuo(data_dir: str | Path, data_type: str = "gen", limit: Optional[int] = None) -> list[dict]:
    """Load SIUO dataset in the format expected by llava tools.
    
    Args:
        data_dir: Path to SIUO dataset directory
        data_type: Type of data to load
        limit: Optional limit on number of samples
        
    Returns:
        List of dictionaries with 'text', 'image', 'category' keys
    """
    data_dir = Path(data_dir)
    meta_file = data_dir / f"siuo_{data_type}.json"
    
    with open(meta_file, "r") as f:
        items = json.load(f)
    
    if limit:
        items = items[:limit]
    
    out = []
    for item in items:
        image_path = data_dir / "images" / item["image"]
        out.append({
            "text": item["question"],
            "image": str(image_path),
            "category": item.get("category", "unknown"),
        })
    
    return out


__all__ = ["SIUOAdapter", "load_siuo"]
