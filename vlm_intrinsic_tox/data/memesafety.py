"""Dataset adapters for intrinsic toxicity extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import datasets

from ..utils.logging import get_logger


LOGGER = get_logger(__name__)


@dataclass
class Sample:
    id: str
    image: str
    prompt: str
    label: Optional[int]
    metadata: Dict[str, object]


class MemeSafetyBenchAdapter:
    """Adapter that yields samples from Meme-Safety-Bench."""

    def __init__(self, name: str, split: str, image_column: str, instruction_column: str, streaming: bool) -> None:
        self.name = name
        self.split = split
        self.image_column = image_column
        self.instruction_column = instruction_column
        self.streaming = streaming
        self._dataset = None

    def _ensure_dataset(self) -> datasets.Dataset:
        if self._dataset is None:
            LOGGER.info("Loading dataset %s split=%s streaming=%s", self.name, self.split, self.streaming)
            self._dataset = datasets.load_dataset(self.name, split=self.split, streaming=self.streaming)
        return self._dataset

    def iter_samples(self, prompt_template: str, labeler: "Labeler") -> Iterator[Sample]:
        dataset = self._ensure_dataset()
        for idx, row in enumerate(dataset):
            image = row[self.image_column]
            instruction = row.get(self.instruction_column, "") or ""
            prompt = prompt_template.format(instruction=instruction)
            label = labeler(row) if labeler else None
            yield Sample(
                id=f"{self.name}_{self.split}_{idx}",
                image=image, 
                prompt=prompt,
                label=label, 
                metadata=dict(row)
            )


class Labeler:
    """Callable labeler used by adapters."""

    def __init__(self, mapping: Optional[Dict[str, int]] = None, column: Optional[str] = None) -> None:
        self.mapping = mapping or {}
        self.column = column

    def __call__(self, row: Dict[str, object]) -> Optional[int]:
        if self.column and self.column in row:
            value = str(row[self.column]).lower()
            if value in self.mapping:
                return self.mapping[value]
        for key, mapped in self.mapping.items():
            if key in row and isinstance(row[key], str):
                value = row[key].lower()
                if value in self.mapping:
                    return self.mapping[value]
        return None


def load_label_mapping(path: Optional[str]) -> Dict[str, int]:
    if not path:
        return {
            "toxic": 1,
            "offensive": 1,
            "abusive": 1,
            "benign": 0,
            "harmless": 0,
            "non-toxic": 0,
        }
    import json

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key).lower(): int(value) for key, value in data.items()}


__all__ = ["MemeSafetyBenchAdapter", "Labeler", "Sample", "load_label_mapping"]
