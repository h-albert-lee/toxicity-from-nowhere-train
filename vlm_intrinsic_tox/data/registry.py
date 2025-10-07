"""Dataset registry and factory functions."""

from __future__ import annotations

from typing import Iterable, Optional

from ..utils.logging import get_logger
from .memesafety import MemeSafetyBenchAdapter, Sample, Labeler, load_label_mapping
from .siuo import SIUOAdapter
from .mdit import MDITBenchAdapter
from .mdit_triple import MDITTripleAdapter

LOGGER = get_logger(__name__)


def get_dataset(name: str, split: str, root: Optional[str] = None) -> Iterable[Sample]:
    """Get dataset iterator by name.
    
    Args:
        name: Dataset name (e.g., 'memesafety_test')
        split: Dataset split (e.g., 'test', 'train')
        root: Optional root directory (unused for HF datasets)
        
    Returns:
        Iterable of Sample objects
        
    Raises:
        ValueError: If dataset name is not recognized
    """
    if name == "memesafety_test" or name.startswith("oneonlee/Meme-Safety-Bench"):
        # Handle both short name and full HF name
        hf_name = "oneonlee/Meme-Safety-Bench" if name == "memesafety_test" else name
        
        # Create adapter
        adapter = MemeSafetyBenchAdapter(
            name=hf_name,
            split=split,
            image_column="meme_image",
            instruction_column="instruction", 
            streaming=False
        )
        
        # Create labeler with default mapping
        label_mapping = load_label_mapping(None)  # Use default mapping
        labeler = Labeler(label_mapping, "sentiment")  # Use sentiment column for labeling
        
        # Default prompt template
        prompt_template = "<image>\n{instruction}"
        
        # Return samples with proper ID generation
        for idx, sample in enumerate(adapter.iter_samples(prompt_template, labeler)):
            # Convert to match specification: add id, change prompt to instruction
            yield Sample(
                id=f"{name}_{split}_{idx}",
                image=sample.image,
                instruction=sample.prompt.replace("<image>\n", ""),  # Remove image token from instruction
                label=sample.label,
                meta=sample.metadata
            )
    
    elif name.startswith("siuo"):
        # Handle SIUO dataset
        if root is None:
            raise ValueError("SIUO dataset requires root directory to be specified")
        
        # Extract data type from name (e.g., "siuo_gen" -> "gen")
        data_type = name.split("_", 1)[1] if "_" in name else "gen"
        
        adapter = SIUOAdapter(data_dir=root, data_type=data_type)
        yield from adapter
    
    elif name.startswith("mdit_triple"):
        # Handle MDIT-Bench-Triple dataset (322K samples with 3 toxicity levels)
        if root is None:
            raise ValueError("MDIT-Bench-Triple dataset requires root directory to be specified")
        
        # Default prompt template
        prompt_template = "<image>\n{instruction}"
        
        adapter = MDITTripleAdapter(
            data_dir=root,
            split=split,
            text_mode="question_answer"
        )
        
        for sample in adapter.iter_samples(prompt_template, labeler=None):
            yield Sample(
                id=sample.id,
                image=sample.image,
                instruction=sample.metadata["instruction"],
                label=sample.label,  # 0=benign, 1=explicit, 2=implicit
                meta=sample.metadata
            )
    
    elif name.startswith("mdit"):
        # Handle MDIT-Bench dataset (original)
        if root is None:
            raise ValueError("MDIT-Bench dataset requires root directory to be specified")
        
        # Extract categories if specified (e.g., "mdit_age" -> ["age"])
        # or use all categories if just "mdit"
        categories = None
        if "_" in name:
            category = name.split("_", 1)[1]
            categories = [category] if category else None
        
        # Default prompt template
        prompt_template = "<image>\n{instruction}"
        
        adapter = MDITBenchAdapter(
            data_dir=root, 
            categories=categories,
            text_mode="question_only"
        )
        
        for sample in adapter.iter_samples(prompt_template, labeler=None):
            yield Sample(
                id=sample.id,
                image=sample.image,
                instruction=sample.metadata["instruction"],
                label=sample.label,  # Already set to 2 (intrinsic) in adapter
                meta=sample.metadata
            )
    
    else:
        raise ValueError(f"Unknown dataset name: {name}")


# Registry of available datasets
DATASET_REGISTRY = {
    "memesafety_test": {
        "hf_name": "oneonlee/Meme-Safety-Bench",
        "splits": ["test", "train"],
        "toxicity_type": "mixed",  # Contains both extrinsic toxic and benign
        "description": "Meme Safety Benchmark dataset"
    },
    "siuo_gen": {
        "data_type": "gen",
        "splits": ["test"],
        "toxicity_type": "intrinsic",  # Intrinsic toxicity
        "description": "SIUO dataset - generated content"
    },
    "mdit_triple": {
        "splits": ["train", "test"],
        "toxicity_type": "multi_level",  # 0=benign, 1=explicit, 2=implicit
        "num_samples": 322921,
        "description": "MDIT-Bench-Triple - 3 toxicity levels per question group (benign/explicit/implicit)"
    },
    "mdit": {
        "splits": ["test"],  # MDIT doesn't have explicit splits
        "toxicity_type": "intrinsic",  # Intrinsic toxicity (image+text bias)
        "categories": ["age", "gender", "behavior", "religion", "disability", 
                      "racial_discrimination", "region_discrimination", "sexual_orientation"],
        "description": "MDIT-Bench - Multimodal Dual-Implicit Toxicity Benchmark"
    }
}


def list_datasets() -> dict:
    """List all available datasets in the registry.
    
    Returns:
        Dictionary of dataset information
    """
    return DATASET_REGISTRY.copy()


__all__ = ["get_dataset", "list_datasets", "DATASET_REGISTRY"]