"""MDIT-Bench-Triple dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from datasets import load_dataset
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


class MDITTripleAdapter:
    """Adapter for MDIT-Bench-Triple dataset.
    
    MDIT-Bench-Triple contains 3 variations per question group:
    - benign (label=0): Neutral, non-toxic answer
    - explicit (label=1): Explicitly toxic answer  
    - implicit (label=2): Implicitly toxic answer
    
    Dataset structure (HF format):
    - id: unique identifier (e.g., '1_benign', '1_explicit', '1_implicit')
    - group_id: question group identifier
    - question: The question text
    - answer: The answer text (varies by toxicity label)
    - text: Combined Q&A text
    - image_path: Relative path to image
    - label: 0 (benign), 1 (explicit), 2 (implicit)
    - label_str: String version of label
    - replace_word: Target demographic/group word
    - source: Origin dataset name
    - split: train/test split
    """
    
    def __init__(
        self, 
        data_dir: str | Path,
        split: str = "train",
        text_mode: str = "question_answer",
        omitted_log: Optional[str | Path] = None
    ) -> None:
        """Initialize MDIT-Bench-Triple adapter.
        
        Args:
            data_dir: Path to local MDIT-Bench-Triple dataset directory
            split: Dataset split ('train' or 'test')
            text_mode: How to construct text:
                - "question_only": Use question only
                - "question_answer" (default): Use question + answer
                - "answer_only": Use answer only
            omitted_log: Optional path to log file for missing images.
                        If None, creates 'mdit_triple_omitted_{split}.txt' in data_dir
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.text_mode = text_mode
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"MDIT-Bench-Triple directory not found: {self.data_dir}")
        
        # Setup omitted images log file
        if omitted_log is None:
            self.omitted_log = self.data_dir / f"mdit_triple_omitted_{split}.txt"
        else:
            self.omitted_log = Path(omitted_log)
        
        # Open log file for appending (create if not exists)
        self.omitted_file = open(self.omitted_log, 'a', encoding='utf-8')
        LOGGER.info(f"Logging omitted images to: {self.omitted_log}")
        
        LOGGER.info(f"Loading MDIT-Bench-Triple from {self.data_dir}, split={split}")
        
        # Load dataset from local directory using HuggingFace datasets
        try:
            self.dataset = load_dataset(str(self.data_dir), split=split)
            LOGGER.info(f"Loaded {len(self.dataset)} samples from MDIT-Bench-Triple {split} split")
        except Exception as e:
            raise RuntimeError(f"Failed to load MDIT-Bench-Triple dataset: {e}")
    
    def iter_samples(self, prompt_template: str, labeler=None) -> Iterator[Sample]:
        """Iterate over MDIT-Bench-Triple samples.
        
        Args:
            prompt_template: Template for formatting the prompt (e.g., "<image>\n{instruction}")
            labeler: Optional labeler (ignored - labels already in dataset)
            
        Yields:
            Sample objects with prompt field
        """
        for idx, item in enumerate(self.dataset):
            try:
                # Extract text based on mode
                text = self._extract_text(item)
                if not text:
                    continue
                
                # Load image
                # item['image_path'] looks like: './total_images/age/elderly/image_1.jpg'
                # But actual directory structure has no 'total_images/' folder
                image_rel = item['image_path'].lstrip('./')
                
                # Always remove "total_images/" prefix if present
                if image_rel.startswith("total_images/"):
                    image_rel = image_rel[len("total_images/"):]
                
                # Construct full image path relative to data_dir
                image_path = self.data_dir / image_rel
                
                if not image_path.exists():
                    # Log to file instead of warning
                    self.omitted_file.write(f"{item['id']}\t{image_path}\n")
                    self.omitted_file.flush()  # Ensure immediate write
                    continue
                
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    # Log to file instead of warning
                    self.omitted_file.write(f"{item['id']}\t{image_path}\tERROR: {e}\n")
                    self.omitted_file.flush()
                    continue
                
                # Label already in dataset: 0=benign, 1=explicit, 2=implicit
                label = item['label']
                
                # Format prompt using the template
                prompt = prompt_template.format(instruction=text)
                
                yield Sample(
                    id=item['id'],
                    image=image,
                    prompt=prompt,
                    label=label,
                    metadata={
                        "dataset": "mdit_bench_triple",
                        "group_id": item['group_id'],
                        "toxicity_type": item['label_str'],
                        "image_path": str(image_path),
                        "instruction": text,
                        "question": item['question'],
                        "answer": item['answer'],
                        "replace_word": item.get('replace_word', ''),
                        "source": item.get('source', 'MDIT-Bench'),
                        "split": item.get('split', self.split),
                    }
                )
                
            except Exception as e:
                LOGGER.warning(f"Failed to process item {idx}: {e}")
                continue
    
    def __del__(self):
        """Close the omitted log file on cleanup."""
        if hasattr(self, 'omitted_file') and self.omitted_file and not self.omitted_file.closed:
            self.omitted_file.close()
    
    def _extract_text(self, item: dict) -> str:
        """Extract text from item based on text_mode setting.
        
        Args:
            item: Dataset item with question and answer
            
        Returns:
            Extracted text string
        """
        if self.text_mode == "question_only":
            return item.get("question", "")
        
        elif self.text_mode == "question_answer":
            # Use question + answer (default)
            question = item.get("question", "")
            answer = item.get("answer", "")
            return f"{question}\n{answer}"
        
        elif self.text_mode == "answer_only":
            return item.get("answer", "")
        
        else:
            LOGGER.warning(f"Unknown text_mode: {self.text_mode}, using question_answer")
            question = item.get("question", "")
            answer = item.get("answer", "")
            return f"{question}\n{answer}"


def load_mdit_triple(
    data_dir: str | Path, 
    split: str = "train",
    limit: Optional[int] = None
) -> list[dict]:
    """Load MDIT-Bench-Triple dataset in a simple format.
    
    Args:
        data_dir: Path to MDIT-Bench-Triple dataset directory
        split: Dataset split ('train' or 'test')
        limit: Optional limit on number of samples
        
    Returns:
        List of dictionaries with 'text', 'image', 'label', 'label_str' keys
    """
    adapter = MDITTripleAdapter(data_dir, split=split)
    prompt_template = "<image>\n{instruction}"
    
    samples = []
    for sample in adapter.iter_samples(prompt_template, labeler=None):
        samples.append({
            "text": sample.metadata["instruction"],
            "image": str(sample.metadata["image_path"]),
            "label": sample.label,
            "label_str": sample.metadata["toxicity_type"],
            "group_id": sample.metadata["group_id"],
        })
        
        if limit and len(samples) >= limit:
            break
    
    return samples


__all__ = ["MDITTripleAdapter", "load_mdit_triple", "Sample"]

