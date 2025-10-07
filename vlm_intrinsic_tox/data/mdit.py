"""MDIT-Bench dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional, List

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


class MDITBenchAdapter:
    """Adapter for MDIT-Bench dataset.
    
    MDIT-Bench is a 5-choice QA dataset for evaluating dual-implicit toxicity.
    Since we need (image, text) pairs for SAE training, we extract only the question
    (not the answer choices) to form text prompts. The question combined with the 
    image creates intrinsic toxicity scenarios.
    
    Dataset structure:
    - 23 category JSON files (age, behavior, gender, etc.)
    - Each item has: modified_question, image, ans1-5, right_choice
    - Images show specific groups (elderly, young people, etc.)
    - Questions contain implicit bias/discrimination when combined with images
    """
    
    def __init__(
        self, 
        data_dir: str | Path, 
        categories: Optional[List[str]] = None,
        text_mode: str = "question_only"
    ) -> None:
        """Initialize MDIT-Bench adapter.
        
        Args:
            data_dir: Path to MDIT-Bench dataset directory
            categories: Optional list of category names to load (e.g., ['age', 'gender']).
                       If None, loads all available categories.
            text_mode: How to construct text from QA format:
                - "question_only" (default): Use modified_question only
                - "question_correct": Use question + correct answer
                - "question_wrong": Use question + a wrong answer (for ablation)
        """
        self.data_dir = Path(data_dir)
        self.text_mode = text_mode
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"MDIT-Bench directory not found: {self.data_dir}")
        
        # Find all JSON category files
        all_json_files = sorted(self.data_dir.glob("*.json"))
        
        if categories:
            # Filter to requested categories
            category_set = set(categories)
            self.json_files = [
                f for f in all_json_files 
                if f.stem in category_set
            ]
            missing = category_set - {f.stem for f in self.json_files}
            if missing:
                LOGGER.warning(f"Requested categories not found: {missing}")
        else:
            # Use all categories
            self.json_files = all_json_files
        
        if not self.json_files:
            raise ValueError(f"No JSON files found in {self.data_dir}")
        
        LOGGER.info(f"Loading MDIT-Bench from {self.data_dir}")
        LOGGER.info(f"Found {len(self.json_files)} categories: {[f.stem for f in self.json_files]}")
    
    def iter_samples(self, prompt_template: str, labeler=None) -> Iterator[Sample]:
        """Iterate over MDIT-Bench samples.
        
        Args:
            prompt_template: Template for formatting the prompt (e.g., "<image>\n{instruction}")
            labeler: Optional labeler (not used for MDIT - all are intrinsic toxic)
            
        Yields:
            Sample objects with prompt field
        """
        sample_idx = 0
        
        for json_file in self.json_files:
            category = json_file.stem
            
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    items = json.load(f)
            except Exception as e:
                LOGGER.error(f"Failed to load {json_file}: {e}")
                continue
            
            LOGGER.info(f"Processing category '{category}' with {len(items)} items")
            
            for item in items:
                try:
                    # Extract text based on mode
                    text = self._extract_text(item)
                    if not text:
                        continue
                    
                    # Load image
                    # JSON has paths like "./total_images/age/elderly/image_1.jpg"
                    # but actual structure is "./age/elderly/image_1.jpg"
                    image_rel = item["image"].lstrip("./")
                    
                    # Remove "total_images/" prefix if present
                    if image_rel.startswith("total_images/"):
                        image_rel = image_rel[len("total_images/"):]
                    
                    image_path = self.data_dir / image_rel
                    
                    if not image_path.exists():
                        LOGGER.warning(f"Image not found: {item['image']}")
                        continue
                    
                    try:
                        image = Image.open(image_path).convert("RGB")
                    except Exception as e:
                        LOGGER.warning(f"Failed to load image {image_path}: {e}")
                        continue
                    
                    # MDIT-Bench samples are all intrinsic toxic by design
                    # The question + image together create implicit bias/toxicity
                    label = 2  # intrinsic toxicity
                    
                    # Format prompt using the template
                    prompt = prompt_template.format(instruction=text)
                    
                    yield Sample(
                        id=f"mdit_{category}_{item.get('final_filtered_id', item.get('id', sample_idx))}",
                        image=image,
                        prompt=prompt,
                        label=label,
                        metadata={
                            "dataset": "mdit_bench",
                            "category": category,
                            "toxicity_type": "intrinsic",
                            "image_path": str(image_path),
                            "instruction": text,
                            "origin_question": item.get("origin_question", ""),
                            "replace_word": item.get("replace_word", ""),
                            "right_choice": item.get("right_choice", ""),
                        }
                    )
                    
                    sample_idx += 1
                    
                except Exception as e:
                    LOGGER.warning(f"Failed to process item in {category}: {e}")
                    continue
    
    def _extract_text(self, item: dict) -> str:
        """Extract text from QA item based on text_mode setting.
        
        Args:
            item: JSON item with question and answer choices
            
        Returns:
            Extracted text string
        """
        if self.text_mode == "question_only":
            # Use only the modified question
            return item.get("modified_question", "")
        
        elif self.text_mode == "question_correct":
            # Use question + correct answer
            question = item.get("modified_question", "")
            right_choice = item.get("right_choice", "A")
            
            # Map choice letter to answer field
            choice_map = {"A": "ans1", "B": "ans2", "C": "ans3", "D": "ans4", "E": "ans5"}
            ans_key = choice_map.get(right_choice, "ans1")
            answer = item.get(ans_key, "")
            
            return f"{question}\n{answer}"
        
        elif self.text_mode == "question_wrong":
            # Use question + a wrong answer (for ablation studies)
            question = item.get("modified_question", "")
            right_choice = item.get("right_choice", "A")
            
            # Pick first wrong answer
            choices = ["A", "B", "C", "D", "E"]
            wrong_choices = [c for c in choices if c != right_choice]
            if not wrong_choices:
                return question
            
            choice_map = {"A": "ans1", "B": "ans2", "C": "ans3", "D": "ans4", "E": "ans5"}
            ans_key = choice_map[wrong_choices[0]]
            answer = item.get(ans_key, "")
            
            return f"{question}\n{answer}"
        
        else:
            LOGGER.warning(f"Unknown text_mode: {self.text_mode}, using question_only")
            return item.get("modified_question", "")


def load_mdit(
    data_dir: str | Path, 
    categories: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> list[dict]:
    """Load MDIT-Bench dataset in a simple format.
    
    Args:
        data_dir: Path to MDIT-Bench dataset directory
        categories: Optional list of category names to load
        limit: Optional limit on number of samples
        
    Returns:
        List of dictionaries with 'text', 'image', 'category' keys
    """
    adapter = MDITBenchAdapter(data_dir, categories=categories)
    prompt_template = "<image>\n{instruction}"
    
    samples = []
    for sample in adapter.iter_samples(prompt_template, labeler=None):
        samples.append({
            "text": sample.metadata["instruction"],
            "image": str(sample.metadata["image_path"]),
            "category": sample.metadata["category"],
        })
        
        if limit and len(samples) >= limit:
            break
    
    return samples


__all__ = ["MDITBenchAdapter", "load_mdit", "Sample"]

