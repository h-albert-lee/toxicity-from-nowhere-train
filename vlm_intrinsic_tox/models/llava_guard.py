"""LLaVA-Guard model loading and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ModelInputs:
    """Structured inputs for the model."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_sizes: Optional[torch.Tensor] = None


@dataclass
class TokenSpans:
    """Token span information for text processing."""
    prefix_span: List[int]
    image_marker_idx: int
    content_span: List[int]


def load_model(cfg) -> Tuple[LlavaForConditionalGeneration, AutoTokenizer, AutoProcessor]:
    """Load HF model/tokenizer/processor.
    
    Args:
        cfg: Configuration object with model settings
        
    Returns:
        Tuple of (model, tokenizer, image_processor)
    """
    LOGGER.info("Loading model %s", cfg.hf_id)
    
    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id)
    processor = AutoProcessor.from_pretrained(cfg.hf_id)
    
    # Set dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(cfg.dtype, torch.bfloat16)
    
    # Load model
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg.hf_id,
        dtype=dtype,
        device_map=cfg.device_map,
    )
    model.eval()
    
    return model, tokenizer, processor


def build_inputs(instruction: str, image: Image.Image, processor: AutoProcessor, prompt_template: str = "llavaguard_default") -> ModelInputs:
    """Build prompt and process inputs.
    
    Args:
        instruction: Text instruction
        image: PIL Image
        processor: HuggingFace processor
        prompt_template: Template name (currently only supports default)
        
    Returns:
        ModelInputs with processed tensors
    """
    # Apply prompt template
    if prompt_template == "llavaguard_default":
        # Standard LLaVA-Guard format with <image> token
        prompt = f"<image>\n{instruction}"
    else:
        # Fallback to simple format
        prompt = f"<image>\n{instruction}"
    
    # Process inputs
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    )
    
    return ModelInputs(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        image_sizes=inputs.get("image_sizes")
    )


def get_text_token_spans(token_ids: torch.Tensor, tokenizer: AutoTokenizer) -> TokenSpans:
    """Get token span indices for different parts of the input.
    
    Args:
        token_ids: Tokenized input IDs [1, seq_len]
        tokenizer: HuggingFace tokenizer
        
    Returns:
        TokenSpans with prefix_span, image_marker_idx, content_span indices
    """
    token_list = token_ids[0].tolist()
    
    # Find image token
    image_token = getattr(tokenizer, "image_token", "<image>")
    image_token_id = tokenizer.convert_tokens_to_ids(image_token) if hasattr(tokenizer, "convert_tokens_to_ids") else None
    
    # If we can't find the image token ID, try to find it by decoding
    if image_token_id is None:
        for i, token_id in enumerate(token_list):
            decoded = tokenizer.decode([token_id])
            if "<image>" in decoded:
                image_token_id = token_id
                break
    
    image_marker_idx = -1
    if image_token_id is not None:
        try:
            image_marker_idx = token_list.index(image_token_id)
        except ValueError:
            # Image token not found, assume it's at the beginning
            image_marker_idx = 0
    else:
        # Fallback: assume image token is at position 1 (after BOS)
        image_marker_idx = 1 if len(token_list) > 1 else 0
    
    # Define spans
    prefix_span = list(range(0, image_marker_idx))
    
    # Content span: tokens after image marker, excluding special tokens
    special_tokens = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    special_tokens.discard(None)  # Remove None values
    
    content_span = []
    for i in range(image_marker_idx + 1, len(token_list)):
        if token_list[i] not in special_tokens:
            content_span.append(i)
    
    return TokenSpans(
        prefix_span=prefix_span,
        image_marker_idx=image_marker_idx,
        content_span=content_span
    )


__all__ = ["load_model", "build_inputs", "get_text_token_spans", "ModelInputs", "TokenSpans"]
