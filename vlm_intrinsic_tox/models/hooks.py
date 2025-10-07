"""Hooks for capturing residual streams and disabling cross attention."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch

from ..utils.logging import get_logger
from .llava_guard import ModelInputs

LOGGER = get_logger(__name__)


class ResidualCapture:
    """Callable hook to capture decoder layer residual streams."""

    def __init__(self, keep_on_device: bool = True) -> None:
        self.activations: Dict[int, torch.Tensor] = {}
        self.keep_on_device = keep_on_device

    def __call__(self, layer_idx: int, hidden: torch.Tensor) -> None:
        if self.keep_on_device:
            # Keep on GPU for faster processing
            self.activations[layer_idx] = hidden.detach()
        else:
            # Move to CPU to save GPU memory
            self.activations[layer_idx] = hidden.detach().to("cpu")

    def clear(self) -> None:
        self.activations.clear()


@contextmanager
def cross_attention_off(layer) -> Iterator[None]:
    """Context manager that zeros the self-attention forward output."""

    original = layer.self_attn.forward

    def _disabled_forward(*args, **kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            output = original(*args, **kwargs)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
        zeros = torch.zeros_like(hidden_states)
        return (zeros, None)

    layer.self_attn.forward = _disabled_forward
    try:
        yield
    finally:
        layer.self_attn.forward = original


def install_post_attention_hook(model, layers: List[int], capture: ResidualCapture) -> List[torch.utils.hooks.RemovableHandle]:
    """Register forward hooks that capture the post-attention residual."""

    handles: List[torch.utils.hooks.RemovableHandle] = []
    decoder_layers = model.language_model.layers
    for idx in layers:
        layer = decoder_layers[idx]

        def _make_hook(i: int):
            def _hook(module, input_, output):
                hidden = output[0] if isinstance(output, tuple) else output
                capture(i, hidden)

            return _hook

        handles.append(layer.register_forward_hook(_make_hook(idx)))
    return handles


def remove_hooks(handles: List[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()


def extract_delta_int(model, inputs: ModelInputs, layers: List[int], capture_point: str, pooling: dict) -> Dict[int, np.ndarray]:
    """Extract delta_int activations by running ON and OFF passes.
    
    Args:
        model: LLaVA model
        inputs: Processed model inputs
        layers: List of layer indices to extract from
        capture_point: Where to capture (post_attn_resid, post_attn_ln, post_mlp)
        pooling: Pooling configuration dict with 'name' and optional kwargs
        
    Returns:
        Dict mapping layer index to pooled delta_int arrays
    """
    from ..extract.pooling import POOLING_FNS
    from .llava_guard import get_text_token_spans
    
    # Move inputs to model device
    device = next(model.parameters()).device
    model_inputs = {
        "input_ids": inputs.input_ids.to(device),
        "attention_mask": inputs.attention_mask.to(device),
        "pixel_values": inputs.pixel_values.to(device),
    }
    if inputs.image_sizes is not None:
        model_inputs["image_sizes"] = inputs.image_sizes.to(device)
    
    results = {}
    
    for layer_idx in layers:
        capture = ResidualCapture()
        
        # Install hook based on capture point
        if capture_point == "post_attn_resid":
            handles = install_post_attention_hook(model, [layer_idx], capture)
        else:
            # For now, default to post_attn_resid
            # TODO: Implement other capture points
            handles = install_post_attention_hook(model, [layer_idx], capture)
        
        try:
            # ON pass - normal forward
            with torch.no_grad():
                _ = model(**model_inputs)
                on_hidden = capture.activations[layer_idx].clone()
            
            capture.clear()
            
            # OFF pass - with cross attention disabled
            decoder_layer = model.language_model.layers[layer_idx]
            with torch.no_grad():
                with cross_attention_off(decoder_layer):
                    _ = model(**model_inputs)
                    off_hidden = capture.activations[layer_idx].clone()
            
            # Compute delta
            delta = (on_hidden - off_hidden).to(torch.float32)
            
            # Apply pooling
            pooling_fn = POOLING_FNS[pooling["name"]]
            
            # Get token spans for pooling
            from ..models.llava_guard import get_text_token_spans
            # We need the tokenizer for this - for now use a simple approach
            # TODO: Pass tokenizer through the function signature
            content_indices = list(range(1, inputs.input_ids.shape[1]))  # Skip first token (BOS)
            
            pooled = pooling_fn(delta.unsqueeze(0), content_indices, **pooling.get("kwargs", {}))
            results[layer_idx] = pooled.squeeze(0).cpu().numpy()
            
        finally:
            remove_hooks(handles)
    
    return results


__all__ = ["ResidualCapture", "cross_attention_off", "install_post_attention_hook", "remove_hooks", "extract_delta_int"]
