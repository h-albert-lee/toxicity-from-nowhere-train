"""Extraction runner for δ_int activations."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List
import time
from contextlib import nullcontext
from itertools import islice

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq
from tqdm import tqdm

from ..utils.config import RootConfig
from ..data.memesafety import Labeler, MemeSafetyBenchAdapter, load_label_mapping
from ..data.siuo import SIUOAdapter
from ..data.mdit_triple import MDITTripleAdapter
from ..utils.env import RunMetadata, resolve_repo_sha, save_metadata, set_global_seed
from ..utils.logging import get_logger
from ..models.hooks import ResidualCapture, cross_attention_off, install_post_attention_hook, remove_hooks
from .pooling import POOLING_FNS
from .sharding import ShardWriter


LOGGER = get_logger(__name__)


def load_image(image_ref) -> Image.Image:
    if isinstance(image_ref, Image.Image):
        return image_ref
    if isinstance(image_ref, dict):
        if "path" in image_ref and image_ref["path"]:
            return Image.open(image_ref["path"]).convert("RGB")
        if "bytes" in image_ref:
            return Image.open(BytesIO(image_ref["bytes"])).convert("RGB")
    return Image.open(image_ref).convert("RGB")


class ExtractRunner:
    def __init__(self, cfg: RootConfig) -> None:
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.model.hf_id)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            cfg.model.hf_id, dtype=_mixed_precision_dtype(cfg.model.dtype), device_map=cfg.model.device_map
        )
        self.model.eval()
        set_global_seed(cfg.repro.seed)

    def run(self) -> None:
        dataset_cfg = self.cfg.data
        
        # Get first dataset name
        dataset_name = dataset_cfg.datasets[0] if dataset_cfg.datasets else "memesafety_test"
        
        # Create labeler for datasets that need it
        label_mapping = load_label_mapping(None)  # Use default mapping
        labeler = Labeler(label_mapping, "sentiment")
        
        # Select adapter based on dataset name
        if dataset_name.startswith("mdit_triple"):
            if not dataset_cfg.root:
                raise ValueError("MDIT-Bench-Triple dataset requires data.root to be specified in config")
            
            LOGGER.info(f"Using MDIT-Triple adapter with data_dir={dataset_cfg.root}, split={dataset_cfg.split}")
            adapter = MDITTripleAdapter(
                data_dir=dataset_cfg.root,
                split=dataset_cfg.split,
                text_mode="question_answer"
            )
            labeler = None  # MDIT-Triple has labels built-in
        elif dataset_name.startswith("siuo"):
            # SIUO dataset - use local adapter
            if not dataset_cfg.root:
                raise ValueError("SIUO dataset requires data.root to be specified in config")
            
            # Extract data type from name (e.g., "siuo_gen" -> "gen")
            data_type = dataset_name.split("_", 1)[1] if "_" in dataset_name else "gen"
            
            LOGGER.info(f"Using SIUO adapter with data_dir={dataset_cfg.root}, data_type={data_type}")
            adapter = SIUOAdapter(data_dir=dataset_cfg.root, data_type=data_type)
            labeler = None  # SIUO doesn't use labeler
        else:
            # HuggingFace dataset - use MemeSafetyBenchAdapter
            # Map short names to full HF names
            if dataset_name == "memesafety_test":
                hf_name = "oneonlee/Meme-Safety-Bench"
            else:
                hf_name = dataset_name
            
            LOGGER.info(f"Using MemeSafetyBench adapter with HF dataset: {hf_name}")
            adapter = MemeSafetyBenchAdapter(
                hf_name,
                dataset_cfg.split,
                image_column="meme_image",  # Standard for Meme-Safety-Bench
                instruction_column="instruction",
                streaming=False,
            )
        
        output_dir = Path(self.cfg.extract.save.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = RunMetadata(
            repo_sha=resolve_repo_sha(),
            model_name=self.cfg.model.hf_id,
            dataset_name=dataset_name,
            dataset_split=dataset_cfg.split,
            seed=self.cfg.repro.seed,
        )
        save_metadata(metadata, output_dir / "metadata.json")
        
        # Build prompt template
        prompt_template = "<image>\n{instruction}"
        
        # For MDIT-Triple, we know the dataset size without loading samples
        if dataset_name.startswith("mdit_triple"):
            # Get dataset size directly from the adapter's dataset attribute
            total_available = len(adapter.dataset)
            LOGGER.info(f"Dataset size: {total_available} samples")
            
            # Limit samples if configured
            if dataset_cfg.max_samples and dataset_cfg.max_samples < total_available:
                total_samples = dataset_cfg.max_samples
                LOGGER.info(f"Limiting to {total_samples} samples")
            else:
                total_samples = total_available
            
            # Create iterator (don't load all into memory)
            samples_iter = adapter.iter_samples(prompt_template, labeler)
        else:
            # For other datasets, keep old behavior
            LOGGER.info("Counting samples...")
            samples_list = list(adapter.iter_samples(prompt_template, labeler))
            
            # Limit samples if configured
            if dataset_cfg.max_samples and dataset_cfg.max_samples < len(samples_list):
                LOGGER.info(f"Limiting to {dataset_cfg.max_samples} samples (total available: {len(samples_list)})")
                samples_list = samples_list[:dataset_cfg.max_samples]
            
            total_samples = len(samples_list)
            samples_iter = iter(samples_list)
        
        LOGGER.info(f"Processing {total_samples} samples across {len(self.cfg.extract.layers)} layers")
        
        for layer_idx in self.cfg.extract.layers:
            LOGGER.info(f"Extracting layer {layer_idx}")
            shard_writer = ShardWriter(output_dir / f"layer_{layer_idx:02d}", layer_idx, self.cfg.extract.save.shard_size)
            capture = ResidualCapture(keep_on_device=True)
            handles = install_post_attention_hook(self.model, [layer_idx], capture)
            
            # Reset iterator for each layer
            if dataset_name.startswith("mdit_triple"):
                samples_iter = adapter.iter_samples(prompt_template, labeler)
            else:
                samples_iter = iter(samples_list)
            
            try:
                start_time = time.time()
                processed_count = 0
                
                with tqdm(total=total_samples, desc=f"Layer {layer_idx}", unit="sample") as pbar:
                    for sample in samples_iter:
                        # Apply max_samples limit
                        if dataset_cfg.max_samples and processed_count >= dataset_cfg.max_samples:
                            break
                        
                        sample_start = time.time()
                        delta = self._extract_delta(sample, layer_idx, capture)
                        shard_writer.add(delta, self._sanitize_metadata(sample))
                        capture.clear()
                        
                        # Update progress
                        sample_time = time.time() - sample_start
                        processed_count += 1
                        avg_time = (time.time() - start_time) / processed_count
                        
                        pbar.set_postfix({
                            'sample_time': f'{sample_time:.2f}s',
                            'avg': f'{avg_time:.2f}s/sample'
                        })
                        pbar.update(1)
                        
            finally:
                total_time = time.time() - start_time
                if processed_count > 0:
                    LOGGER.info(f"Layer {layer_idx} completed in {total_time:.1f}s ({total_time/processed_count:.2f}s/sample)")
                shard_writer.close()
                remove_hooks(handles)

    def _extract_delta_batch(self, batch: List, layer_idx: int, capture: ResidualCapture) -> List[np.ndarray]:
        """Extract delta for a batch of samples (faster than processing one by one)."""
        # Load all images
        images = [load_image(sample.image) for sample in batch]
        texts = [sample.prompt for sample in batch]
        
        # Process batch
        inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)
        
        # Use autocast for faster computation if enabled
        use_amp = self.cfg.extract.amp and self.model.device.type == 'cuda'
        autocast_ctx = torch.amp.autocast('cuda') if use_amp else nullcontext()
        
        # ON pass
        with torch.no_grad(), autocast_ctx:
            _ = self.model(**inputs)
            on_hidden = capture.activations[layer_idx]
        
        capture.clear()
        
        # OFF pass
        with torch.no_grad(), autocast_ctx:
            decoder_layer = self.model.language_model.layers[layer_idx]
            with cross_attention_off(decoder_layer):
                _ = self.model(**inputs)
                off_hidden = capture.activations[layer_idx]
        
        # Compute delta (keep on GPU, convert to float32 for consistency)
        delta_batch = (on_hidden - off_hidden).float()
        
        # Get pooling function and kwargs
        pooling_fn = POOLING_FNS[self.cfg.extract.pooling.name]
        pooling_kwargs = {}
        if self.cfg.extract.pooling.name == "last_k_mean":
            pooling_kwargs["k"] = self.cfg.extract.pooling.last_k
        
        # Get image token id
        image_token = getattr(self.processor.tokenizer, "image_token", None)
        image_token_id = None
        if image_token is not None:
            image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        
        # Process each sample in batch
        results = []
        input_ids = inputs["input_ids"]
        
        for i in range(len(batch)):
            # Filter out image tokens to get content (text) token indices for this sample
            content_indices = [j for j, token in enumerate(input_ids[i].tolist()) 
                             if token != image_token_id and token != self.tokenizer.pad_token_id]
            
            if not content_indices:
                content_indices = list(range(input_ids.shape[1]))
            
            # Get delta for this sample
            delta_sample = delta_batch[i:i+1]  # Keep batch dimension
            
            # Pool on GPU
            pooled = pooling_fn(delta_sample, content_indices, **pooling_kwargs)
            
            # Remove batch dimension if present
            if pooled.dim() == 2 and pooled.shape[0] == 1:
                pooled = pooled.squeeze(0)
            
            # Move to CPU and convert to numpy
            results.append(pooled.cpu().numpy())
        
        return results

    def _extract_delta(self, sample, layer_idx: int, capture: ResidualCapture) -> np.ndarray:
        image = load_image(sample.image)
        
        # Process image and text separately to ensure correct format
        inputs = self.processor(images=image, text=sample.prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        # Use autocast for faster computation if enabled
        use_amp = self.cfg.extract.amp and self.model.device.type == 'cuda'
        autocast_ctx = torch.amp.autocast('cuda') if use_amp else nullcontext()
        
        # ON pass
        with torch.no_grad(), autocast_ctx:
            _ = self.model(**inputs)
            on_hidden = capture.activations[layer_idx]
        
        capture.clear()
        
        # OFF pass
        with torch.no_grad(), autocast_ctx:
            decoder_layer = self.model.language_model.layers[layer_idx]
            with cross_attention_off(decoder_layer):
                _ = self.model(**inputs)
                off_hidden = capture.activations[layer_idx]
        
        # Compute delta (keep on GPU, convert to float32 for consistency)
        delta = (on_hidden - off_hidden).float()
        
        pooling_fn = POOLING_FNS[self.cfg.extract.pooling.name]
        input_ids = inputs["input_ids"]
        
        image_token = getattr(self.processor.tokenizer, "image_token", None)
        image_token_id = None
        if image_token is not None:
            image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        
        # Filter out image tokens to get content (text) token indices
        content_indices = [i for i, token in enumerate(input_ids[0].tolist()) if token != image_token_id]
        if not content_indices:
            content_indices = list(range(input_ids.shape[1]))
        
        # Build pooling kwargs from config
        pooling_kwargs = {}
        if self.cfg.extract.pooling.name == "last_k_mean":
            pooling_kwargs["k"] = self.cfg.extract.pooling.last_k
        
        # Ensure delta has batch dimension [batch_size, seq_len, hidden_dim]
        if delta.dim() == 2:
            # [seq_len, hidden_dim] -> [1, seq_len, hidden_dim]
            delta_batched = delta.unsqueeze(0)
        elif delta.dim() == 3:
            # Already [batch_size, seq_len, hidden_dim]
            delta_batched = delta
        else:
            raise ValueError(f"Unexpected delta shape: {delta.shape}, expected 2D or 3D tensor")
        
        # Pool on GPU
        pooled = pooling_fn(delta_batched, content_indices, **pooling_kwargs)
        
        # Remove batch dimension if present
        if pooled.dim() == 2 and pooled.shape[0] == 1:
            pooled = pooled.squeeze(0)
        
        # Only move to CPU at the very end for numpy conversion
        return pooled.cpu().numpy()

    def _sanitize_metadata(self, sample) -> Dict[str, object]:
        payload: Dict[str, object] = {"label": sample.label}
        if hasattr(sample, 'metadata') and sample.metadata:
            for key, value in sample.metadata.items():
                # Skip image data and large binary fields in metadata
                if key in ["image", "meme_image", "bytes"]:
                    continue
                payload[key] = _to_jsonable(value)
        return payload


def _mixed_precision_dtype(mode: str) -> torch.dtype:
    mode = mode.lower()
    if mode in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if mode in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


__all__ = ["ExtractRunner"]


def _to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)
