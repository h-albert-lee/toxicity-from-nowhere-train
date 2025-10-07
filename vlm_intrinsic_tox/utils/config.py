"""Configuration dataclasses and loader for the intrinsic toxicity pipeline."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence
import typing
import yaml


@dataclasses.dataclass
class ModelConfig:
    hf_id: str = "AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    prompt_template: str = "llavaguard_default"


@dataclasses.dataclass
class PoolingConfig:
    name: str = "text_only_mean"
    last_k: int = 5


@dataclasses.dataclass
class SaveConfig:
    out_dir: str = "artifacts/activations/llavaguard"
    shard_size: int = 4000


@dataclasses.dataclass
class ExtractConfig:
    layers: Sequence[int] = dataclasses.field(default_factory=lambda: [0, 2, 4])
    capture_point: str = "post_attn_resid"
    cross_attn_off_impl: str = "zeros"
    pooling: PoolingConfig = dataclasses.field(default_factory=PoolingConfig)
    batch_size: int = 8
    num_workers: int = 4
    amp: bool = True
    save: SaveConfig = dataclasses.field(default_factory=SaveConfig)


@dataclasses.dataclass
class DataConfig:
    datasets: Sequence[str] = dataclasses.field(default_factory=lambda: ["memesafety_test"])
    split: str = "test"
    root: Optional[str] = None
    max_samples: Optional[int] = None  # Limit number of samples for testing


@dataclasses.dataclass
class ReproConfig:
    seed: int = 42


@dataclasses.dataclass
class NormalizationConfig:
    scheme: str = "global_mean_std"


@dataclasses.dataclass
class IOConfig:
    in_dir: str = "artifacts/activations/llavaguard"
    out_dir: str = "artifacts/sae/llavaguard"
    scaler_filename: str = "scaler_layer{l}.json"
    sae_filename: str = "sae_layer{l}.pt"
    activ_dir: str = "artifacts/activations/llavaguard"
    sae_dir: str = "artifacts/sae/llavaguard"
    report_dir: str = "artifacts/reports/llavaguard"


@dataclasses.dataclass
class SAEConfig:
    type: str = "topk"
    latent_ratio: float = 4.0
    topk: int = 64
    weight_decay: float = 0.0
    lr: float = 3e-4
    epochs: int = 12
    batch_size: int = 2048
    val_fraction: float = 0.05
    device: str = "cuda"
    seed: int = 42


@dataclasses.dataclass
class SAETrainConfig:
    sae: SAEConfig = dataclasses.field(default_factory=SAEConfig)
    normalization: NormalizationConfig = dataclasses.field(default_factory=NormalizationConfig)
    layers: Sequence[int] = dataclasses.field(default_factory=lambda: [0, 2, 4])
    io: IOConfig = dataclasses.field(default_factory=IOConfig)


@dataclasses.dataclass
class ConsensusConfig:
    weights: dict = dataclasses.field(default_factory=lambda: {"dfreq": 1.0, "dmag": 1.0, "auroc": 1.0, "kl": 0.5})


@dataclasses.dataclass
class ThresholdConfig:
    active: str = "auto"


@dataclasses.dataclass
class InterventionConfig:
    enabled: bool = True
    method: str = "gate"


@dataclasses.dataclass
class EvalConfig:
    datasets: Sequence[str] = dataclasses.field(default_factory=lambda: ["memesafety_test"])
    layers: Sequence[int] = dataclasses.field(default_factory=lambda: [0, 2, 4])
    metrics: Sequence[str] = dataclasses.field(default_factory=lambda: ["dfreq", "dmag", "auroc", "kl", "consensus"])
    consensus: ConsensusConfig = dataclasses.field(default_factory=ConsensusConfig)
    threshold: ThresholdConfig = dataclasses.field(default_factory=ThresholdConfig)
    topk_latents: int = 50
    intervention: InterventionConfig = dataclasses.field(default_factory=InterventionConfig)
    io: IOConfig = dataclasses.field(default_factory=IOConfig)


@dataclasses.dataclass
class RootConfig:
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    extract: ExtractConfig = dataclasses.field(default_factory=ExtractConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    repro: ReproConfig = dataclasses.field(default_factory=ReproConfig)
    sae: SAETrainConfig = dataclasses.field(default_factory=SAETrainConfig)
    eval: EvalConfig = dataclasses.field(default_factory=EvalConfig)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True))


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(paths: Iterable[Path], overrides: Optional[List[str]] = None) -> RootConfig:
    """Load configuration from one or more YAML files with optional overrides."""
    import ast
    
    merged: dict = {}
    for path in paths:
        merged = _merge_dicts(merged, _load_yaml(path))
    
    # Apply overrides (e.g., "extract.layers=[0,2,4]")
    if overrides:
        for override in overrides:
            if "=" not in override:
                continue
            key_path, value_str = override.split("=", 1)
            
            # Parse value
            try:
                value = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                value = value_str  # Keep as string if not parseable
            
            # Set nested key
            keys = key_path.split(".")
            current = merged
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
    
    data = _merge_dicts(dataclasses.asdict(RootConfig()), merged)
    return _dict_to_config(data)


def _merge_dicts(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            result[key] = _merge_dicts(base[key], value)
        else:
            result[key] = value
    return result


def _coerce_types(data: dict, target_class) -> dict:
    """Coerce dict values to match dataclass field types."""
    result = {}
    field_types = typing.get_type_hints(target_class)
    
    for key, value in data.items():
        if key not in field_types:
            result[key] = value
            continue
        
        target_type = field_types[key]
        # Handle Optional types
        if hasattr(target_type, '__origin__') and target_type.__origin__ is typing.Union:
            target_type = next((t for t in target_type.__args__ if t is not type(None)), str)
        
        # Convert string to appropriate numeric type
        if target_type in (float, int) and isinstance(value, str):
            try:
                result[key] = target_type(value)
            except (ValueError, TypeError):
                result[key] = value
        else:
            result[key] = value
    
    return result


def _dict_to_config(data: dict) -> RootConfig:
    model = ModelConfig(**_coerce_types(data.get("model", {}), ModelConfig))
    
    # Handle nested extract config
    extract_data = data.get("extract", {})
    pooling = PoolingConfig(**_coerce_types(extract_data.get("pooling", {}), PoolingConfig))
    save = SaveConfig(**_coerce_types(extract_data.get("save", {}), SaveConfig))
    extract_clean = {k: v for k, v in extract_data.items() if k not in ["pooling", "save"]}
    extract = ExtractConfig(**{**_coerce_types(extract_clean, ExtractConfig), "pooling": pooling, "save": save})
    
    data_cfg = DataConfig(**_coerce_types(data.get("data", {}), DataConfig))
    repro = ReproConfig(**_coerce_types(data.get("repro", {}), ReproConfig))
    
    # Handle nested SAE config
    sae_data = data.get("sae", {})
    if "sae" in sae_data:
        sae_cfg = SAEConfig(**_coerce_types(sae_data["sae"], SAEConfig))
    else:
        sae_cfg = SAEConfig(**_coerce_types(sae_data, SAEConfig)) if sae_data else SAEConfig()
    normalization = NormalizationConfig(**_coerce_types(sae_data.get("normalization", {}), NormalizationConfig))
    io = IOConfig(**_coerce_types(sae_data.get("io", {}), IOConfig))
    sae = SAETrainConfig(
        sae=sae_cfg,
        normalization=normalization,
        layers=sae_data.get("layers", [0, 2, 4]),
        io=io
    )
    
    # Handle nested eval config
    eval_data = data.get("eval", {})
    consensus = ConsensusConfig(**_coerce_types(eval_data.get("consensus", {}), ConsensusConfig))
    threshold = ThresholdConfig(**_coerce_types(eval_data.get("threshold", {}), ThresholdConfig))
    intervention_data = data.get("intervention", {})
    intervention = InterventionConfig(**_coerce_types(intervention_data, InterventionConfig))
    
    # Remove nested objects from eval_data to avoid conflicts
    eval_clean = {k: v for k, v in eval_data.items() if k not in ["consensus", "threshold", "intervention", "io"]}
    eval_cfg = EvalConfig(**{
        **_coerce_types(eval_clean, EvalConfig),
        "consensus": consensus,
        "threshold": threshold,
        "intervention": intervention,
        "io": io
    })
    
    return RootConfig(
        model=model,
        extract=extract,
        data=data_cfg,
        repro=repro,
        sae=sae,
        eval=eval_cfg
    )


__all__ = [
    "ModelConfig",
    "PoolingConfig", 
    "SaveConfig",
    "ExtractConfig",
    "DataConfig",
    "ReproConfig",
    "NormalizationConfig",
    "IOConfig",
    "SAEConfig",
    "SAETrainConfig",
    "ConsensusConfig",
    "ThresholdConfig",
    "InterventionConfig",
    "EvalConfig",
    "RootConfig",
    "load_config",
]
