"""CLI for SAE training."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..utils.config import load_config
from ..sae.io import load_shards, save_train_result
from ..sae.train import train_sae
from ..utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Top-K SAE")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("overrides", nargs="*", help="Config overrides (e.g., layers=[0,2,4])")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    cfg = load_config([args.config], overrides=args.overrides)
    
    from ..utils.logging import get_logger
    LOGGER = get_logger(__name__)
    
    # Train SAE for each layer
    for layer in cfg.sae.layers:
        LOGGER.info(f"Training SAE for layer {layer}")
        
        # Find shard files for this layer
        shard_dir = Path(cfg.sae.io.in_dir) / f"layer_{layer:02d}"
        if not shard_dir.exists():
            LOGGER.error(f"Shard directory not found: {shard_dir}")
            continue
            
        shard_files = sorted(shard_dir.glob("*.npz"))
        if not shard_files:
            LOGGER.error(f"No shard files found in {shard_dir}")
            continue
        
        LOGGER.info(f"Found {len(shard_files)} shard files for layer {layer}")
        
        # Load shards and train
        shard_arrays = list(load_shards(shard_files))
        result = train_sae(layer, shard_arrays, cfg.sae.sae)
        
        # Save results
        output_dir = Path(cfg.sae.io.out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"sae_layer{layer:02d}"
        save_train_result(result, output_path)
        LOGGER.info(f"Saved SAE for layer {layer} to {output_path}")


if __name__ == "__main__":
    main()
