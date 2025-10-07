"""CLI entrypoint for δ_int extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..utils.config import load_config
from ..utils.logging import configure_logging
from ..extract.runner import ExtractRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract δ_int activations")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    parser.add_argument("overrides", nargs="*", help="Config overrides (e.g., extract.layers=[0,2,4])")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    cfg = load_config([args.config], overrides=args.overrides)
    runner = ExtractRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
