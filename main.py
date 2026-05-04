from __future__ import annotations

import argparse

from src.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Forklift entry/exit direction tracking")
    parser.add_argument("--config", default="config/cameras.yaml", help="Path to camera configuration file")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
