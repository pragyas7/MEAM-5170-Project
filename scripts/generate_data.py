#!/usr/bin/env python3
"""Label train layouts with the optimizer and write JSONL for SFT/DPO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.generate import generate  # noqa: E402
from opt.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate optimizer-labeled plans")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--out", default=str(ROOT / "data" / "out" / "train.jsonl"))
    parser.add_argument("--optimizer", default=str(ROOT / "configs" / "optimizer_default.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.optimizer)
    summary = generate(
        args.split,
        args.n,
        args.out,
        candidates=args.candidates,
        config=config,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
