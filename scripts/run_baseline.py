#!/usr/bin/env python3
"""End-to-end baseline.

    python scripts/run_baseline.py --split test_id --n 50

Writes results/baseline_<split>/metrics.json, success.png, and failures.png.
Default planners are naive, repair, and oracle so the command runs on a Mac
with no API key and no GPU. Add --planners llm,llm_repair when OPENAI_API_KEY
or a local model is configured.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.runner import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TAMP baseline")
    parser.add_argument("--split", default="test_id", choices=["train", "test_id", "test_ood"])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--planners", default="naive,repair,oracle")
    parser.add_argument("--optimizer", default=str(ROOT / "configs" / "optimizer_default.json"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--llm-backend", default="auto")
    args = parser.parse_args()
    out = args.out or str(ROOT / "results" / f"baseline_{args.split}")
    spec = {
        "name": f"baseline_{args.split}",
        "split": args.split,
        "n": args.n,
        "planners": [p.strip() for p in args.planners.split(",") if p.strip()],
        "optimizer_config": args.optimizer,
        "output_dir": out,
        "K": args.K,
        "model": args.model,
        "llm_backend": args.llm_backend,
    }
    run_experiment(spec)


if __name__ == "__main__":
    main()
