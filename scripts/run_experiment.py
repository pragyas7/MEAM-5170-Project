#!/usr/bin/env python3
"""Run an experiment from a JSON config. See configs/e1_test_id.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.plots import plot_id_ood  # noqa: E402
from eval.runner import load_spec, run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured TAMP experiment")
    parser.add_argument("config", help="JSON experiment spec")
    parser.add_argument("--compare", default=None, help="second metrics.json for an ID vs OOD plot")
    args = parser.parse_args()
    spec = load_spec(args.config)
    summary = run_experiment(spec)
    if args.compare:
        import json

        other = json.loads(Path(args.compare).read_text())
        out = Path(spec.get("output_dir", "results/exp")) / "id_ood.png"
        # The file passed to --compare is the other summary. If this run is OOD,
        # plot it on the right.
        if "ood" in spec.get("split", ""):
            plot_id_ood(other, summary, out)
        else:
            plot_id_ood(summary, other, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
