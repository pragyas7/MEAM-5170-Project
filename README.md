# Trajectory-guided learning for task and motion planning

MEAM 5170 final project. A language-model task planner proposes symbolic plans for a wheeled robot that pushes boxes through a floor plan. A trajectory optimizer accepts or rejects each step, and that accept/reject signal is what we fine-tune on.

Team: Pragya Singh, Vasumathi Venkat.

## Questions

1. Does a planner fine-tuned on small floor plans still succeed on larger ones, or does it overfit the training distribution?
2. Does fine-tuning teach the task (which doorway to clear, which room to push into) or the optimizer's limits (speed, turning radius, push force)?

## What is implemented

The laptop side is complete and runs without a GPU or an API key:

- Procedural layouts (`env/generator.py`). Train and test-ID are chains and L-shapes with 2–3 rooms. Test-OOD is a 2×2 grid or a star with 4–5 rooms. `gen_layout(seed, ...)` is deterministic.
- JSON and a text scene description for the planner.
- A matplotlib renderer and GIF export.
- A unicycle navigator and a quasi-static push checker (`opt/`).
- A plan parser that returns a structured error on malformed JSON.
- Three classical planners: `naive` (one-shot, ignores blockers), `repair` (feeds the optimizer's failure back, up to 3 retries), `oracle` (symbolic search on the room graph).
- An OpenAI-compatible client and a Hugging Face client for the zero-shot and fine-tuned conditions.
- Dataset generation (SFT and DPO JSONL) and LoRA training scripts for a 24 GB GPU.
- Experiment configs for E1 (test-ID), E2 (test-OOD), and E3 (optimizer capability shift).

## Design choices

**CasADi + IPOPT, not Drake.** Drake does not ship a wheel for the system Python on this machine. CasADi is the fallback the spec allows, and its pip wheel includes IPOPT for Apple Silicon. The collocation problem is a direct transcription of unicycle dynamics. If IPOPT fails, we keep an analytically timed feasible trajectory (stop-and-spin, or circular fillets when a minimum turning radius is set). A plan is still accepted only after an exact collision post-check.

**Rigid face attachment.** During `push`, the robot is fixed to one face of the box. The box translates only along that face's inward normal and yaws with the robot. Sliding requires `max_push_force >= mu * mass * g`. Yaw authority is the fraction of the force budget left after friction is paid, so a robot that can barely overcome friction cannot also turn the box. This is the Q2 knob, together with `v_max`, `w_max`, `min_turn_radius`, and the horizon.

**Classical baselines stand in for the LLM until a checkpoint exists.** `naive` is the untrained guess (walk up to the box and push it at the goal). `repair` is the optimizer-in-the-loop baseline. `oracle` is an upper bound that never sees the language model. `llm` and `llm_repair` use the same checker the moment `OPENAI_API_KEY` or a local model is set. The prompt includes the active optimizer limits, so a model that reads the prompt can adapt when those limits change and a model that memorized training motions cannot. That is the E3 contrast.

## Setup

```bash
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest
```

GPU training extras (not needed on the Mac):

```bash
pip install -e '.[train]'
```

## Commands

```bash
# Status-report baseline. Writes metrics JSON and plots.
.venv/bin/python scripts/run_baseline.py --split test_id --n 50

# Same runner from a config file. E2 is the OOD split. E3 swaps optimizer limits.
.venv/bin/python scripts/run_experiment.py configs/e1_test_id.json
.venv/bin/python scripts/run_experiment.py configs/e2_test_ood.json --compare results/e1/metrics.json
.venv/bin/python scripts/run_experiment.py configs/e3_shift.json

# Label training layouts. Writes data/out/train.jsonl, .sft.jsonl, and .dpo.jsonl.
.venv/bin/python scripts/generate_data.py --split train --n 20 --candidates 4

# Poster frame and GIF for one layout.
.venv/bin/python scripts/render_demo.py --fixture push

# Fine-tune. Dry-run works here; the real fit is CUDA, under 2 hours at the default config.
.venv/bin/python train/sft_lora.py --config configs/sft_lora.json --data data/examples/sft_tiny.jsonl --dry-run
.venv/bin/python train/dpo_lora.py --config configs/dpo_lora.json --data data/examples/dpo_tiny.jsonl --dry-run
python train/sft_lora.py --config configs/sft_lora.json
```

Zero-shot LLM, once a key is exported:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini   # optional
.venv/bin/python scripts/run_baseline.py --split test_id --n 20 --planners llm,llm_repair
```

## Repo map

| Path | Role |
| --- | --- |
| `env/` | Layouts, geometry, text, renderer |
| `opt/` | Navigate, push, plan checker |
| `planner/` | Parser, prompt, symbolic planner, repair, LLM clients |
| `data/` | Optimizer-labeled JSONL |
| `train/` | LoRA SFT and DPO |
| `eval/` | Metrics and plots |
| `configs/` | Optimizer limits and experiment specs |
| `scripts/` | Command line |
| `tests/` | Geometry, generator, parser, dynamics, optimizer |

## Not in scope

3D, humanoids, hardware, perception, and multi-robot. The full state is known.
