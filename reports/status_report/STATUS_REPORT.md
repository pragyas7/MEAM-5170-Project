# MEAM 5170 Project Status Report

**Progress so far.** We built a baseline for trajectory-guided task and motion planning that runs on a laptop. It includes a procedural room-layout generator, a unicycle-robot navigator, and a rigid-face push optimizer (CasADi + IPOPT, with an analytic fallback). It also has a JSON plan parser and feasibility checker, plus three classical planners (naive, repair, and a symbolic oracle) that stand in for the LLM. The full unit test suite passes (27 of 27). On 50 in-distribution test layouts, the naive planner succeeds on 62% of tasks, and all of its failures are collisions. With optimizer-feedback repair, success reaches 100%, at 1.56 attempts on average, which matches the oracle. Scripts for training-data generation and LoRA SFT/DPO are written but have not been run yet.

**By the end of the semester,** we hope to fine-tune an LLM on trajectory-optimizer feedback. We will compare it against in-context feedback and an un-tuned LLM on three experiments: in-distribution layouts, out-of-distribution layouts, and shifted optimizer limits. We will report plan success and feasibility rates.

## Baseline results (`scripts/run_baseline.py --split test_id --n 50`)

| Planner | Success | Success rate | Mean attempts | Mean optimizer calls | Mean plan length | Mean wall time (s) |
|---|---|---|---|---|---|---|
| Naive | 31/50 | 0.62 | 1.00 | 3.80 | 3.52 | 0.190 |
| Repair | 50/50 | 1.00 | 1.56 | 6.06 | 3.82 | 0.280 |
| Oracle | 50/50 | 1.00 | 1.00 | 4.32 | 2.28 | 0.175 |

All 19 naive failures were collisions. Raw metrics are in `baseline_test_id_metrics.json`. Plots are in `baseline_test_id_success_rate.png` and `baseline_test_id_failure_breakdown.png`.
