# Roadmap Version Runners

This directory mirrors the experiment roadmap. Each version has its own folder and `run.py` so the report progression can be reproduced step by step.

The neural-network implementation remains in `src/`:

- `src/classification_nn.py`: MLP, training loop, checkpoint evaluation.
- `src/classification_experiments.py`: ablation configs and focused experiment suites.
- `src/data_split.py`: fixed IRIS/MNIST splits and test split loading.

Version folders:

| Version | Folder | Purpose |
|---|---|---|
| V0 | `v0_untrained_random/` | Untrained random-network baseline. |
| V1 | `v1_linear_softmax/` | Linear softmax classifier. |
| V2 | `v2_small_mlp/` | Small one-hidden-layer MLP. |
| V3 | `v3_stable_mlp/` | Stable MLP components such as scaling, initialization, and mini-batch training. |
| V4 | `v4_component_checks/` | Loss, L2, learning-rate, width, and training-strategy checks. |
| Final | `final_selected_model/` | Final selected MNIST MLP and official test evaluation. |

Run examples:

```bash
uv run python versions/v0_untrained_random/run.py --dataset all --profile quick
uv run python versions/v1_linear_softmax/run.py --dataset all --profile quick
uv run python versions/v2_small_mlp/run.py --dataset all --profile quick
uv run python versions/v3_stable_mlp/run.py --dataset all --profile quick
uv run python versions/v4_component_checks/run.py --dataset all --profile quick
uv run python versions/final_selected_model/run.py --mnist-limit 1000 --epochs 5 --skip-test
```

Default outputs are written as JSON files under `results/versions/<version>/`.
