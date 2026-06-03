# V4: Component Checks

This version runs focused checks for components that are not a single linear evolution step: loss function, L2 strength, learning rate, hidden width, and training strategy.

Run:

```bash
uv run python versions/v4_component_checks/run.py --dataset all --profile quick
```

Outputs:

```text
results/versions/v4_component_checks/loss-comparison_quick.json
results/versions/v4_component_checks/l2-sweep_quick.json
results/versions/v4_component_checks/lr-sweep_quick.json
results/versions/v4_component_checks/width-sweep_quick.json
results/versions/v4_component_checks/training-strategies_quick.json
```
