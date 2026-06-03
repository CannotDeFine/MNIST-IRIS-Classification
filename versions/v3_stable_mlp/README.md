# V3: Stable MLP

This version checks the stabilizing pieces used by the main MLP path: input scaling, He initialization, mini-batch updates, and the current stable MLP.

Run:

```bash
uv run python versions/v3_stable_mlp/run.py --dataset all --profile quick
```

Outputs:

```text
results/versions/v3_stable_mlp/quick.json
```
