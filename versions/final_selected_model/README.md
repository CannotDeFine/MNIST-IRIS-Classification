# Final: Selected MNIST MLP

This version trains the final selected MLP:

```text
784 -> ReLU(256) -> softmax(10)
optimizer = momentum
learning_rate = 0.05
momentum = 0.9
l2 = 0
max epochs = 100
early stopping patience = 3
```

Run:

```bash
uv run python versions/final_selected_model/run.py
```

Outputs:

```text
results/versions/final_selected_model/train_val_metrics.json
results/versions/final_selected_model/test_metrics.json
results/versions/final_selected_model/final_mnist_mlp.npz
```

For a fast smoke run:

```bash
uv run python versions/final_selected_model/run.py --mnist-limit 1000 --epochs 5 --skip-test
```

