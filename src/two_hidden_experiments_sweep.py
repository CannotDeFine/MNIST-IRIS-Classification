"""Two-hidden-layer MLP experiments for IRIS/MNIST.

This script implements a hand-written two-hidden-layer neural network using
NumPy only. It supports both a single run and a structure sweep.

Architecture:
    X -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Softmax

Single run example:
python src/two_hidden_experiments.py --dataset mnist --hidden-dim1 256 --hidden-dim2 128 --epochs 5 --learning-rate 0.05 --batch-size 128 --optimizer momentum --momentum 0.9 --output results/mnist_two_hidden_metrics.json --markdown-output results/mnist_two_hidden_metrics.md

Structure sweep example:
python src/two_hidden_experiments.py --dataset mnist --sweep-structures --structures 128,128 128,256 256,256 256,128 --epochs 5 --learning-rate 0.05 --batch-size 128 --optimizer momentum --momentum 0.9 --output results/mnist_two_hidden_sweep.json --markdown-output results/mnist_two_hidden_sweep.md

Quick structure sweep:
python src/two_hidden_experiments.py --dataset mnist --sweep-structures --structures 128,128 128,256 256,256 256,128 --train-limit 1000 --val-limit 1000 --epochs 3 --output results/mnist_two_hidden_sweep_quick.json --markdown-output results/mnist_two_hidden_sweep_quick.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .classification_nn import SplitMeta, _encode_labels, save_metrics
    from .data_split import load_iris_split, load_mnist_split, load_mnist_test
except ImportError:  # direct execution: python src/two_hidden_experiments.py
    from classification_nn import SplitMeta, _encode_labels, save_metrics
    from data_split import load_iris_split, load_mnist_split, load_mnist_test


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class TwoHiddenConfig:
    dataset: str
    epochs: int
    hidden_dim1: int
    hidden_dim2: int
    learning_rate: float
    batch_size: int
    seed: int = 42
    l2: float = 0.0
    optimizer: str = "sgd"
    momentum: float = 0.9
    lr_decay: float = 0.0
    train_limit: int | None = None
    val_limit: int | None = None
    scale_inputs: bool = True
    name: str | None = None


class TwoHiddenMLPClassifier:
    """Two-hidden-layer MLP trained by hand-written backpropagation.

    Architecture:
        X -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Softmax
    Loss:
        Softmax Cross-Entropy + optional L2 regularization
    Optimizer:
        SGD or Momentum SGD
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        output_dim: int,
        learning_rate: float = 0.05,
        l2: float = 0.0,
        optimizer: str = "sgd",
        momentum: float = 0.9,
        lr_decay: float = 0.0,
        seed: int = 42,
    ) -> None:
        if optimizer not in {"sgd", "momentum"}:
            raise ValueError("optimizer must be 'sgd' or 'momentum'")

        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.initial_learning_rate = learning_rate
        self.l2 = l2
        self.optimizer = optimizer
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.rng = np.random.default_rng(seed)

        # He initialization for ReLU layers.
        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim1))
        self.b1 = np.zeros(hidden_dim1, dtype=np.float64)

        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / hidden_dim1), (hidden_dim1, hidden_dim2))
        self.b2 = np.zeros(hidden_dim2, dtype=np.float64)

        self.W3 = self.rng.normal(0.0, np.sqrt(2.0 / hidden_dim2), (hidden_dim2, output_dim))
        self.b3 = np.zeros(output_dim, dtype=np.float64)

        # Momentum buffers.
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(z, 0.0)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def _forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        logits = a2 @ self.W3 + self.b3
        probs = self._softmax(logits)
        return z1, a1, z2, a2, probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self._forward(X)[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        probs = self.predict_proba(X)
        n = y.shape[0]
        ce = -np.log(probs[np.arange(n), y] + 1e-12).mean()
        if self.l2 > 0.0:
            ce += 0.5 * self.l2 * (
                np.sum(self.W1 * self.W1)
                + np.sum(self.W2 * self.W2)
                + np.sum(self.W3 * self.W3)
            )
        return float(ce)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.int64)
        return float(np.mean(self.predict(X) == y))

    def _current_lr(self, epoch: int) -> float:
        if self.lr_decay <= 0.0:
            return self.initial_learning_rate
        return self.initial_learning_rate / (1.0 + self.lr_decay * max(0, epoch - 1))

    def _apply_update(self, grads: dict[str, np.ndarray], lr: float) -> None:
        if self.optimizer == "momentum":
            self.vW1 = self.momentum * self.vW1 - lr * grads["dW1"]
            self.vb1 = self.momentum * self.vb1 - lr * grads["db1"]
            self.vW2 = self.momentum * self.vW2 - lr * grads["dW2"]
            self.vb2 = self.momentum * self.vb2 - lr * grads["db2"]
            self.vW3 = self.momentum * self.vW3 - lr * grads["dW3"]
            self.vb3 = self.momentum * self.vb3 - lr * grads["db3"]

            self.W1 += self.vW1
            self.b1 += self.vb1
            self.W2 += self.vW2
            self.b2 += self.vb2
            self.W3 += self.vW3
            self.b3 += self.vb3
        else:
            self.W1 -= lr * grads["dW1"]
            self.b1 -= lr * grads["db1"]
            self.W2 -= lr * grads["dW2"]
            self.b2 -= lr * grads["db2"]
            self.W3 -= lr * grads["dW3"]
            self.b3 -= lr * grads["db3"]

    def _train_batch(self, X: np.ndarray, y: np.ndarray, lr: float) -> None:
        n = X.shape[0]
        z1, a1, z2, a2, probs = self._forward(X)

        # Softmax + Cross-Entropy gradient.
        dz3 = probs.copy()
        dz3[np.arange(n), y] -= 1.0
        dz3 /= n

        dW3 = a2.T @ dz3
        db3 = dz3.sum(axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0.0)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0.0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        if self.l2 > 0.0:
            dW1 += self.l2 * self.W1
            dW2 += self.l2 * self.W2
            dW3 += self.l2 * self.W3

        self._apply_update(
            {
                "dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2,
                "dW3": dW3,
                "db3": db3,
            },
            lr=lr,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        val_data: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        history: dict[str, list[float]] = {
            "loss": [],
            "accuracy": [],
            "learning_rate": [],
        }
        if val_data is not None:
            history["val_loss"] = []
            history["val_accuracy"] = []

        def record_metrics(lr: float) -> None:
            history["loss"].append(self.loss(X, y))
            history["accuracy"].append(self.accuracy(X, y))
            history["learning_rate"].append(lr)
            if val_data is not None:
                X_val, y_val = val_data
                history["val_loss"].append(self.loss(X_val, y_val))
                history["val_accuracy"].append(self.accuracy(X_val, y_val))

        record_metrics(self._current_lr(epoch=1))

        for epoch in range(1, epochs + 1):
            lr = self._current_lr(epoch)
            order = self.rng.permutation(X.shape[0])
            for start in range(0, X.shape[0], batch_size):
                idx = order[start : start + batch_size]
                self._train_batch(X[idx], y[idx], lr=lr)

            record_metrics(lr)
            if verbose:
                msg = (
                    f"epoch {epoch:03d} "
                    f"loss={history['loss'][-1]:.4f} "
                    f"acc={history['accuracy'][-1]:.4f}"
                )
                if val_data is not None:
                    msg += (
                        f" val_loss={history['val_loss'][-1]:.4f} "
                        f"val_acc={history['val_accuracy'][-1]:.4f}"
                    )
                print(msg)

        return history


def _label_meta(labels: np.ndarray) -> SplitMeta:
    label_to_id = {str(label): i for i, label in enumerate(sorted(set(labels)))}
    return SplitMeta(mean=None, std=None, label_to_id=label_to_id)


def load_split(
    dataset: str,
    which: str,
    config: TwoHiddenConfig,
    meta: SplitMeta | None = None,
) -> tuple[np.ndarray, np.ndarray, SplitMeta]:
    limit = config.train_limit if which == "train" else config.val_limit

    if dataset == "iris":
        X, labels = load_iris_split(which)
        if limit is not None:
            X = X[:limit]
            labels = labels[:limit]
        X = X.astype(np.float64)

        if meta is None:
            meta = _label_meta(labels)
            if config.scale_inputs:
                mean = X.mean(axis=0)
                std = X.std(axis=0)
                std[std == 0.0] = 1.0
                meta = SplitMeta(mean=mean, std=std, label_to_id=meta.label_to_id)

        if config.scale_inputs:
            assert meta.mean is not None and meta.std is not None
            X = (X - meta.mean) / meta.std

        return X, _encode_labels(labels, meta.label_to_id), meta

    if dataset == "mnist":
        images, labels = load_mnist_split(which)
        if limit is not None:
            images = images[:limit]
            labels = labels[:limit]

        X = images.reshape(images.shape[0], -1).astype(np.float64)
        if config.scale_inputs:
            X /= 255.0

        if meta is None:
            meta = SplitMeta(mean=None, std=None, label_to_id={str(i): i for i in range(10)})

        return X, _encode_labels(labels, meta.label_to_id), meta

    raise ValueError("dataset must be 'iris' or 'mnist'")


def load_test_split(config: TwoHiddenConfig, meta: SplitMeta) -> tuple[np.ndarray, np.ndarray] | None:
    if config.dataset != "mnist":
        return None

    images, labels = load_mnist_test()
    X = images.reshape(images.shape[0], -1).astype(np.float64)
    if config.scale_inputs:
        X /= 255.0
    y = _encode_labels(labels, meta.label_to_id)
    return X, y


def run_two_hidden_experiment(config: TwoHiddenConfig, verbose: bool = False) -> dict[str, Any]:
    X_train, y_train, meta = load_split(config.dataset, "train", config)
    X_val, y_val, _ = load_split(config.dataset, "val", config, meta=meta)

    output_dim = int(np.max(y_train)) + 1
    batch_size = X_train.shape[0] if config.batch_size <= 0 else config.batch_size

    model = TwoHiddenMLPClassifier(
        input_dim=X_train.shape[1],
        hidden_dim1=config.hidden_dim1,
        hidden_dim2=config.hidden_dim2,
        output_dim=output_dim,
        learning_rate=config.learning_rate,
        l2=config.l2,
        optimizer=config.optimizer,
        momentum=config.momentum,
        lr_decay=config.lr_decay,
        seed=config.seed,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=config.epochs,
        batch_size=batch_size,
        val_data=(X_val, y_val),
        verbose=verbose,
    )

    result_name = config.name or f"{config.dataset}_two_hidden_{config.hidden_dim1}_{config.hidden_dim2}"
    result: dict[str, Any] = {
        "name": result_name,
        "dataset": config.dataset,
        "description": (
            f"Two-hidden-layer MLP: input -> {config.hidden_dim1} ReLU -> "
            f"{config.hidden_dim2} ReLU -> softmax."
        ),
        "ablates": "two hidden layer structure",
        "config": asdict(config) | {"effective_batch_size": batch_size},
        "epochs_trained": len(history["loss"]) - 1,
        "final_train_accuracy": history["accuracy"][-1],
        "final_val_accuracy": history["val_accuracy"][-1],
        "final_train_loss": history["loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "history": history,
    }

    test_data = load_test_split(config, meta)
    if test_data is not None:
        X_test, y_test = test_data
        result.update(
            {
                "test_count": int(y_test.shape[0]),
                "final_test_accuracy": model.accuracy(X_test, y_test),
                "final_test_loss": model.loss(X_test, y_test),
            }
        )

    return result


def parse_structure(text: str) -> tuple[int, int]:
    """Parse a hidden-layer structure like '128,256'."""
    pieces = text.replace("x", ",").replace("-", ",").split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError(
            f"Invalid structure '{text}'. Use format like 128,256."
        )
    try:
        h1, h2 = int(pieces[0]), int(pieces[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid structure '{text}'. Hidden dimensions must be integers."
        ) from exc
    if h1 <= 0 or h2 <= 0:
        raise argparse.ArgumentTypeError("Hidden dimensions must be positive.")
    return h1, h2


def format_markdown_table(results: list[dict[str, Any]]) -> str:
    has_test = any("final_test_accuracy" in result for result in results)
    header = (
        "| Model | Hidden 1 | Hidden 2 | Epochs | Optimizer | LR | L2 | "
        "Train Acc | Val Acc | Val Loss |"
    )
    if has_test:
        header += " Test Acc | Test Loss |"
    separator = "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"
    if has_test:
        separator += "---:|---:|"

    rows = []
    for result in results:
        config = result["config"]
        row = (
            f"| {result['name']} | {config['hidden_dim1']} | {config['hidden_dim2']} | "
            f"{result['epochs_trained']} | {config['optimizer']} | {config['learning_rate']:g} | "
            f"{config['l2']:g} | {result['final_train_accuracy']:.4f} | "
            f"{result['final_val_accuracy']:.4f} | {result['final_val_loss']:.4f} |"
        )
        if has_test:
            if "final_test_accuracy" in result:
                row += f" {result['final_test_accuracy']:.4f} | {result['final_test_loss']:.4f} |"
            else:
                row += " — | — |"
        rows.append(row)

    return header + "\n" + separator + "\n" + "\n".join(rows)


def format_markdown_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Two-Hidden-Layer MLP Experiments",
        "",
    ]
    if any(result.get("dataset") == "mnist" and "final_test_accuracy" in result for result in results):
        lines.extend([
            "## MNIST Test Set Evaluation",
            "",
            "Each MNIST two-hidden-layer model is evaluated on the official t10k test set after training.",
            "",
        ])
    lines.append(format_markdown_table(results))
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train hand-written two-hidden-layer MLP models.")
    parser.add_argument("--dataset", choices=["iris", "mnist"], default="mnist")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim1", type=int, default=256)
    parser.add_argument("--hidden-dim2", type=int, default=128)
    parser.add_argument(
        "--sweep-structures",
        action="store_true",
        help="Run multiple two-hidden-layer structures in one command.",
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        type=parse_structure,
        default=None,
        help="Structures for sweep mode, e.g. --structures 128,128 128,256 256,256 256,128",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--optimizer", choices=["sgd", "momentum"], default="momentum")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--lr-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--no-scale-inputs", action="store_true")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "two_hidden_metrics.json")
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def make_config(args: argparse.Namespace, hidden_dim1: int, hidden_dim2: int) -> TwoHiddenConfig:
    return TwoHiddenConfig(
        dataset=args.dataset,
        epochs=args.epochs,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        momentum=args.momentum,
        l2=args.l2,
        lr_decay=args.lr_decay,
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        scale_inputs=not args.no_scale_inputs,
        name=f"{args.dataset}_two_hidden_{hidden_dim1}_{hidden_dim2}",
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.sweep_structures:
        structures = args.structures or [(128, 128), (128, 256), (256, 256), (256, 128)]
    else:
        structures = [(args.hidden_dim1, args.hidden_dim2)]

    results = []
    for h1, h2 in structures:
        config = make_config(args, hidden_dim1=h1, hidden_dim2=h2)
        print(f"running {config.name}: {h1}->{h2}")
        results.append(run_two_hidden_experiment(config, verbose=args.verbose))

    output_payload: dict[str, Any] | list[dict[str, Any]]
    output_payload = results[0] if len(results) == 1 else results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(output_payload, args.output)
    print(f"results saved to {args.output}")

    report = format_markdown_report(results)
    print(report)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(report, encoding="utf-8")
        print(f"markdown report saved to {args.markdown_output}")


if __name__ == "__main__":
    main()
