"""Small NumPy CNN for MNIST as an optional extension.

The main project model remains the MLP in classification_nn.py. This module is
kept separate because convolution and pooling logic are specific to image data.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .data_split import load_mnist_split
except ImportError:  # pragma: no cover - direct script execution.
    from data_split import load_mnist_split


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


@dataclass
class CNNConfig:
    """Configuration for the compact MNIST CNN."""

    filters: int = 8
    kernel_size: int = 3
    learning_rate: float = 0.05
    epochs: int = 3
    batch_size: int = 32
    seed: int = 42
    train_limit: int | None = 1000
    val_limit: int | None = 1000


class SimpleMNISTCNN:
    """Conv(3x3) -> ReLU -> 2x2 max-pool -> dense softmax."""

    def __init__(
        self,
        filters: int = 8,
        kernel_size: int = 3,
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.filters = filters
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.output_dim = 10
        self.rng = np.random.default_rng(seed)

        scale = np.sqrt(2.0 / (kernel_size * kernel_size))
        self.W_conv = self.rng.normal(0.0, scale, (filters, 1, kernel_size, kernel_size))
        self.b_conv = np.zeros(filters, dtype=np.float64)

        pooled_side = (28 - kernel_size + 1) // 2
        dense_input_dim = filters * pooled_side * pooled_side
        self.W_fc = self.rng.normal(0.0, np.sqrt(2.0 / dense_input_dim), (dense_input_dim, 10))
        self.b_fc = np.zeros(10, dtype=np.float64)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def _conv_forward(self, X: np.ndarray) -> np.ndarray:
        n, _, height, width = X.shape
        out_h = height - self.kernel_size + 1
        out_w = width - self.kernel_size + 1
        out = np.zeros((n, self.filters, out_h, out_w), dtype=np.float64)
        for i in range(out_h):
            for j in range(out_w):
                patch = X[:, :, i : i + self.kernel_size, j : j + self.kernel_size]
                out[:, :, i, j] = np.tensordot(patch, self.W_conv, axes=([1, 2, 3], [1, 2, 3]))
        out += self.b_conv[None, :, None, None]
        return out

    @staticmethod
    def _maxpool_forward(X: np.ndarray) -> np.ndarray:
        n, channels, height, width = X.shape
        pooled = np.zeros((n, channels, height // 2, width // 2), dtype=np.float64)
        for i in range(0, height, 2):
            for j in range(0, width, 2):
                pooled[:, :, i // 2, j // 2] = X[:, :, i : i + 2, j : j + 2].max(axis=(2, 3))
        return pooled

    @staticmethod
    def _maxpool_backward(dpooled: np.ndarray, relu: np.ndarray) -> np.ndarray:
        drelu = np.zeros_like(relu)
        _, _, height, width = relu.shape
        for i in range(0, height, 2):
            for j in range(0, width, 2):
                window = relu[:, :, i : i + 2, j : j + 2]
                max_values = window.max(axis=(2, 3), keepdims=True)
                mask = window == max_values
                mask_count = mask.sum(axis=(2, 3), keepdims=True)
                drelu[:, :, i : i + 2, j : j + 2] += (
                    mask * dpooled[:, :, i // 2, j // 2, None, None] / mask_count
                )
        return drelu

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        conv = self._conv_forward(X)
        relu = np.maximum(conv, 0.0)
        pooled = self._maxpool_forward(relu)
        flat = pooled.reshape(X.shape[0], -1)
        logits = flat @ self.W_fc + self.b_fc
        return conv, relu, pooled, flat, logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, _, _, _, logits = self._forward(np.asarray(X, dtype=np.float64))
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        probs = self.predict_proba(X)
        n = y.shape[0]
        return float(-np.log(probs[np.arange(n), y] + 1e-12).mean())

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def _train_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        n = X.shape[0]
        conv, relu, pooled, flat, logits = self._forward(X)
        probs = self._softmax(logits)
        probs[np.arange(n), y] -= 1.0
        probs /= n

        dW_fc = flat.T @ probs
        db_fc = probs.sum(axis=0)
        dflat = probs @ self.W_fc.T
        dpooled = dflat.reshape(pooled.shape)
        drelu = self._maxpool_backward(dpooled, relu)
        dconv = drelu * (conv > 0.0)

        dW_conv = np.zeros_like(self.W_conv)
        db_conv = dconv.sum(axis=(0, 2, 3))
        dX = np.zeros_like(X)
        out_h, out_w = dconv.shape[2], dconv.shape[3]
        for i in range(out_h):
            for j in range(out_w):
                patch = X[:, :, i : i + self.kernel_size, j : j + self.kernel_size]
                for f in range(self.filters):
                    grad = dconv[:, f, i, j]
                    dW_conv[f] += np.sum(patch * grad[:, None, None, None], axis=0)
                    dX[:, :, i : i + self.kernel_size, j : j + self.kernel_size] += (
                        grad[:, None, None, None] * self.W_conv[f]
                    )

        self.W_fc -= self.learning_rate * dW_fc
        self.b_fc -= self.learning_rate * db_fc
        self.W_conv -= self.learning_rate * dW_conv
        self.b_conv -= self.learning_rate * db_conv

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
        history: dict[str, list[float]] = {"loss": [], "accuracy": []}
        if val_data is not None:
            history["val_loss"] = []
            history["val_accuracy"] = []

        def record_metrics() -> None:
            history["loss"].append(self.loss(X, y))
            history["accuracy"].append(self.accuracy(X, y))
            if val_data is not None:
                X_val, y_val = val_data
                history["val_loss"].append(self.loss(X_val, y_val))
                history["val_accuracy"].append(self.accuracy(X_val, y_val))

        record_metrics()
        if verbose:
            print(self._format_metrics(history, epoch=0, has_val=val_data is not None))

        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(X.shape[0])
            for start in range(0, X.shape[0], batch_size):
                idx = order[start : start + batch_size]
                self._train_batch(X[idx], y[idx])
            record_metrics()
            if verbose:
                print(self._format_metrics(history, epoch=epoch, has_val=val_data is not None))
        return history

    @staticmethod
    def _format_metrics(history: dict[str, list[float]], epoch: int, has_val: bool) -> str:
        message = f"epoch {epoch:03d} loss={history['loss'][-1]:.4f} acc={history['accuracy'][-1]:.4f}"
        if has_val:
            message += f" val_loss={history['val_loss'][-1]:.4f} val_acc={history['val_accuracy'][-1]:.4f}"
        return message


def prepare_mnist_images(which: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST split as normalized NCHW images."""
    images, labels = load_mnist_split(which)
    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]
    X = images[:, None, :, :].astype(np.float64) / 255.0
    y = labels.astype(np.int64)
    return X, y


def train_cnn(config: CNNConfig, verbose: bool = True) -> dict[str, Any]:
    X_train, y_train = prepare_mnist_images("train", limit=config.train_limit)
    X_val, y_val = prepare_mnist_images("val", limit=config.val_limit)
    model = SimpleMNISTCNN(
        filters=config.filters,
        kernel_size=config.kernel_size,
        learning_rate=config.learning_rate,
        seed=config.seed,
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_data=(X_val, y_val),
        verbose=verbose,
    )
    return {
        "dataset": "mnist",
        "model": "simple_cnn",
        "config": config.__dict__,
        "final_train_accuracy": history["accuracy"][-1],
        "final_val_accuracy": history["val_accuracy"][-1],
        "final_train_loss": history["loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "history": history,
    }


def save_metrics(metrics: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def _optional_limit(value: int) -> int | None:
    return None if value <= 0 else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small NumPy CNN on MNIST.")
    parser.add_argument("--filters", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=1000, help="Use <=0 for full MNIST train split.")
    parser.add_argument("--val-limit", type=int, default=1000, help="Use <=0 for full MNIST validation split.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "cnn_mnist_metrics.json")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = CNNConfig(
        filters=args.filters,
        kernel_size=args.kernel_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        train_limit=_optional_limit(args.train_limit),
        val_limit=_optional_limit(args.val_limit),
    )
    metrics = train_cnn(config, verbose=not args.quiet)
    save_metrics(metrics, args.output)
    print(
        f"cnn mnist: train_acc={metrics['final_train_accuracy']:.4f}, "
        f"val_acc={metrics['final_val_accuracy']:.4f}"
    )
    print(f"metrics saved to {args.output}")


if __name__ == "__main__":
    main()
